#!/usr/bin/env python3
"""Download the latest DBLP XML dump and DTD.

The XML dump is downloaded as dblp.xml.gz, decompressed with streaming gzip,
and then swapped into data/dblp/dump/dblp.xml only after the new file is fully
written. This avoids loading the multi-GB dump into memory.
"""

from __future__ import annotations

import argparse
import gzip
import multiprocessing as mp
import queue
import shutil
import ssl
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Callable


DBLP_XML_GZ_URL = "https://dblp.org/xml/dblp.xml.gz"
DBLP_DTD_URL = "https://dblp.org/xml/dblp.dtd"
CHUNK_SIZE = 8 * 1024 * 1024
NETWORK_READ_CHUNK_SIZE = 1 * 1024 * 1024
DEFAULT_DOWNLOAD_WORKERS = 16
MAX_DOWNLOAD_WORKERS = 16
DEFAULT_SEGMENT_SIZE = 32 * 1024 * 1024
DOWNLOAD_ATTEMPTS = 4
URL_TIMEOUT_SECONDS = 30
STALL_RECONNECT_SECONDS = 45
RANGE_ATTEMPTS = 4


class RangeDownloadFailed(RuntimeError):
    """A single segment exhausted its retries; never restart the whole gzip file."""


def emit_progress(message: str, callback: Callable[[str], None] | None = None) -> None:
    if callback is None:
        print(message, file=sys.stderr, flush=True)
    else:
        callback(message)


def write_download_diagnostic(target: Path, message: str, *, clear: bool = False) -> None:
    """Keep retry causes available without adding noisy terminal output."""
    path = target.with_name(".dblp_download_diagnostics.log")
    if clear:
        path.unlink(missing_ok=True)
        return
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {message}\n")


def default_ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def content_length(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, context=default_ssl_context(), timeout=URL_TIMEOUT_SECONDS) as response:
        return int(response.headers.get("Content-Length") or 0)


def supports_range_requests(url: str) -> bool:
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, context=default_ssl_context(), timeout=URL_TIMEOUT_SECONDS) as response:
        return response.status == 206


def download_file(
    url: str,
    target: Path,
    *,
    workers: int = 1,
    segment_size: int = DEFAULT_SEGMENT_SIZE,
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    """Download with silent whole-request retries for transient TLS/network errors."""
    display_name = "dblp.xml.gz" if "xml" in target.name else "dblp.dtd"
    emit_progress(f"download starting: {display_name}", progress_callback)
    if "xml" in target.name:
        write_download_diagnostic(target, "", clear=True)
    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            download_file_once(
                url,
                target,
                workers=workers,
                segment_size=segment_size,
                progress_callback=progress_callback,
            )
            return
        except KeyboardInterrupt:
            raise
        except RangeDownloadFailed:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if target.exists():
                target.unlink()
            if attempt < DOWNLOAD_ATTEMPTS:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"download failed after {DOWNLOAD_ATTEMPTS} attempts: {last_error}") from last_error


def download_file_once(
    url: str,
    target: Path,
    *,
    workers: int = 1,
    segment_size: int = DEFAULT_SEGMENT_SIZE,
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    if workers > 1:
        try:
            total = content_length(url)
            if total > segment_size and supports_range_requests(url):
                download_file_parallel(
                    url,
                    target,
                    total=total,
                    workers=workers,
                    segment_size=segment_size,
                    progress_callback=progress_callback,
                )
                return
            emit_progress("server does not support ranged downloads; falling back to single connection", progress_callback)
        except RangeDownloadFailed:
            raise
        except Exception as exc:
            if target.exists():
                target.unlink()
            raise RuntimeError(f"segmented download setup failed: {exc}") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    started = time.monotonic()
    last_report = started

    with urllib.request.urlopen(
        url, context=default_ssl_context(), timeout=URL_TIMEOUT_SECONDS
    ) as response, target.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)

            now = time.monotonic()
            if now - last_report >= 5:
                if total:
                    pct = downloaded / total * 100
                    emit_progress(
                        f"downloaded {format_bytes(downloaded)} / {format_bytes(total)} ({pct:.1f}%)",
                        progress_callback,
                    )
                else:
                    emit_progress(f"downloaded {format_bytes(downloaded)}", progress_callback)
                last_report = now

    elapsed = max(time.monotonic() - started, 0.001)
    emit_progress(
        f"download complete: {target} ({format_bytes(downloaded)}, {format_bytes(downloaded / elapsed)}/s)",
        progress_callback,
    )


def download_range_worker(
    url: str,
    start: int,
    end: int,
    attempt: int,
    part_path_text: str,
    events,
) -> None:
    """Download one range in an isolated process so it can be force-stopped."""
    part_path = Path(part_path_text)
    expected = end - start + 1
    try:
        if part_path.exists():
            part_path.unlink()
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(
            request, context=default_ssl_context(), timeout=URL_TIMEOUT_SECONDS
        ) as response:
            if response.status != 206:
                raise RuntimeError(f"range request {start}-{end} returned HTTP {response.status}")
            actual = 0
            with part_path.open("wb") as part:
                while True:
                    chunk = response.read(NETWORK_READ_CHUNK_SIZE)
                    if not chunk:
                        break
                    part.write(chunk)
                    actual += len(chunk)
                    events.put(("progress", start, end, attempt, actual, ""))
        if actual != expected:
            raise RuntimeError(f"range request {start}-{end} wrote {actual} bytes; expected {expected}")
        events.put(("done", start, end, attempt, actual, ""))
    except Exception as exc:  # noqa: BLE001
        if part_path.exists():
            part_path.unlink()
        events.put(("error", start, end, attempt, 0, str(exc)))


def download_file_parallel(
    url: str,
    target: Path,
    *,
    total: int,
    workers: int,
    segment_size: int,
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    """Download bounded concurrent isolated segments; retry only failed segments."""
    target.parent.mkdir(parents=True, exist_ok=True)
    ranges = [
        (start, min(start + segment_size - 1, total - 1))
        for start in range(0, total, segment_size)
    ]
    started = time.monotonic()
    last_report = started
    with target.open("wb") as handle:
        handle.truncate(total)

    workers = min(max(1, workers), MAX_DOWNLOAD_WORKERS, len(ranges))
    range_progress = {range_key: 0 for range_key in ranges}

    def report_progress() -> None:
        nonlocal last_report
        now = time.monotonic()
        if now - last_report < 5:
            return
        downloaded = sum(range_progress.values())
        pct = downloaded / total * 100
        emit_progress(
            f"downloaded {format_bytes(downloaded)} / {format_bytes(total)} ({pct:.1f}%)",
            progress_callback,
        )
        last_report = now

    def stop_process(process: mp.Process) -> None:
        if process.is_alive():
            process.terminate()
        process.join(timeout=3)
        if process.is_alive():
            process.kill()
            process.join()

    emit_progress(
        f"segmented download: {format_bytes(total)} in {len(ranges)} segments with {workers} workers",
        progress_callback,
    )
    context = mp.get_context("spawn")
    events = context.Queue()
    running: dict[tuple[int, int], tuple[mp.Process, int]] = {}
    attempts = {range_key: 0 for range_key in ranges}
    waiting = list(ranges)
    range_last_progress = {range_key: time.monotonic() for range_key in ranges}
    try:
        def start_segment(range_key: tuple[int, int]) -> None:
            start, end = range_key
            attempts[range_key] += 1
            range_progress[range_key] = 0
            range_last_progress[range_key] = time.monotonic()
            part_path = target.with_name(f"{target.name}.part.{start}-{end}")
            part_path.unlink(missing_ok=True)
            process = context.Process(
                target=download_range_worker,
                args=(url, start, end, attempts[range_key], str(part_path), events),
            )
            process.start()
            running[range_key] = (process, attempts[range_key])

        def retry_or_fail(range_key: tuple[int, int], reason: str) -> None:
            process, _ = running.pop(range_key)
            stop_process(process)
            start, end = range_key
            target.with_name(f"{target.name}.part.{start}-{end}").unlink(missing_ok=True)
            number = ranges.index(range_key) + 1
            if attempts[range_key] >= RANGE_ATTEMPTS:
                write_download_diagnostic(
                    target,
                    f"segment={number}/{len(ranges)} attempt={attempts[range_key]} failed: {reason}",
                )
                raise RangeDownloadFailed(
                    f"segment {number}/{len(ranges)} ({start}-{end}) failed after "
                    f"{RANGE_ATTEMPTS} attempts: {reason}"
                )
            emit_progress(
                f"segment {number}/{len(ranges)} retrying ({attempts[range_key] + 1}/{RANGE_ATTEMPTS})",
                progress_callback,
            )
            write_download_diagnostic(
                target,
                f"segment={number}/{len(ranges)} attempt={attempts[range_key]} retrying: {reason}",
            )
            start_segment(range_key)

        while waiting and len(running) < workers:
            start_segment(waiting.pop(0))
        while running:
            try:
                kind, start, end, attempt, value, detail = events.get(timeout=1)
            except queue.Empty:
                now = time.monotonic()
                for range_key in list(running):
                    process, _ = running[range_key]
                    if now - range_last_progress[range_key] >= STALL_RECONNECT_SECONDS:
                        retry_or_fail(range_key, "no download progress")
                    elif not process.is_alive():
                        retry_or_fail(range_key, f"worker exited with code {process.exitcode}")
                continue
            range_key = (start, end)
            active = running.get(range_key)
            if active is None or active[1] != attempt:
                continue
            if kind == "progress":
                range_progress[range_key] = value
                range_last_progress[range_key] = time.monotonic()
                report_progress()
                continue
            if kind == "error":
                retry_or_fail(range_key, detail)
                continue
            if kind != "done":
                continue
            process, _ = running.pop(range_key)
            stop_process(process)
            expected = end - start + 1
            part_path = target.with_name(f"{target.name}.part.{start}-{end}")
            if not part_path.exists() or part_path.stat().st_size != expected:
                raise RuntimeError(f"segment {start}-{end} completed without a valid part file")
            with part_path.open("rb") as part, target.open("r+b") as handle:
                handle.seek(start)
                shutil.copyfileobj(part, handle, length=CHUNK_SIZE)
            part_path.unlink()
            range_progress[range_key] = expected
            report_progress()
            while waiting and len(running) < workers:
                start_segment(waiting.pop(0))
    except BaseException:
        for process, _ in running.values():
            stop_process(process)
        if target.exists():
            target.unlink()
        for part_path in target.parent.glob(f"{target.name}.part.*"):
            part_path.unlink()
        raise
    finally:
        events.close()
        events.join_thread()

    elapsed = max(time.monotonic() - started, 0.001)
    emit_progress(
        f"download complete: {target} ({format_bytes(total)}, {format_bytes(total / elapsed)}/s)",
        progress_callback,
    )


def replace_file(new_path: Path, final_path: Path) -> None:
    if final_path.exists():
        final_path.unlink()
    new_path.replace(final_path)


def cleanup_partial_downloads(output_dir: Path) -> int:
    removed = 0
    for pattern in (".dblp.*.tmp", ".dblp.*.part.*"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()
                removed += 1
    return removed


def download_dtd(
    output_dir: Path,
    dtd_url: str,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    final_path = output_dir / "dblp.dtd"
    with tempfile.NamedTemporaryFile(
        prefix=".dblp.", suffix=".dtd.tmp", dir=output_dir, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        download_file(dtd_url, tmp_path, progress_callback=progress_callback)
        replace_file(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    emit_progress(f"wrote {final_path}", progress_callback)
    return final_path


def download_and_decompress_xml(
    output_dir: Path,
    xml_gz_url: str,
    keep_gz: bool,
    *,
    download_workers: int,
    segment_size: int,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    final_xml = output_dir / "dblp.xml"
    final_gz = output_dir / "dblp.xml.gz"
    gz_tmp = output_dir / ".dblp.xml.gz.tmp"
    xml_tmp = output_dir / ".dblp.xml.tmp"

    for path in [gz_tmp, xml_tmp]:
        if path.exists():
            path.unlink()

    try:
        download_file(
            xml_gz_url,
            gz_tmp,
            workers=download_workers,
            segment_size=segment_size,
            progress_callback=progress_callback,
        )

        written = 0
        started = time.monotonic()
        last_report = started
        with gzip.open(gz_tmp, "rb") as src, xml_tmp.open("wb") as dst:
            while True:
                chunk = src.read(CHUNK_SIZE)
                if not chunk:
                    break
                dst.write(chunk)
                written += len(chunk)

                now = time.monotonic()
                if now - last_report >= 5:
                    emit_progress(f"decompressed {format_bytes(written)}", progress_callback)
                    last_report = now

        elapsed = max(time.monotonic() - started, 0.001)
        emit_progress(
            f"decompress complete: {format_bytes(written)} ({format_bytes(written / elapsed)}/s)",
            progress_callback,
        )

        replace_file(xml_tmp, final_xml)
        emit_progress(f"wrote {final_xml}", progress_callback)

        if keep_gz:
            replace_file(gz_tmp, final_gz)
            emit_progress(f"wrote {final_gz}", progress_callback)
        elif gz_tmp.exists():
            gz_tmp.unlink()

    finally:
        for path in [gz_tmp, xml_tmp]:
            if path.exists():
                path.unlink()

    return final_xml


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and unpack the latest DBLP dump.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/dblp/dump"),
        help="Directory where dblp.xml and dblp.dtd are stored.",
    )
    parser.add_argument("--xml-url", default=DBLP_XML_GZ_URL)
    parser.add_argument("--dtd-url", default=DBLP_DTD_URL)
    parser.add_argument(
        "--dtd-only",
        action="store_true",
        help="Only download dblp.dtd. Useful for a small network test.",
    )
    parser.add_argument(
        "--xml-only",
        action="store_true",
        help="Only download and decompress dblp.xml.gz.",
    )
    parser.add_argument(
        "--keep-gz",
        action="store_true",
        help="Keep the downloaded dblp.xml.gz next to dblp.xml.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=DEFAULT_DOWNLOAD_WORKERS,
        help="Concurrent XML segment workers. Values above 16 are capped.",
    )
    parser.add_argument(
        "--segment-size-mb",
        type=int,
        default=DEFAULT_SEGMENT_SIZE // 1024 // 1024,
        help="Segment size in MiB for concurrent XML gzip download.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dtd_only and args.xml_only:
        raise SystemExit("--dtd-only and --xml-only cannot be used together")
    if args.download_workers < 1:
        raise SystemExit("--download-workers must be >= 1")
    if args.segment_size_mb < 1:
        raise SystemExit("--segment-size-mb must be >= 1")

    try:
        if not args.xml_only:
            download_dtd(args.output_dir, args.dtd_url, progress_callback=getattr(args, "progress_callback", None))

        if not args.dtd_only:
            download_and_decompress_xml(
                args.output_dir,
                args.xml_url,
                args.keep_gz,
                download_workers=args.download_workers,
                segment_size=args.segment_size_mb * 1024 * 1024,
                progress_callback=getattr(args, "progress_callback", None),
            )
    except KeyboardInterrupt:
        removed = cleanup_partial_downloads(args.output_dir)
        emit_progress(
            f"Interrupted; removed {removed} partial download file(s).",
            getattr(args, "progress_callback", None),
        )
        return 130

    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
