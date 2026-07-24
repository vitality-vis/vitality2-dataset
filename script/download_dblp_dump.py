#!/usr/bin/env python3
"""Download the latest DBLP XML dump and DTD.

The XML dump is downloaded as dblp.xml.gz, decompressed with streaming gzip,
and then swapped into data/dblp/dump/dblp.xml only after the new file is fully
written. This avoids loading the multi-GB dump into memory.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import ssl
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DBLP_XML_GZ_URL = "https://dblp.org/xml/dblp.xml.gz"
DBLP_DTD_URL = "https://dblp.org/xml/dblp.dtd"
CHUNK_SIZE = 8 * 1024 * 1024
DEFAULT_DOWNLOAD_WORKERS = 16
DEFAULT_SEGMENT_SIZE = 32 * 1024 * 1024


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
    with urllib.request.urlopen(request, context=default_ssl_context()) as response:
        return int(response.headers.get("Content-Length") or 0)


def supports_range_requests(url: str) -> bool:
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, context=default_ssl_context()) as response:
        return response.status == 206


def download_file(url: str, target: Path, *, workers: int = 1, segment_size: int = DEFAULT_SEGMENT_SIZE) -> None:
    if workers > 1:
        try:
            total = content_length(url)
            if total > segment_size and supports_range_requests(url):
                download_file_parallel(url, target, total=total, workers=workers, segment_size=segment_size)
                return
            print("server does not support ranged downloads; falling back to single connection", file=sys.stderr)
        except Exception as exc:
            if target.exists():
                target.unlink()
            raise RuntimeError(f"ranged download failed: {exc}") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    started = time.monotonic()
    last_report = started

    with urllib.request.urlopen(url, context=default_ssl_context()) as response, target.open("wb") as handle:
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
                    print(
                        f"downloaded {format_bytes(downloaded)} / {format_bytes(total)} ({pct:.1f}%)",
                        file=sys.stderr,
                    )
                else:
                    print(f"downloaded {format_bytes(downloaded)}", file=sys.stderr)
                last_report = now

    elapsed = max(time.monotonic() - started, 0.001)
    print(
        f"download complete: {target} ({format_bytes(downloaded)}, {format_bytes(downloaded / elapsed)}/s)",
        file=sys.stderr,
    )


def download_file_parallel(
    url: str,
    target: Path,
    *,
    total: int,
    workers: int,
    segment_size: int,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    ranges = [
        (start, min(start + segment_size - 1, total - 1))
        for start in range(0, total, segment_size)
    ]
    workers = min(workers, len(ranges))
    downloaded = 0
    lock = threading.Lock()
    started = time.monotonic()
    last_report = started

    with target.open("wb") as handle:
        handle.truncate(total)

    def report_progress() -> None:
        nonlocal last_report
        now = time.monotonic()
        if now - last_report < 5:
            return
        pct = downloaded / total * 100
        print(
            f"downloaded {format_bytes(downloaded)} / {format_bytes(total)} ({pct:.1f}%)",
            file=sys.stderr,
        )
        last_report = now

    def download_range(start: int, end: int) -> None:
        nonlocal downloaded
        expected = end - start + 1
        part_path = target.with_name(f"{target.name}.part.{start}-{end}")
        last_error = ""

        for attempt in range(1, 5):
            if part_path.exists():
                part_path.unlink()
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, context=default_ssl_context()) as response:
                    if response.status != 206:
                        raise RuntimeError(f"range request {start}-{end} returned HTTP {response.status}")
                    actual = 0
                    with part_path.open("wb") as part:
                        while True:
                            chunk = response.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            part.write(chunk)
                            actual += len(chunk)
                    if actual != expected:
                        raise RuntimeError(
                            f"range request {start}-{end} wrote {actual} bytes; expected {expected}"
                        )

                with part_path.open("rb") as part, target.open("r+b") as handle:
                    handle.seek(start)
                    shutil.copyfileobj(part, handle, length=CHUNK_SIZE)
                part_path.unlink()
                with lock:
                    downloaded += expected
                    report_progress()
                return
            except Exception as exc:
                last_error = str(exc)
                if attempt < 4:
                    print(
                        f"range {start}-{end} failed on attempt {attempt}/4: {last_error[:180]}; retrying",
                        file=sys.stderr,
                    )
                    time.sleep(1.5 * attempt)

        if part_path.exists():
            part_path.unlink()
        raise RuntimeError(f"range request {start}-{end} failed after retries: {last_error}")

    print(
        f"ranged download: {format_bytes(total)} in {len(ranges)} segments with {workers} workers",
        file=sys.stderr,
    )
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(download_range, start, end) for start, end in ranges]
            for future in as_completed(futures):
                future.result()
    except BaseException:
        if target.exists():
            target.unlink()
        for part_path in target.parent.glob(f"{target.name}.part.*"):
            part_path.unlink()
        raise

    elapsed = max(time.monotonic() - started, 0.001)
    print(
        f"download complete: {target} ({format_bytes(total)}, {format_bytes(total / elapsed)}/s)",
        file=sys.stderr,
    )


def replace_file(new_path: Path, final_path: Path) -> None:
    if final_path.exists():
        final_path.unlink()
    new_path.replace(final_path)


def download_dtd(output_dir: Path, dtd_url: str) -> Path:
    final_path = output_dir / "dblp.dtd"
    with tempfile.NamedTemporaryFile(
        prefix=".dblp.", suffix=".dtd.tmp", dir=output_dir, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        download_file(dtd_url, tmp_path)
        replace_file(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    print(f"wrote {final_path}", file=sys.stderr)
    return final_path


def download_and_decompress_xml(
    output_dir: Path,
    xml_gz_url: str,
    keep_gz: bool,
    *,
    download_workers: int,
    segment_size: int,
) -> Path:
    final_xml = output_dir / "dblp.xml"
    final_gz = output_dir / "dblp.xml.gz"
    gz_tmp = output_dir / ".dblp.xml.gz.tmp"
    xml_tmp = output_dir / ".dblp.xml.tmp"

    for path in [gz_tmp, xml_tmp]:
        if path.exists():
            path.unlink()

    try:
        download_file(xml_gz_url, gz_tmp, workers=download_workers, segment_size=segment_size)

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
                    print(f"decompressed {format_bytes(written)}", file=sys.stderr)
                    last_report = now

        elapsed = max(time.monotonic() - started, 0.001)
        print(
            f"decompress complete: {format_bytes(written)} ({format_bytes(written / elapsed)}/s)",
            file=sys.stderr,
        )

        replace_file(xml_tmp, final_xml)
        print(f"wrote {final_xml}", file=sys.stderr)

        if keep_gz:
            replace_file(gz_tmp, final_gz)
            print(f"wrote {final_gz}", file=sys.stderr)
        elif gz_tmp.exists():
            gz_tmp.unlink()

    finally:
        for path in [gz_tmp, xml_tmp]:
            if path.exists():
                path.unlink()

    return final_xml


def parse_args() -> argparse.Namespace:
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
        help="Concurrent range-download workers for dblp.xml.gz. Use 1 for a single connection.",
    )
    parser.add_argument(
        "--segment-size-mb",
        type=int,
        default=DEFAULT_SEGMENT_SIZE // 1024 // 1024,
        help="Range segment size in MiB for concurrent XML gzip download.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dtd_only and args.xml_only:
        raise SystemExit("--dtd-only and --xml-only cannot be used together")
    if args.download_workers < 1:
        raise SystemExit("--download-workers must be >= 1")
    if args.segment_size_mb < 1:
        raise SystemExit("--segment-size-mb must be >= 1")

    if not args.xml_only:
        download_dtd(args.output_dir, args.dtd_url)

    if not args.dtd_only:
        download_and_decompress_xml(
            args.output_dir,
            args.xml_url,
            args.keep_gz,
            download_workers=args.download_workers,
            segment_size=args.segment_size_mb * 1024 * 1024,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
