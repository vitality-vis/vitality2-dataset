# Export

Export all paper metadata from Zilliz `paper_new` (excludes `embedding`, `search_text`, and `search_sparse`).

Default output: `../data/zilliz/paper_new_metadata.jsonl`

Timeouts / transient query errors are retried (default: 5 retries, exponential backoff). If the iterator still fails, it is recreated and the export continues (use `--resume` so already-written rows are not duplicated).

## Commands

From this directory:

```bash
python3 export_paper_metadata.py --load
```

### Optional

```bash
# Smoke test
python3 export_paper_metadata.py --load --limit 10

# Resume / continue without overwriting (skips paper_uids already in the file;
# also merges any leftover *.jsonl.tmp from an interrupted run)
python3 export_paper_metadata.py --load --resume

# Custom output path
python3 export_paper_metadata.py --load --output ../data/zilliz/paper_new_metadata.jsonl

# Tune retries
python3 export_paper_metadata.py --load --resume --max-retries 8 --retry-backoff 3 --query-timeout 300
```
