"""One-time: pull the HF team-radio parquet shards locally (~2.57 GB) so the app
reads catalog/clips/audio from disk instead of the flaky datasets-server API."""
import os
from huggingface_hub import snapshot_download

DEST = os.environ.get("LOCAL_DATASET_DIR", os.path.join(".hfcache", "f1-team-radio"))
print(f"downloading MikCil/f1-team-radio -> {DEST} (~2.57 GB, one time)...", flush=True)
path = snapshot_download(
    "MikCil/f1-team-radio", repo_type="dataset", local_dir=DEST,
    allow_patterns=["data/*.parquet", "README.md"],
)
print("done:", path, flush=True)
for f in sorted(os.listdir(os.path.join(DEST, "data"))):
    fp = os.path.join(DEST, "data", f)
    print(f"  {f}  {os.path.getsize(fp)/1e6:.1f} MB", flush=True)
