"""python -m scripts.ingest --config configs/x.yaml [--force]"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from app.cache import Cache, CostLedger
from app.config import load_config
from app.index import build_index


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--force", action="store_true", help="rebuild even if the index exists"
    )
    a = ap.parse_args()
    cfg = load_config(a.config)
    ledger = CostLedger(cfg.path("log_dir"))
    build_index(cfg, Cache(cfg.path("cache_path")), ledger, force=a.force)
    print(f"all-time spend so far: ${ledger.total_all_time():.4f}")


if __name__ == "__main__":
    main()
