"""SQLite cache for embeddings / LLM calls + JSONL cost ledger."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path


def cache_key(model: str, text: str) -> str:
    return hashlib.sha256((model + "\x00" + text).encode()).hexdigest()


class Cache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        with self._lock:
            row = self._conn.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def set(self, key: str, value) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, json.dumps(value)))
            self._conn.commit()

    def get_many(self, keys: list[str]) -> dict[str, object]:
        out: dict[str, object] = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                part = keys[i : i + 500]
                q = f"SELECT k, v FROM kv WHERE k IN ({','.join('?' * len(part))})"
                for k, v in self._conn.execute(q, part):
                    out[k] = json.loads(v)
        self.hits += len(out)
        self.misses += len(set(keys)) - len(out)
        return out

    def set_many(self, items: dict[str, object]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO kv VALUES (?,?)",
                [(k, json.dumps(v)) for k, v in items.items()],
            )
            self._conn.commit()


class CostLedger:
    """Append-only logs/costs.jsonl; keeps a running session total."""

    def __init__(self, log_dir: str | Path):
        self.path = Path(log_dir) / "costs.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_usd = 0.0

    def record(self, kind: str, model: str, tokens_in: int, tokens_out: int, usd: float) -> None:
        self.session_usd += usd
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "kind": kind,
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "usd": round(usd, 8),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def total_all_time(self) -> float:
        if not self.path.exists():
            return 0.0
        total = 0.0
        with open(self.path) as f:
            for line in f:
                try:
                    total += json.loads(line)["usd"]
                except (json.JSONDecodeError, KeyError):
                    continue
        return total
