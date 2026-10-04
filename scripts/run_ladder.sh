#!/usr/bin/env bash
# Run the experiment ladder: ingest (cached/free if the index exists) then eval each config with the judge.
# usage: scripts/run_ladder.sh 01_baseline 02_pymupdf_tables ...
set -u
PY=.venv/bin/python
for c in "$@"; do
  echo "=== $c: ingest"; $PY -u -m scripts.ingest --config configs/$c.yaml 2>&1 | grep -v -i -E "deprecat|pymupdf_layout"
  echo "=== $c: eval";   $PY -u -m eval.run_eval --config configs/$c.yaml --judge 2>&1 | grep -v -i -E "deprecat|pymupdf_layout" | tr '\r' '\n' | grep -v -E '^\s+[qc] [0-9]+/'
done
echo "=== LADDER DONE"
