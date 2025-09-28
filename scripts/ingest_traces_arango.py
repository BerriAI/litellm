#!/usr/bin/env python3
"""
Ingest mini-agent trace JSONL into ArangoDB (optional utility).

Reads MINI_AGENT_STORE_PATH (JSONL), writes to an ArangoDB collection.

Env:
  ARANGO_URL (e.g., http://127.0.0.1:8529)
  ARANGO_DB (e.g., litellm)
  ARANGO_USER / ARANGO_PASS
  MINI_AGENT_STORE_PATH (path to JSONL emitted by agent with MINI_AGENT_STORE_TRACES=1)

Usage:
  python scripts/ingest_traces_arango.py --collection agent_runs
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    try:
        from arango import ArangoClient  # type: ignore
    except Exception:
        print("python-arango not installed. pip install python-arango", file=sys.stderr)
        return 2

    apath = os.getenv("MINI_AGENT_STORE_PATH")
    if not apath or not os.path.exists(apath):
        print("MINI_AGENT_STORE_PATH missing or not found", file=sys.stderr)
        return 1

    url = os.getenv("ARANGO_URL", "http://127.0.0.1:8529")
    dbname = os.getenv("ARANGO_DB", "litellm")
    user = os.getenv("ARANGO_USER", "root")
    pw = os.getenv("ARANGO_PASS", "")

    p = argparse.ArgumentParser()
    p.add_argument("--collection", default="agent_runs")
    args = p.parse_args()

    client = ArangoClient(hosts=url)
    sys_db = client.db("_system", username=user, password=pw)
    if not sys_db.has_database(dbname):
        sys_db.create_database(dbname)
    db = client.db(dbname, username=user, password=pw)
    colname = args.collection
    if not db.has_collection(colname):
        db.create_collection(colname)
    col = db.collection(colname)

    # Ingest JSONL (store only keys we care about + trace)
    n = 0
    with open(apath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            doc = {
                "ts": rec.get("metrics", {}).get("ttotal_ms"),  # placeholder if you add explicit ts later
                "model": rec.get("metrics", {}).get("used_model"),
                "ok": rec.get("ok"),
                "stopped_reason": rec.get("stopped_reason"),
                "metrics": rec.get("metrics"),
                "trace": rec.get("trace"),
                "final_answer_len": len((rec.get("final_answer") or "")),
            }
            try:
                col.insert(doc)
                n += 1
            except Exception as e:
                print(f"insert failed: {e}", file=sys.stderr)
                continue
    print(f"ingested {n} records into {dbname}/{colname}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

