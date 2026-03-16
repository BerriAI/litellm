#!/usr/bin/env python3
"""
Create a LiteLLM-managed OpenAI vector store from CLAUDE.md.
Requires OPENAI_API_KEY in environment or .env.

Usage:
  python scripts/create_vector_store.py

Output: vector store ID (vs_xxx) to use in file_search tools.
"""

import os
import sys
from pathlib import Path

# Load .env if present
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if " #" in v:
                v = v.split(" #")[0].strip()
            os.environ[k.strip()] = v

from openai import OpenAI

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not set. Add it to .env or export it.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(api_key=api_key)
repo_root = Path(__file__).resolve().parent.parent
doc_path = repo_root / "CLAUDE.md"

if not doc_path.exists():
    print(f"ERROR: {doc_path} not found", file=sys.stderr)
    sys.exit(1)

print("Uploading CLAUDE.md...")
with open(doc_path, "rb") as f:
    file = client.files.create(file=f, purpose="assistants")
file_id = file.id
print(f"File ID: {file_id}")

print("Creating vector store...")
vs = client.vector_stores.create(
    name="LiteLLM Docs Store",
    file_ids=[file_id],
)
print(f"Vector Store ID: {vs.id}")
print()
print("=== Use this ID in file_search ===")
print(vs.id)
