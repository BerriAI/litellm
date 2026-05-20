#!/usr/bin/env python3
"""Extract the OpenAPI spec from the LiteLLM proxy FastAPI app.

Run from the repository root:
    uv run python sdk/litellm-ai/scripts/extract-openapi.py
"""
import json
import os
import sys

os.environ.setdefault("LITELLM_TELEMETRY", "False")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # pragma: allowlist secret

from litellm.proxy.proxy_server import app

schema = app.openapi()

output_path = os.path.join(os.path.dirname(__file__), "..", "openapi.json")
with open(output_path, "w") as f:
    json.dump(schema, f, indent=2)

print(
    f"Wrote OpenAPI spec to {output_path}: "
    f"{len(schema.get('paths', {}))} paths, "
    f"{len(schema.get('components', {}).get('schemas', {}))} schemas"
)
