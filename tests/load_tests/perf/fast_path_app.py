"""ASGI entry point: LiteLLM proxy wrapped in the Rust fast-path middleware.

Use with any ASGI server, e.g.:

    CONFIG_FILE_PATH=path/to/config.yaml \\
    PYTHONPATH=/repo:/repo/tests/load_tests/perf \\
    uvicorn fast_path_app:app --port 4001 --workers 4

The mock_map is built from `model_list[*].litellm_params.mock_response` in the
config YAML. Any model without a configured mock_response falls through to the
normal LiteLLM pipeline unchanged.
"""
from __future__ import annotations

import os
from typing import Dict

import yaml

from fast_path_middleware import FastPathMiddleware
from litellm.proxy.proxy_server import app as _inner_app


def _load_mock_map(config_path: str) -> Dict[str, str]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    out: Dict[str, str] = {}
    for entry in cfg.get("model_list", []):
        name = entry.get("model_name")
        mock = (entry.get("litellm_params") or {}).get("mock_response")
        if name and mock:
            out[name] = mock
    return out


_config_path = os.environ.get("CONFIG_FILE_PATH")
_mock_map: Dict[str, str] = _load_mock_map(_config_path) if _config_path else {}

print(
    f"[fast_path_app] mock_map has {len(_mock_map)} entries: {list(_mock_map)}",
    flush=True,
)

app = FastPathMiddleware(_inner_app, _mock_map)
