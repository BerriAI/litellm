"""Tests for .github/scripts/auto_update_price_and_context_window_file.py."""

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Final

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_MODULE_PATH: Final = _REPO_ROOT / ".github" / "scripts" / "auto_update_price_and_context_window_file.py"
_spec: Final = importlib.util.spec_from_file_location("auto_update_price_and_context_window_file", _MODULE_PATH)
script: Final = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = script
_spec.loader.exec_module(script)

_LOCAL_FILE: Final = "model_prices_and_context_window.json"
_GENERATED_AT: Final = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def _openrouter_row(model_id: str) -> dict:
    return {"id": model_id, "context_length": 8192, "pricing": {"prompt": "0.000001", "completion": "0.000002"}}


def _serve(openrouter_rows: list) -> object:
    async def fetch_data(url: str) -> list:
        return openrouter_rows if "openrouter" in url else []

    return fetch_data


def _read_local(tmp_path: Path) -> dict:
    return json.loads((tmp_path / _LOCAL_FILE).read_text())


def test_main_stamps_provenance_only_when_the_sync_changed_the_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "feedface")
    monkeypatch.setattr(script, "fetch_data", _serve([_openrouter_row("acme/x")]))
    (tmp_path / _LOCAL_FILE).write_text(json.dumps({"sample_spec": {"input_cost_per_token": "USD"}}, indent=4) + "\n")

    script.main()

    written = _read_local(tmp_path)
    assert written["openrouter/acme/x"]["litellm_provider"] == "openrouter"
    assert written["_metadata"]["source_revision"] == "feedface"
    assert _GENERATED_AT.fullmatch(written["_metadata"]["generated_at"])

    sentinel = {**written, "_metadata": {**written["_metadata"], "generated_at": "2000-01-01T00:00:00Z"}}
    (tmp_path / _LOCAL_FILE).write_text(json.dumps(sentinel, indent=4) + "\n")

    script.main()

    assert _read_local(tmp_path) == sentinel
