"""Regression tests for the FriendliAI serverless catalog entries.

The catalog previously carried only ``friendliai/meta-llama-3.1-70b-instruct``
and ``friendliai/meta-llama-3.1-8b-instruct``, both retired and no longer
served by ``https://api.friendli.ai/serverless/v1/models``. These tests pin the
current serverless lineup, assert the retired entries stay gone, and assert the
root JSON and the bundled backup JSON never drift apart for this provider.
"""

import json
from pathlib import Path
from typing import Final, NamedTuple

import pytest


class FriendliModel(NamedTuple):
    key: str
    input_cost_per_token: float
    output_cost_per_token: float
    cache_read_input_token_cost: float | None
    max_input_tokens: int
    max_output_tokens: int


EXPECTED_MODELS: Final[tuple[FriendliModel, ...]] = (
    FriendliModel(
        "friendliai/LGAI-EXAONE/K-EXAONE-236B-A23B", 2e-07, 8e-07, 1e-07, 262144, 262144
    ),
    FriendliModel(
        "friendliai/MiniMaxAI/MiniMax-M2.5", 3e-07, 1.2e-06, 6e-08, 196608, 196608
    ),
    FriendliModel(
        "friendliai/Qwen/Qwen3-235B-A22B-Instruct-2507", 2e-07, 8e-07, None, 262144, 262144
    ),
    FriendliModel(
        "friendliai/deepseek-ai/DeepSeek-V3.2", 5e-07, 1.5e-06, 2.5e-07, 163840, 163840
    ),
    FriendliModel(
        "friendliai/google/gemma-4-31B-it", 1.4e-07, 4e-07, None, 262144, 262144
    ),
    FriendliModel(
        "friendliai/zai-org/GLM-5.1", 1.4e-06, 4.4e-06, 2.6e-07, 202752, 202752
    ),
    FriendliModel(
        "friendliai/zai-org/GLM-5.2", 1.4e-06, 4.4e-06, 2.6e-07, 1048576, 1048576
    ),
)

RETIRED_KEYS: Final[tuple[str, ...]] = (
    "friendliai/meta-llama-3.1-70b-instruct",
    "friendliai/meta-llama-3.1-8b-instruct",
)

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
ROOT_PATH: Final[Path] = _REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH: Final[Path] = (
    _REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"
)
CATALOG_PATHS: Final[tuple[Path, ...]] = (ROOT_PATH, BACKUP_PATH)


def _load(path: Path) -> dict[str, dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", CATALOG_PATHS, ids=lambda p: p.name)
@pytest.mark.parametrize("expected", EXPECTED_MODELS, ids=lambda m: m.key)
def test_serverless_model_pricing_and_context(path: Path, expected: FriendliModel) -> None:
    entry = _load(path).get(expected.key)
    assert entry is not None, f"{expected.key} missing from {path.name}"
    assert entry["litellm_provider"] == "friendliai"
    assert entry["mode"] == "chat"
    assert entry["input_cost_per_token"] == expected.input_cost_per_token
    assert entry["output_cost_per_token"] == expected.output_cost_per_token
    assert entry.get("cache_read_input_token_cost") == expected.cache_read_input_token_cost
    assert entry["max_input_tokens"] == expected.max_input_tokens
    assert entry["max_output_tokens"] == expected.max_output_tokens
    assert entry["max_tokens"] == expected.max_output_tokens


@pytest.mark.parametrize("path", CATALOG_PATHS, ids=lambda p: p.name)
@pytest.mark.parametrize("expected", EXPECTED_MODELS, ids=lambda m: m.key)
def test_serverless_models_advertise_tool_calling(
    path: Path, expected: FriendliModel
) -> None:
    entry = _load(path)[expected.key]
    assert entry["supports_function_calling"] is True
    assert entry["supports_parallel_function_calling"] is True
    assert entry["supports_response_schema"] is True
    assert entry["supports_tool_choice"] is True


@pytest.mark.parametrize("path", CATALOG_PATHS, ids=lambda p: p.name)
@pytest.mark.parametrize("key", RETIRED_KEYS)
def test_retired_llama_entries_absent(path: Path, key: str) -> None:
    assert key not in _load(path), f"{key} is retired and must not be re-added"


def test_prompt_caching_flag_tracks_cache_read_cost() -> None:
    catalog = _load(ROOT_PATH)
    for expected in EXPECTED_MODELS:
        entry = catalog[expected.key]
        assert entry.get("supports_prompt_caching", False) is (
            expected.cache_read_input_token_cost is not None
        ), f"{expected.key} prompt-caching flag disagrees with cache_read cost"


def test_deprecating_model_carries_iso_date() -> None:
    entry = _load(ROOT_PATH)["friendliai/Qwen/Qwen3-235B-A22B-Instruct-2507"]
    assert entry["deprecation_date"] == "2026-08-05"


def test_root_and_backup_agree_on_every_friendliai_entry() -> None:
    root = {k: v for k, v in _load(ROOT_PATH).items() if k.startswith("friendliai/")}
    backup = {k: v for k, v in _load(BACKUP_PATH).items() if k.startswith("friendliai/")}
    assert root == backup


def test_catalog_covers_exactly_the_expected_lineup() -> None:
    keys = {k for k in _load(ROOT_PATH) if k.startswith("friendliai/")}
    assert keys == {m.key for m in EXPECTED_MODELS}
