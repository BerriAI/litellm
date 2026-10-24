"""Unit tests for the Vercel AI Gateway and Friendli transforms in
`.github/scripts/auto_update_price_and_context_window_file.py`."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from litellm.litellm_core_utils.llm_cost_calc.utils import _get_token_base_cost
from litellm.types.utils import Usage

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "scripts"
    / "auto_update_price_and_context_window_file.py"
)


@pytest.fixture(scope="module")
def sync_module():
    spec = importlib.util.spec_from_file_location(
        "auto_update_price_and_context_window_file", SCRIPT_PATH
    )
    assert spec and spec.loader, f"Could not load spec for {SCRIPT_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["auto_update_price_and_context_window_file"] = module
    spec.loader.exec_module(module)
    return module


def _reasoning_model(**overrides: object) -> dict:
    model = {
        "id": "zai-org/GLM-Test",
        "base_model": "zhipuai/glm-test",
        "context_length": 1048576,
        "max_completion_tokens": 131072,
        "pricing": {"input": "0.00000015", "output": "0.0000005", "input_cache_read": "0.00000003"},
        "reasoning": True,
        "reasoning_options": [{"type": "effort", "values": ["max", "high", "low"]}],
        "functionality": {
            "tool_call": True,
            "parallel_tool_call": True,
            "structured_output": True,
            "system_messages": True,
            "tool_choice": True,
        },
        "input_modalities": ["text", "image", "video"],
        "mode": "chat",
    }
    model.update(overrides)
    return model


def test_transform_emits_declared_effort_levels_in_canonical_order(sync_module):
    entry = sync_module.transform_friendli_data([_reasoning_model()], {})[
        "friendliai/zai-org/GLM-Test"
    ]
    assert entry["supports_reasoning"] is True
    assert entry["reasoning_effort_levels"] == ["low", "high", "max"]
    assert not any(k.endswith("_reasoning_effort") for k in entry)


def test_transform_reasoning_model_without_effort_options_declares_empty_levels(sync_module):
    model = _reasoning_model(reasoning_options=[{"type": "budget_tokens", "values": []}])
    entry = sync_module.transform_friendli_data([model], {})["friendliai/zai-org/GLM-Test"]
    assert entry["reasoning_effort_levels"] == []


def test_transform_non_reasoning_model_declares_no_levels(sync_module):
    model = _reasoning_model(reasoning=False, reasoning_options=[])
    entry = sync_module.transform_friendli_data([model], {})["friendliai/zai-org/GLM-Test"]
    assert entry["supports_reasoning"] is False
    assert "reasoning_effort_levels" not in entry


def test_transform_max_tokens_mirrors_output_cap_not_context(sync_module):
    entry = sync_module.transform_friendli_data([_reasoning_model()], {})[
        "friendliai/zai-org/GLM-Test"
    ]
    assert entry["max_input_tokens"] == 1048576
    assert entry["max_output_tokens"] == 131072
    assert entry["max_tokens"] == entry["max_output_tokens"]


def test_transform_prompt_caching_follows_cache_pricing(sync_module):
    cached = sync_module.transform_friendli_data([_reasoning_model()], {})[
        "friendliai/zai-org/GLM-Test"
    ]
    assert cached["supports_prompt_caching"] is True
    assert cached["cache_read_input_token_cost"] == 3e-08

    uncached_model = _reasoning_model(pricing={"input": "0.00000014", "output": "0.0000004"})
    uncached = sync_module.transform_friendli_data([uncached_model], {})[
        "friendliai/zai-org/GLM-Test"
    ]
    assert uncached["supports_prompt_caching"] is False
    assert "cache_read_input_token_cost" not in uncached


def test_transform_modalities_set_vision_image_and_video_flags(sync_module):
    entry = sync_module.transform_friendli_data([_reasoning_model()], {})[
        "friendliai/zai-org/GLM-Test"
    ]
    assert entry["supports_vision"] is True
    assert entry["supports_image_input"] is True
    assert entry["supports_video_input"] is True

    text_only = _reasoning_model(input_modalities=["text"])
    entry_text = sync_module.transform_friendli_data([text_only], {})[
        "friendliai/zai-org/GLM-Test"
    ]
    assert entry_text["supports_vision"] is False
    assert entry_text["supports_image_input"] is False
    assert entry_text["supports_video_input"] is False


def test_transform_skips_rows_without_valid_token_prices_so_priced_local_entries_survive(sync_module):
    local = {
        "friendliai/zai-org/GLM-Test": {
            "litellm_provider": "friendliai",
            "input_cost_per_token": 1.5e-07,
            "output_cost_per_token": 5e-07,
        }
    }
    unpriced_rows = [
        _reasoning_model(pricing={}),
        _reasoning_model(pricing=None),
        _reasoning_model(pricing={"input": "0.00000015"}),
        _reasoning_model(pricing={"output": "0.0000005"}),
        _reasoning_model(pricing={"input": "not-a-number", "output": "0.0000005"}),
        _reasoning_model(pricing={"input": "-0.00000015", "output": "0.0000005"}),
        _reasoning_model(pricing={"input": "inf", "output": "0.0000005"}),
        _reasoning_model(pricing={"input": "nan", "output": "0.0000005"}),
    ]
    remote = sync_module.transform_friendli_data(unpriced_rows, local)
    assert remote == {}
    sync_module.sync_local_data_with_remote(local, remote, replace_keys=frozenset(remote))
    assert local["friendliai/zai-org/GLM-Test"]["input_cost_per_token"] == 1.5e-07
    assert local["friendliai/zai-org/GLM-Test"]["output_cost_per_token"] == 5e-07


def test_transform_keeps_zero_priced_rows(sync_module):
    free_model = _reasoning_model(pricing={"input": "0", "output": "0"})
    entry = sync_module.transform_friendli_data([free_model], {})["friendliai/zai-org/GLM-Test"]
    assert entry["input_cost_per_token"] == 0.0
    assert entry["output_cost_per_token"] == 0.0


def test_transforms_survive_failed_fetch(sync_module):
    assert sync_module.transform_friendli_data(None, {}) == {}
    assert sync_module.transform_friendli_data([], {}) == {}
    assert sync_module.transform_vercel_ai_gateway_data(None) == {}


VERCEL_LANGUAGE_ROW = {
    "id": "anthropic/claude-sonnet-4.5",
    "type": "language",
    "context_window": 200000,
    "max_tokens": 64000,
    "pricing": {
        "input": "0.000003",
        "output": "0.000015",
        "input_cache_read": "0.0000003",
        "input_cache_write": "0.00000375",
    },
}
VERCEL_EMBEDDING_ROW = {
    "id": "google/gemini-embedding-001",
    "type": "embedding",
    "context_window": 2048,
    "max_tokens": 0,
    "pricing": {"input": "0.00000015"},
}
VERCEL_UNPRICED_LANGUAGE_ROW = {
    "id": "perplexity/sonar",
    "type": "language",
    "context_window": 128000,
    "max_tokens": 8000,
}
VERCEL_VIDEO_ROW = {
    "id": "google/veo-3.1",
    "type": "video",
    "pricing": {"video_duration_pricing": [{"resolution": "720p", "price_per_second": "0.4"}]},
}
VERCEL_IMAGE_ROW = {"id": "openai/gpt-image-1", "type": "image", "pricing": {"input": "0.000005", "output": "0.00004"}}
VERCEL_NEGATIVE_PRICE_LANGUAGE_ROW = {
    "id": "router/auto",
    "type": "language",
    "context_window": 2000000,
    "max_tokens": 2000000,
    "pricing": {"input": "-1", "output": "-1"},
}


def test_vercel_transform_keeps_only_per_token_priced_language_and_embedding_rows(sync_module):
    result = sync_module.transform_vercel_ai_gateway_data(
        [
            VERCEL_LANGUAGE_ROW,
            VERCEL_EMBEDDING_ROW,
            VERCEL_UNPRICED_LANGUAGE_ROW,
            VERCEL_VIDEO_ROW,
            VERCEL_IMAGE_ROW,
            VERCEL_NEGATIVE_PRICE_LANGUAGE_ROW,
        ]
    )

    assert result == {
        "vercel_ai_gateway/anthropic/claude-sonnet-4.5": {
            "max_tokens": 64000,
            "max_input_tokens": 200000,
            "max_output_tokens": 64000,
            "input_cost_per_token": 3e-06,
            "output_cost_per_token": 1.5e-05,
            "cache_read_input_token_cost": 3e-07,
            "cache_creation_input_token_cost": 3.75e-06,
            "litellm_provider": "vercel_ai_gateway",
            "mode": "chat",
        },
        "vercel_ai_gateway/google/gemini-embedding-001": {
            "max_input_tokens": 2048,
            "input_cost_per_token": 1.5e-07,
            "output_cost_per_token": 0.0,
            "litellm_provider": "vercel_ai_gateway",
            "mode": "embedding",
        },
    }


@pytest.mark.parametrize("price", ["not-a-number", "nan", "inf"])
def test_vercel_transform_skips_a_row_whose_prices_do_not_parse_and_keeps_the_rest(sync_module, price):
    broken_row = {**VERCEL_LANGUAGE_ROW, "id": "broken/model", "pricing": {"input": price, "output": "0.000015"}}

    result = sync_module.transform_vercel_ai_gateway_data([broken_row, VERCEL_LANGUAGE_ROW])

    assert list(result) == ["vercel_ai_gateway/anthropic/claude-sonnet-4.5"]


def test_vercel_transform_omits_zero_context_and_output_limits(sync_module):
    row = {**VERCEL_EMBEDDING_ROW, "context_window": 0, "max_tokens": 0}

    entry = sync_module.transform_vercel_ai_gateway_data([row])["vercel_ai_gateway/google/gemini-embedding-001"]

    assert {"max_tokens", "max_input_tokens", "max_output_tokens"}.isdisjoint(entry)


VERCEL_LONG_CONTEXT_ROW = {
    "id": "openai/gpt-long",
    "type": "language",
    "context_window": 1050000,
    "max_tokens": 128000,
    "pricing": {
        "input": "0.000004",
        "input_tiers": [{"cost": "0.000004", "max": 272000}, {"cost": "0.000008", "min": 272000}],
        "output": "0.00002",
        "output_tiers": [{"cost": "0.00002", "min": 0, "max": 272001}, {"cost": "0.00003", "min": 272001}],
        "input_cache_read": "0.0000004",
        "service_tiers": {
            "priority": {
                "input": "0.000008",
                "output": "0.00004",
                "long_context": {"threshold": 272001, "input": "0.000016", "output": "0.00006"},
            },
            "unknown_tier": {"input": "0.1", "output": "0.1"},
        },
    },
}


@pytest.mark.parametrize(
    ("prompt_tokens", "service_tier", "expected_input_rate", "expected_output_rate"),
    [
        (272000, None, 4e-06, 2e-05),
        (272001, None, 8e-06, 3e-05),
        (272000, "priority", 8e-06, 4e-05),
        (272001, "priority", 1.6e-05, 6e-05),
    ],
)
def test_vercel_long_context_and_service_tier_prices_bill_at_the_catalog_rates(
    sync_module, prompt_tokens, service_tier, expected_input_rate, expected_output_rate
):
    entry = sync_module.transform_vercel_ai_gateway_data([VERCEL_LONG_CONTEXT_ROW])["vercel_ai_gateway/openai/gpt-long"]
    usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=10, total_tokens=prompt_tokens + 10)

    input_rate, output_rate, *_ = _get_token_base_cost(entry, usage, service_tier=service_tier)

    assert (input_rate, output_rate) == pytest.approx((expected_input_rate, expected_output_rate))


def test_vercel_transform_drops_service_tiers_litellm_cannot_bill(sync_module):
    entry = sync_module.transform_vercel_ai_gateway_data([VERCEL_LONG_CONTEXT_ROW])["vercel_ai_gateway/openai/gpt-long"]

    assert not [key for key in entry if "unknown_tier" in key]


def test_vercel_capability_tags_only_ever_turn_flags_on(sync_module):
    tagged = {**VERCEL_LANGUAGE_ROW, "tags": ["tool-use", "vision"]}
    local = {"vercel_ai_gateway/anthropic/claude-sonnet-4.5": {"supports_reasoning": True}}

    sync_module.sync_local_data_with_remote(local, sync_module.transform_vercel_ai_gateway_data([tagged]))

    synced = local["vercel_ai_gateway/anthropic/claude-sonnet-4.5"]
    assert synced["supports_function_calling"] is True
    assert synced["supports_vision"] is True
    assert synced["supports_reasoning"] is True
    assert "supports_response_schema" not in synced


def _write_cost_maps(tmp_path: Path) -> tuple[str, str]:
    root = tmp_path / "model_prices_and_context_window.json"
    backup = tmp_path / "model_prices_and_context_window_backup.json"
    original = json.dumps({"sample_spec": {"mode": "chat"}}, indent=4) + "\n"
    root.write_text(original, encoding="utf-8")
    backup.write_text(original, encoding="utf-8")
    return str(root), str(backup)


def test_main_fails_the_run_and_writes_nothing_when_a_catalog_fetch_fails(sync_module, tmp_path):
    paths = _write_cost_maps(tmp_path)
    before = [Path(path).read_text(encoding="utf-8") for path in paths]

    def fetch(url: str) -> list[dict[str, object]] | None:
        return None if url == sync_module.VERCEL_AI_GATEWAY_URL else []

    with pytest.raises(SystemExit) as exit_info:
        sync_module.main(fetch=fetch, paths=paths)

    assert exit_info.value.code not in (0, None)
    assert [Path(path).read_text(encoding="utf-8") for path in paths] == before


def test_main_writes_the_synced_map_to_both_cost_map_files(sync_module, tmp_path):
    paths = _write_cost_maps(tmp_path)

    def fetch(url: str) -> list[dict[str, object]] | None:
        return [VERCEL_LANGUAGE_ROW] if url == sync_module.VERCEL_AI_GATEWAY_URL else []

    sync_module.main(fetch=fetch, paths=paths)

    root, backup = (json.loads(Path(path).read_text(encoding="utf-8")) for path in paths)
    assert root == backup
    assert root["vercel_ai_gateway/anthropic/claude-sonnet-4.5"]["input_cost_per_token"] == 3e-06


def test_write_to_file_matches_the_checked_in_cost_map_format(sync_module, tmp_path):
    target = tmp_path / "model_prices_and_context_window.json"

    sync_module.write_to_file(str(target), {"vercel_ai_gateway/é": {"mode": "chat", "input_cost_per_token": 1e-06}})

    assert target.read_text(encoding="utf-8") == (
        '{\n    "vercel_ai_gateway/é": {\n        "mode": "chat",\n        "input_cost_per_token": 1e-06\n    }\n}\n'
    )


def test_sync_replaces_friendli_entries_so_dropped_cache_pricing_does_not_survive(sync_module):
    local = {
        "friendliai/zai-org/GLM-Test": {
            "litellm_provider": "friendliai",
            "cache_read_input_token_cost": 3e-08,
            "supports_prompt_caching": True,
        }
    }
    uncached_model = _reasoning_model(pricing={"input": "0.00000014", "output": "0.0000004"})
    remote = sync_module.transform_friendli_data([uncached_model], local)
    sync_module.sync_local_data_with_remote(local, remote, replace_keys=frozenset(remote))
    synced = local["friendliai/zai-org/GLM-Test"]
    assert "cache_read_input_token_cost" not in synced
    assert synced["supports_prompt_caching"] is False


def test_sync_still_merges_entries_outside_replace_keys(sync_module):
    local = {"openrouter/some-model": {"input_cost_per_token": 1e-06, "supports_vision": True}}
    remote = {"openrouter/some-model": {"input_cost_per_token": 2e-06}}
    sync_module.sync_local_data_with_remote(local, remote)
    assert local["openrouter/some-model"] == {"input_cost_per_token": 2e-06, "supports_vision": True}


def test_transform_inherits_allowlisted_keys_from_base_model_entry(sync_module):
    local = {
        "zhipuai/glm-test": {
            "supports_pdf_input": True,
            "supports_assistant_prefill": True,
            "input_cost_per_token": 9e-06,
        }
    }
    entry = sync_module.transform_friendli_data([_reasoning_model()], local)[
        "friendliai/zai-org/GLM-Test"
    ]
    assert entry["supports_pdf_input"] is True
    assert entry["supports_assistant_prefill"] is True
    assert entry["input_cost_per_token"] == 1.5e-07
