import importlib.util
import json
from pathlib import Path
from types import MappingProxyType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "sync_together_ai_models.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "together_ai_sync"

_spec = importlib.util.spec_from_file_location("sync_together_ai_models", SCRIPT)
assert _spec is not None and _spec.loader is not None
sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync)

RECORDED_CATALOG = sync.load_catalog(FIXTURES.joinpath("models_serverless.json").read_bytes())
RECORDED_DOC = sync.parse_deprecations(FIXTURES.joinpath("deprecations.md").read_text())


def _doc(removal_dates: dict[str, str], redirects: dict[str, str] | None = None) -> object:
    return sync.DeprecationDoc(
        removal_dates=MappingProxyType(removal_dates),
        redirects=MappingProxyType(redirects or {}),
    )


def _chat_model(model_id: str, ctx: int = 4096, price: float = 1.0, cached: float | None = None) -> object:
    return sync.CatalogModel(
        id=model_id,
        type="chat",
        context_length=ctx,
        pricing=sync.CatalogPricing(input=price, output=price, cached_input=cached),
    )


@pytest.mark.parametrize(
    ("per_million", "expected"),
    [
        (3, 3e-06),
        (15, 1.5e-05),
        (1.4, 1.4e-06),
        (0.25999999999999995, 2.6e-07),
        (0.060000000000000005, 6e-08),
        (1.0399999999999998, 1.04e-06),
        (0, 0.0),
    ],
)
def test_per_token_normalizes_float_artifacts(per_million: float, expected: float) -> None:
    assert sync.per_token(per_million) == expected


def test_parse_deprecations_recorded_fixture() -> None:
    assert dict(RECORDED_DOC.redirects) == {
        "mistralai/Mistral-7B-Instruct-v0.3": "mistralai/Ministral-3-14B-Instruct-2512",
        "Kimi-K2": "Kimi-K2-0905",
        "DeepSeek-V3": "DeepSeek-V3.1",
        "DeepSeek-V3-0324": "DeepSeek-V3.1",
        "DeepSeek-R1": "DeepSeek-R1-0528",
    }
    assert len(RECORDED_DOC.removal_dates) == 208
    assert RECORDED_DOC.removal_dates["google/gemma-3n-E4B-it"] == "2026-08-04"


def test_parse_deprecations_duplicate_rows_keep_most_recent_date() -> None:
    assert RECORDED_DOC.removal_dates["Qwen/Qwen3-235B-A22B-Thinking-2507"] == "2026-04-16"


@pytest.mark.parametrize(
    "markdown",
    [
        "# Deprecations\n\nNothing here anymore.\n",
        "\n## Active model redirects\n\n| A | B |\n| --- | --- |\n| `x` | `y` |\n\n## Something else\n",
        "\n## Deprecation history\n\n### Inference\n\n| Date | Model | R |\n| --- | --- | --- |\n| 2026-01-01 | `m` | No |\n",
    ],
)
def test_parse_deprecations_raises_when_a_table_parses_empty(markdown: str) -> None:
    with pytest.raises(sync.SyncError):
        sync.parse_deprecations(markdown)


def test_load_catalog_raises_on_shape_change() -> None:
    with pytest.raises(sync.SyncError):
        sync.load_catalog(b'[{"id": "x", "type": "chat"}]')


def test_load_catalog_raises_when_no_token_models_remain() -> None:
    only_video = json.dumps([{"id": "v", "type": "video", "pricing": {"input": 0, "output": 0}}]).encode()
    with pytest.raises(sync.SyncError):
        sync.load_catalog(only_video)


def test_recorded_catalog_counts() -> None:
    assert len(RECORDED_CATALOG) == 102
    assert sum(1 for model in RECORDED_CATALOG if model.type in sync.TYPE_TO_MODE) == 26
    assert sum(1 for model in RECORDED_CATALOG if model.pricing.cached_input) == 13


def test_added_chat_model_matches_reviewed_registry_shape() -> None:
    outcome = sync.compute_sync({}, RECORDED_CATALOG, RECORDED_DOC)
    assert len(outcome.added) == 26
    assert not outcome.deprecated
    assert outcome.cost_map["together_ai/moonshotai/Kimi-K3"] == {
        "cache_read_input_token_cost": 3e-07,
        "input_cost_per_token": 3e-06,
        "litellm_provider": "together_ai",
        "max_input_tokens": 1048576,
        "max_tokens": 1048576,
        "mode": "chat",
        "output_cost_per_token": 1.5e-05,
        "source": "https://docs.together.ai/docs/serverless-models",
        "supports_function_calling": True,
        "supports_parallel_function_calling": True,
        "supports_prompt_caching": True,
        "supports_reasoning": True,
        "supports_response_schema": True,
        "supports_tool_choice": True,
        "supports_vision": True,
    }


def test_added_embedding_model_has_no_output_token_cap() -> None:
    outcome = sync.compute_sync({}, RECORDED_CATALOG, RECORDED_DOC)
    assert outcome.cost_map["together_ai/intfloat/multilingual-e5-large-instruct"] == {
        "input_cost_per_token": 2e-08,
        "litellm_provider": "together_ai",
        "max_input_tokens": 514,
        "max_tokens": 514,
        "mode": "embedding",
        "output_cost_per_token": 2e-08,
        "output_vector_size": 1024,
        "source": "https://docs.together.ai/docs/serverless-models",
    }


def test_moderation_type_maps_to_chat_mode() -> None:
    outcome = sync.compute_sync({}, RECORDED_CATALOG, RECORDED_DOC)
    guard = outcome.cost_map["together_ai/meta-llama/Llama-Guard-4-12B"]
    assert guard["mode"] == "chat"
    assert "max_output_tokens" not in guard


def test_output_ceiling_comes_from_the_rule_never_from_context_length() -> None:
    glm = next(model for model in RECORDED_CATALOG if model.id == "zai-org/GLM-5.2")
    fresh = sync.compute_sync({}, [_chat_model("acme/unreviewed", ctx=1048576), glm], _doc({"x": "2026-01-01"}))
    unreviewed = fresh.cost_map["together_ai/acme/unreviewed"]
    assert "max_output_tokens" not in unreviewed
    assert (unreviewed["max_input_tokens"], unreviewed["max_tokens"]) == (1048576, 1048576)
    reviewed = fresh.cost_map["together_ai/zai-org/GLM-5.2"]
    assert (reviewed["max_input_tokens"], reviewed["max_output_tokens"], reviewed["max_tokens"]) == (
        1048575,
        128000,
        128000,
    )
    inflated = {
        "together_ai/zai-org/GLM-5.2": {
            "input_cost_per_token": 1.4e-06,
            "litellm_provider": "together_ai",
            "max_input_tokens": 1048575,
            "max_output_tokens": 1048575,
            "max_tokens": 1048575,
            "mode": "chat",
            "output_cost_per_token": 4.4e-06,
        }
    }
    corrected = sync.compute_sync(inflated, [glm], _doc({"x": "2026-01-01"}))
    assert corrected.cost_map["together_ai/zai-org/GLM-5.2"]["max_output_tokens"] == 128000
    assert any("max_output_tokens: 1048575 -> 128000" in line for line in corrected.updated)


def test_docs_removed_but_live_model_stays_live_with_warning() -> None:
    outcome = sync.compute_sync({}, RECORDED_CATALOG, RECORDED_DOC)
    gemma = outcome.cost_map["together_ai/google/gemma-3n-E4B-it"]
    assert "deprecation_date" not in gemma
    assert any("gemma-3n-E4B-it" in warning and "2026-08-04" in warning for warning in outcome.warnings)


def test_price_change_updates_api_fields_and_keeps_curated_ones() -> None:
    registry = {
        "together_ai/acme/chat-1": {
            "input_cost_per_token": 9e-07,
            "litellm_provider": "together_ai",
            "max_input_tokens": 4096,
            "max_output_tokens": 2048,
            "max_tokens": 2048,
            "mode": "chat",
            "output_cost_per_token": 9e-07,
            "supports_audio_input": True,
        }
    }
    outcome = sync.compute_sync(registry, [_chat_model("acme/chat-1", ctx=8192, price=2.0)], _doc({"x": "2026-01-01"}))
    entry = outcome.cost_map["together_ai/acme/chat-1"]
    assert entry["input_cost_per_token"] == 2e-06
    assert entry["max_input_tokens"] == 8192
    assert entry["max_output_tokens"] == 2048
    assert entry["supports_audio_input"] is True
    assert len(outcome.updated) == 1
    assert "input_cost_per_token" in outcome.updated[0]


def test_cached_input_appearing_and_disappearing() -> None:
    registry = {
        "together_ai/acme/chat-1": {
            "cache_read_input_token_cost": 1e-07,
            "input_cost_per_token": 1e-06,
            "litellm_provider": "together_ai",
            "max_input_tokens": 4096,
            "mode": "chat",
            "output_cost_per_token": 1e-06,
            "supports_prompt_caching": True,
        },
        "together_ai/acme/chat-2": {
            "input_cost_per_token": 1e-06,
            "litellm_provider": "together_ai",
            "max_input_tokens": 4096,
            "mode": "chat",
            "output_cost_per_token": 1e-06,
        },
    }
    catalog = [_chat_model("acme/chat-1"), _chat_model("acme/chat-2", cached=0.25999999999999995)]
    outcome = sync.compute_sync(registry, catalog, _doc({"x": "2026-01-01"}))
    assert "cache_read_input_token_cost" not in outcome.cost_map["together_ai/acme/chat-1"]
    assert "supports_prompt_caching" not in outcome.cost_map["together_ai/acme/chat-1"]
    assert outcome.cost_map["together_ai/acme/chat-2"]["cache_read_input_token_cost"] == 2.6e-07
    assert outcome.cost_map["together_ai/acme/chat-2"]["supports_prompt_caching"] is True


def test_capability_rule_backfills_existing_entry() -> None:
    registry = {
        "together_ai/moonshotai/Kimi-K3": {
            "input_cost_per_token": 3e-06,
            "litellm_provider": "together_ai",
            "max_input_tokens": 1048576,
            "mode": "chat",
            "output_cost_per_token": 1.5e-05,
        }
    }
    kimi = next(model for model in RECORDED_CATALOG if model.id == "moonshotai/Kimi-K3")
    outcome = sync.compute_sync(registry, [kimi], _doc({"x": "2026-01-01"}))
    assert outcome.cost_map["together_ai/moonshotai/Kimi-K3"]["supports_reasoning"] is True
    assert any("supports_reasoning" in line for line in outcome.updated)


def test_disappeared_model_gets_docs_date_and_is_never_deleted() -> None:
    registry = {
        "together_ai/acme/gone": {
            "input_cost_per_token": 1e-06,
            "litellm_provider": "together_ai",
            "mode": "chat",
            "output_cost_per_token": 1e-06,
        }
    }
    outcome = sync.compute_sync(registry, [_chat_model("acme/alive")], _doc({"acme/gone": "2026-07-01"}))
    assert outcome.cost_map["together_ai/acme/gone"]["deprecation_date"] == "2026-07-01"
    assert outcome.deprecated == ("together_ai/acme/gone: deprecation_date",)


def test_disappeared_model_without_docs_date_warns_instead() -> None:
    registry = {
        "together_ai/acme/gone": {
            "input_cost_per_token": 1e-06,
            "litellm_provider": "together_ai",
            "mode": "chat",
            "output_cost_per_token": 1e-06,
        }
    }
    outcome = sync.compute_sync(registry, [_chat_model("acme/alive")], _doc({"other": "2026-07-01"}))
    assert "deprecation_date" not in outcome.cost_map["together_ai/acme/gone"]
    assert not outcome.deprecated
    assert any("acme/gone" in warning and "human" in warning for warning in outcome.warnings)


def test_curated_deprecation_date_is_never_overwritten() -> None:
    registry = {
        "together_ai/acme/gone": {
            "deprecation_date": "2026-06-15",
            "input_cost_per_token": 1e-06,
            "litellm_provider": "together_ai",
            "mode": "chat",
            "output_cost_per_token": 1e-06,
        }
    }
    outcome = sync.compute_sync(registry, [_chat_model("acme/alive")], _doc({"acme/gone": "2026-07-01"}))
    assert outcome.cost_map["together_ai/acme/gone"]["deprecation_date"] == "2026-06-15"
    assert any("2026-06-15" in warning and "2026-07-01" in warning for warning in outcome.warnings)


def test_redirect_chain_resolves_to_final_live_model() -> None:
    doc = _doc({"acme/a": "2026-01-01"}, redirects={"acme/a": "acme/b", "acme/b": "acme/c"})
    live = frozenset({"acme/c"})
    assert sync.resolve_successor("acme/a", doc, live) == "acme/c"


def test_redirect_dead_end_yields_no_successor() -> None:
    doc = _doc({"acme/a": "2026-01-01"}, redirects={"acme/a": "acme/b"})
    assert sync.resolve_successor("acme/a", doc, frozenset({"acme/other"})) is None


def test_redirect_short_names_resolve_by_unique_suffix() -> None:
    doc = _doc({"moonshotai/Kimi-K2": "2026-01-01"}, redirects={"Kimi-K2": "Kimi-K2-0905"})
    live = frozenset({"moonshotai/Kimi-K2-0905"})
    assert sync.resolve_successor("moonshotai/Kimi-K2", doc, live) == "moonshotai/Kimi-K2-0905"


def test_successor_written_only_when_not_curated() -> None:
    registry = {
        "together_ai/acme/a": {
            "input_cost_per_token": 1e-06,
            "litellm_provider": "together_ai",
            "mode": "chat",
            "output_cost_per_token": 1e-06,
        },
        "together_ai/acme/b": {
            "input_cost_per_token": 1e-06,
            "litellm_provider": "together_ai",
            "metadata": {"successor": "together_ai/acme/curated"},
            "mode": "chat",
            "output_cost_per_token": 1e-06,
        },
    }
    doc = _doc({"acme/a": "2026-01-01", "acme/b": "2026-01-01"}, redirects={"acme/a": "acme/c", "acme/b": "acme/c"})
    outcome = sync.compute_sync(registry, [_chat_model("acme/c")], doc)
    assert outcome.cost_map["together_ai/acme/a"]["metadata"] == {"successor": "together_ai/acme/c"}
    assert outcome.cost_map["together_ai/acme/b"]["metadata"] == {"successor": "together_ai/acme/curated"}
    assert any("acme/curated" in warning for warning in outcome.warnings)


def test_reappearance_clears_deprecation_date() -> None:
    registry = {
        "together_ai/acme/back": {
            "deprecation_date": "2026-05-01",
            "input_cost_per_token": 1e-06,
            "litellm_provider": "together_ai",
            "max_input_tokens": 4096,
            "mode": "chat",
            "output_cost_per_token": 1e-06,
        }
    }
    outcome = sync.compute_sync(registry, [_chat_model("acme/back")], _doc({"x": "2026-01-01"}))
    assert "deprecation_date" not in outcome.cost_map["together_ai/acme/back"]
    assert outcome.reappeared == ("together_ai/acme/back",)


def test_new_chat_model_without_rule_is_flagged() -> None:
    outcome = sync.compute_sync({}, [_chat_model("acme/unreviewed")], _doc({"x": "2026-01-01"}))
    assert any("acme/unreviewed" in warning and "capability rule" in warning for warning in outcome.warnings)


def test_new_keys_land_at_the_end_of_the_provider_block() -> None:
    registry = {
        "aaa": {"mode": "chat"},
        "together_ai/acme/old": {
            "input_cost_per_token": 1e-06,
            "litellm_provider": "together_ai",
            "mode": "chat",
            "output_cost_per_token": 1e-06,
        },
        "zzz": {"mode": "chat"},
    }
    outcome = sync.compute_sync(registry, [_chat_model("acme/old"), _chat_model("acme/new")], _doc({"x": "2026-01-01"}))
    assert list(outcome.cost_map) == ["aaa", "together_ai/acme/old", "together_ai/acme/new", "zzz"]


def test_sync_is_idempotent_over_the_repo_cost_map() -> None:
    cost_map = json.loads((ROOT / "model_prices_and_context_window.json").read_text())
    first = sync.compute_sync(cost_map, RECORDED_CATALOG, RECORDED_DOC)
    second = sync.compute_sync(first.cost_map, RECORDED_CATALOG, RECORDED_DOC)
    assert not second.has_changes
    assert second.cost_map == first.cost_map


def test_pr_body_lists_every_section_and_the_skipped_types() -> None:
    outcome = sync.compute_sync({}, RECORDED_CATALOG, RECORDED_DOC)
    body = sync.render_pr_body(outcome)
    assert "### Added (26)" in body
    assert "### Warnings needing a human call" in body
    assert "image (29)" in body
    assert "video (38)" in body
