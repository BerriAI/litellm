import json
from typing import Final
from uuid import UUID

import pytest

import litellm
from litellm._catalog import _install, _read
from litellm._catalog._load import LoadedCatalog, load
from litellm.litellm_core_utils.get_model_cost_map import AliasedCostEntry
from litellm.llms.scaledown.chat.transformation import ScaleDownError
from litellm.rust_bridge._native import ModelNotFound, ResolvedModel
from litellm.utils import ModelNotMappedError, _invalidate_model_cost_lowercase_map, get_model_cost_key


@pytest.mark.parametrize(
    ("prices", "expected"),
    (
        ({}, False),
        ({"input_cost_per_token": None, "output_cost_per_token": None}, False),
        ({"input_cost_per_token": 0, "output_cost_per_token": 0}, True),
        ({"input_cost_per_token": 0.25}, True),
        ({"output_cost_per_token": 0.5}, True),
        ({"input_cost_per_token": -1}, False),
        ({"input_cost_per_token": "invalid-price"}, False),
        ({"input_cost_per_second": 1.0}, False),
        ({"cache_read_input_token_cost": 0.0}, True),
        ({"cache_creation_input_token_cost_above_1hr": 0.0}, True),
        ({"output_cost_per_reasoning_token": 0.5}, True),
        ({"input_cost_per_audio_token": 0.5}, True),
        ({"output_cost_per_image_token": 0.5}, True),
        ({"input_cost_per_token_priority": 0.5}, True),
        ({"input_cost_per_token_batches": 0.5}, True),
        ({"input_cost_per_token_above_128k_tokens": 0.5}, True),
        ({"tiered_pricing": [{"range": [0, 200], "input_cost_per_token": 0.5}]}, True),
        ({"tiered_pricing": [{"range": [100, 200], "input_cost_per_token": 0.5}]}, False),
        ({"off_peak_pricing": {"hours_utc": "01:00-02:00", "input_cost_per_token": 0.5}}, True),
        (
            {
                "input_cost_per_token": 1.0,
                "tiered_pricing": [{"range": [0, 100], "input_cost_per_token": 2.0}],
            },
            True,
        ),
    ),
)
def test_is_priced_distinguishes_explicit_token_rates(
    monkeypatch: pytest.MonkeyPatch, prices: dict[str, object], expected: bool
) -> None:
    from litellm._catalog import is_priced

    monkeypatch.setattr(litellm, "model_cost", {"openai/price-probe": {"litellm_provider": "openai", **prices}})
    assert is_priced("price-probe", "openai") is expected
    assert is_priced("unmapped-price-probe", "openai") is False


def test_is_priced_tracks_selected_live_row_and_tombstones(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm._catalog import is_priced

    rows: Final[dict[str, dict[str, object]]] = {
        "price-probe": {"litellm_provider": "openai", "input_cost_per_token": 1},
        "openai/price-probe": {"litellm_provider": "openai"},
    }
    monkeypatch.setattr(litellm, "model_cost", rows)
    assert is_priced("price-probe", "openai") is False
    rows["openai/price-probe"]["input_cost_per_token"] = 0
    assert is_priced("price-probe", "openai") is True
    rows["openai/price-probe"] = {"litellm_provider": "openai"}
    assert is_priced("price-probe", "openai") is False
    del rows["openai/price-probe"]
    assert is_priced("price-probe", "openai") is True
    del rows["price-probe"]
    assert is_priced("price-probe", "openai") is False


def test_providerless_deployment_prices_without_writing_provider_context(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm._catalog import is_priced

    row: Final = {"input_cost_per_token": 1.0, "output_cost_per_token": 2.0}
    monkeypatch.setitem(litellm.model_cost, "providerless-priced-deployment", row)
    assert is_priced("providerless-priced-deployment", "openai") is True
    assert litellm.cost_per_token(
        "providerless-priced-deployment", 10, 20, custom_llm_provider="openai"
    ) == pytest.approx((10.0, 40.0))
    assert row == {"input_cost_per_token": 1.0, "output_cost_per_token": 2.0}


@pytest.mark.parametrize("provider", (None, "", 123))
def test_presence_does_not_repair_an_explicit_invalid_provider(
    monkeypatch: pytest.MonkeyPatch, provider: object
) -> None:
    from litellm._catalog import is_priced

    row: Final = {"litellm_provider": provider, "input_cost_per_token": 1.0, "output_cost_per_token": 2.0}
    monkeypatch.setitem(litellm.model_cost, "openai/invalid-provider-price-probe", row)
    assert is_priced("invalid-provider-price-probe", "openai") is False
    assert row == {"litellm_provider": provider, "input_cost_per_token": 1.0, "output_cost_per_token": 2.0}


def test_model_info_preserves_every_bundled_row() -> None:
    for key, row in litellm.model_cost.items():
        result: Final = _read.resolve(key)
        assert isinstance(result, ResolvedModel), key
        assert result.key == key
        assert _read.model_info(result.key) == row, key


@pytest.mark.parametrize("field", ("supports_sampling_params", "supports_output_config"))
def test_model_info_forwards_declared_bundled_capabilities(field: str) -> None:
    rows: Final = tuple((key, row) for key, row in litellm.model_cost.items() if field in row)
    assert rows
    for key, row in rows:
        assert litellm.get_model_info(key).get(field) == row[field], key


@pytest.mark.parametrize("nested", (False, True))
def test_model_info_reads_direct_price_and_capability_writes(monkeypatch: pytest.MonkeyPatch, nested: bool) -> None:
    key: Final = "openai/live-read-model"
    row: Final = {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 0.25, "supports_vision": False}
    monkeypatch.setattr(litellm, "model_cost", {key: row})
    before: Final = litellm.get_model_info(key)
    assert before["input_cost_per_token"] == 0.25
    assert litellm.supports_vision(key) is False
    if nested:
        litellm.model_cost[key].update(input_cost_per_token=1.25, supports_vision=True)
    else:
        litellm.model_cost[key] = {**row, "input_cost_per_token": 1.25, "supports_vision": True}
    after: Final = litellm.get_model_info(key)
    assert after["input_cost_per_token"] == 1.25
    assert litellm.supports_vision(key) is True
    assert before["input_cost_per_token"] == 0.25
    assert before is not after


def test_provider_indexes_follow_size_changes_and_forced_same_size_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    key: Final = "live-provider-model"
    monkeypatch.setattr(litellm, "model_cost", {})
    _read.refresh_provider_sets()
    litellm.model_cost.setdefault(key, {"litellm_provider": "openai", "mode": "chat", "supports_vision": True})
    assert litellm.supports_vision(key) is True
    assert key in litellm.open_ai_chat_completion_models
    assert key in litellm.models_by_provider["openai"]
    litellm.model_cost[key]["litellm_provider"] = "anthropic"
    _invalidate_model_cost_lowercase_map()
    assert key not in litellm.open_ai_chat_completion_models
    assert key in litellm.anthropic_models
    del litellm.model_cost[key]
    assert get_model_cost_key(key) is None
    assert key not in litellm.anthropic_models


@pytest.mark.parametrize("initially_listed", (False, True))
def test_registration_listing_intent_survives_rebuilds_and_public_opt_in(
    monkeypatch: pytest.MonkeyPatch, initially_listed: bool
) -> None:
    key: Final = "listing-intent-model"
    row: Final = {"litellm_provider": "gemini", "mode": "chat", "supports_vision": True}
    monkeypatch.setattr(litellm, "model_cost", {key: row} if initially_listed else {})
    _invalidate_model_cost_lowercase_map()
    litellm.register_model({key: row}, add_to_provider_model_sets=False)
    assert (key in litellm.gemini_models) is initially_listed
    litellm.register_model({"unrelated-listing-model": row})
    _invalidate_model_cost_lowercase_map()
    assert (key in litellm.gemini_models) is initially_listed
    assert "unrelated-listing-model" in litellm.gemini_models
    litellm.model_cost[key]["supports_vision"] = False
    assert litellm.supports_vision(key, custom_llm_provider="gemini") is False
    assert (key in litellm.gemini_models) is initially_listed
    litellm.model_cost[key]["litellm_provider"] = "openai"
    _invalidate_model_cost_lowercase_map()
    assert (key in litellm.gemini_models) is initially_listed
    assert key not in litellm.open_ai_chat_completion_models
    litellm.register_model({key: row})
    assert key in litellm.gemini_models


@pytest.mark.parametrize("delete_first", (False, True))
def test_replacing_an_unlisted_entry_drops_only_its_listing_intent(
    monkeypatch: pytest.MonkeyPatch, delete_first: bool
) -> None:
    key: Final = "replaced-listing-model"
    row: Final = {"litellm_provider": "gemini", "mode": "chat"}
    monkeypatch.setattr(litellm, "model_cost", {})
    _invalidate_model_cost_lowercase_map()
    litellm.register_model({key: row}, add_to_provider_model_sets=False)
    assert key not in litellm.gemini_models
    if delete_first:
        del litellm.model_cost[key]
        _invalidate_model_cost_lowercase_map()
        assert key not in litellm.gemini_models
    litellm.model_cost[key] = dict(row)
    _invalidate_model_cost_lowercase_map()
    assert key in litellm.gemini_models


def test_skip_listing_preserves_existing_provider_memberships(monkeypatch: pytest.MonkeyPatch) -> None:
    key: Final = "provider-listing-model"
    monkeypatch.setattr(litellm, "model_cost", {key: {"litellm_provider": "gemini", "mode": "chat"}})
    _invalidate_model_cost_lowercase_map()
    litellm.register_model({key: {"litellm_provider": "openai"}}, add_to_provider_model_sets=False)
    _invalidate_model_cost_lowercase_map()
    assert key in litellm.gemini_models
    assert key not in litellm.open_ai_chat_completion_models
    litellm.register_model({key: {"litellm_provider": "openai"}})
    assert key not in litellm.gemini_models
    assert key in litellm.open_ai_chat_completion_models


def test_mixed_alias_listing_survives_reload_without_changing_shared_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    row: Final = {"litellm_provider": "gemini", "mode": "chat", "supports_vision": True}
    shared: Final = AliasedCostEntry(row, ("listed-alias-base", "unlisted-alias"))
    monkeypatch.setattr(litellm, "model_cost", {"listed-alias-base": shared, "unlisted-alias": shared})
    _invalidate_model_cost_lowercase_map()
    litellm.gemini_models.discard("unlisted-alias")
    litellm.register_model({"unlisted-alias": {"supports_vision": False}}, add_to_provider_model_sets=False)
    _invalidate_model_cost_lowercase_map()
    assert "listed-alias-base" in litellm.gemini_models
    assert "unlisted-alias" not in litellm.gemini_models
    assert litellm.model_cost["listed-alias-base"] is litellm.model_cost["unlisted-alias"]
    assert litellm.model_cost["listed-alias-base"]["supports_vision"] is False

    catalog: Final = load(json.dumps({"listed-alias-base": {**row, "aliases": ["unlisted-alias"]}}).encode())
    assert isinstance(catalog, LoadedCatalog)
    _install.install(catalog, provenance=_install.Provenance(source="bundled"))
    assert "listed-alias-base" in litellm.gemini_models
    assert "unlisted-alias" not in litellm.gemini_models
    assert litellm.model_cost["listed-alias-base"] is litellm.model_cost["unlisted-alias"]
    expected: Final = {**row, "supports_vision": False}
    assert _read.model_info("listed-alias-base") == expected
    assert json.loads(json.dumps(litellm.model_cost["unlisted-alias"])) == expected
    assert {key: fields for key, fields, _ in _install.LAYERS.compose(catalog.registry)} == {
        "listed-alias-base": expected,
        "unlisted-alias": expected,
    }

    del litellm.model_cost["unlisted-alias"]
    _install.install(catalog, provenance=_install.Provenance(source="bundled"))
    assert "unlisted-alias" not in litellm.model_cost
    assert "listed-alias-base" in litellm.gemini_models
    litellm.model_cost["unlisted-alias"] = dict(row)
    _invalidate_model_cost_lowercase_map()
    assert "unlisted-alias" in litellm.gemini_models


def test_native_index_recovers_removed_hits_but_needs_invalidation_for_unseen_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row: Final = {"litellm_provider": "hosted_vllm", "mode": "chat"}
    monkeypatch.setattr(litellm, "model_cost", {"hosted_vllm/Acme-Model": row})
    assert litellm.get_model_info("hosted_vllm/acme-model")["key"] == "hosted_vllm/Acme-Model"
    litellm.model_cost["hosted_vllm/ACME-model"] = litellm.model_cost.pop("hosted_vllm/Acme-Model")
    assert litellm.get_model_info("hosted_vllm/acme-model")["key"] == "hosted_vllm/ACME-model"
    litellm.model_cost["hosted_vllm/New-Model"] = litellm.model_cost.pop("hosted_vllm/ACME-model")
    assert isinstance(_read.resolve("hosted_vllm/new-model"), ModelNotFound)
    _invalidate_model_cost_lowercase_map()
    assert litellm.get_model_info("hosted_vllm/new-model")["key"] == "hosted_vllm/New-Model"
    litellm.model_cost["hosted_vllm/Other-Model"] = litellm.model_cost.pop("hosted_vllm/New-Model")
    assert get_model_cost_key("hosted_vllm/other-model") is None
    litellm.model_cost["size-change"] = {}
    assert get_model_cost_key("hosted_vllm/other-model") == "hosted_vllm/Other-Model"


def test_missing_model_is_typed_silent_and_does_not_change_global_debug_setting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    debug: Final = litellm.suppress_debug_info
    with pytest.raises(ModelNotMappedError):
        litellm.get_model_info("m4-unmapped-no-provider")
    assert capsys.readouterr().out == ""
    assert litellm.suppress_debug_info is debug
    assert litellm.get_model_info.cache_clear() is None


@pytest.mark.parametrize("provider", (None, "scaledown"))
def test_provider_validation_error_precedes_missing_catalog_error(provider: str | None) -> None:
    model: Final = "scaledown/m4-invalid-model"
    with pytest.raises(ScaleDownError) as error:
        litellm.get_model_info(model, custom_llm_provider=provider)
    assert error.value.status_code == 400
    assert model in str(error.value)


@pytest.mark.parametrize("value", (UUID(int=17), {"nested": [UUID(int=17)]}))
def test_opaque_reads_preserve_registration_shaping_without_writing_catalog_state(
    monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    key: Final = "openai/opaque-read-model"
    row: Final = {
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_vision": True,
        "metadata": value,
        "input_cost_per_token": 1.25,
    }
    monkeypatch.setattr(litellm, "model_cost", {key: row})
    opaque: Final = dict(_install.STATE.opaque)
    registrations: Final = _install.LAYERS.registrations()
    assert litellm.get_model_info(key)["input_cost_per_token"] == 1.25
    assert litellm.supports_vision(key) is True
    assert _read.model_info(key)["metadata"] is value
    assert _install.STATE.opaque == opaque
    assert _install.LAYERS.registrations() == registrations
    assert litellm.model_cost[key] is row
    litellm.register_model({key: {"output_cost_per_token": 2.5}})
    assert litellm.model_cost[key]["litellm_provider"] == "openai"
    assert litellm.model_cost[key]["metadata"] == value
    assert litellm.model_cost[key]["input_cost_per_token"] == 1.25
    assert litellm.model_cost[key]["output_cost_per_token"] == 2.5


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_tokens", "1234"),
        ("input_cost_per_token", "0.5"),
        ("input_cost_per_token", -1.0),
        ("supports_vision", "true"),
        ("supports_vision", 1),
    ),
)
def test_non_lowerable_known_fields_keep_permissive_public_read_values(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    key: Final = "openai/non-lowerable-model"
    monkeypatch.setattr(litellm, "model_cost", {key: {"litellm_provider": "openai", "mode": "chat", field: value}})
    result: Final = litellm.get_model_info(key)
    assert result[field] == value
    assert type(result[field]) is type(value)
    if field == "supports_vision":
        assert litellm.supports_vision(key) is False


@pytest.mark.parametrize(
    ("query", "provider", "key"),
    (
        ("read-model", "openai", "openai/read-model"),
        ("hosted_vllm/read-model", "hosted_vllm", "hosted_vllm/Read-Model"),
        ("read-model-001", "vertex_ai", "vertex_ai/read-model"),
        ("read-model-2026-01-01", "openai", "openai/read-model"),
        ("ft:read-model:org:suffix:id", "openai", "ft:read-model"),
        ("us.anthropic.read-model-v1:0", "bedrock", "bedrock/anthropic.read-model-v1:0"),
        ("us-east-1/anthropic.read-model", "bedrock_mantle", "bedrock_mantle/anthropic.read-model"),
    ),
)
def test_public_reads_and_capabilities_share_native_resolution(
    monkeypatch: pytest.MonkeyPatch, query: str, provider: str, key: str
) -> None:
    from litellm import utils

    functions: Final = (
        utils.supports_vision,
        utils.supports_function_calling,
        utils.supports_parallel_function_calling,
        utils.supports_tool_choice,
        utils.supports_system_messages,
        utils.supports_native_streaming,
        utils.supports_response_schema,
        utils.supports_audio_input,
        utils.supports_video_input,
        utils.supports_box_questions,
        utils.supports_pdf_input,
        utils.supports_audio_output,
        utils.supports_prompt_caching,
        utils.supports_prompt_cache_breakpoint,
        utils.supports_assistant_prefill,
        utils.supports_thinking_cache_preservation,
        utils.supports_computer_use,
        utils.supports_reasoning,
        utils.supports_anthropic_thinking_payload,
        utils.supports_none_reasoning_effort,
        utils.supports_mid_conversation_system,
        utils.supports_native_structured_output,
        utils.supports_embedding_image_input,
        utils.supports_web_search,
        utils.supports_tool_search,
        utils.supports_url_context,
    )
    capabilities: Final = {function.__name__: True for function in functions}
    monkeypatch.setattr(litellm, "model_cost", {key: {"litellm_provider": provider, "mode": "chat", **capabilities}})
    assert litellm.get_model_info(query, custom_llm_provider=provider)["key"] == key
    for function in functions:
        assert function(query, custom_llm_provider=provider) is True, function.__name__
        litellm.model_cost[key][function.__name__] = False
        assert function(query, custom_llm_provider=provider) is False, function.__name__
