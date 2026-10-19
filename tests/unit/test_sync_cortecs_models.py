import importlib.util
import json
import sys
from pathlib import Path

import pytest

import litellm

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "sync_cortecs_models.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "cortecs_sync"

_spec = importlib.util.spec_from_file_location("sync_cortecs_models", SCRIPT)
assert _spec is not None and _spec.loader is not None
sync = importlib.util.module_from_spec(_spec)
sys.modules["sync_cortecs_models"] = sync
_spec.loader.exec_module(sync)

RECORDED_CATALOG = sync.load_catalog(FIXTURES.joinpath("models.json").read_bytes())
COST_MAP = json.loads((ROOT / "model_prices_and_context_window.json").read_text())


def _model(
    model_id: str = "some-model",
    *,
    inputs: tuple[str, ...] = ("text",),
    outputs: tuple[str, ...] = ("text",),
    features: tuple[str, ...] = (),
    context: int = 128000,
    max_output: int = 8192,
    input_price: float = 1.0,
    output_price: float = 2.0,
    cached_read: float | None = None,
    cached_write: float | None = None,
) -> sync.CatalogModel:
    return sync.CatalogModel(
        id=model_id,
        context_size=context,
        max_output_tokens=max_output,
        input_modalities=inputs,
        output_modalities=outputs,
        supported_features=features,
        pricing=sync.CatalogPricing(
            currency="USD",
            input_token=input_price,
            output_token=output_price,
            cache_read_cost=cached_read,
            cache_write_cost=cached_write,
        ),
    )


def _cost_map(*entries: tuple[str, object]) -> dict[str, object]:
    return dict(entries)


@pytest.mark.parametrize(
    ("per_million", "expected"),
    [
        (3, 3e-06),
        (15, 1.5e-05),
        (1.4, 1.4e-06),
        (0.2424, 2.424e-07),
        (0, 0.0),
        (0.30000000000000004, 3e-07),
        (0.0001234, 1.234e-10),
    ],
)
def test_per_token_normalizes_float_artifacts(per_million: float, expected: float) -> None:
    assert sync.per_token(per_million) == expected


def test_complete_entry_maps_catalog_metadata_onto_registry_fields() -> None:
    entry = sync.entry_for(
        _model(
            inputs=("text", "image", "file", "audio"),
            features=("tools", "reasoning"),
            context=200000,
            max_output=64000,
            input_price=3,
            output_price=15,
            cached_read=0.3,
            cached_write=3.75,
        )
    )

    assert entry == {
        "cache_creation_input_token_cost": 3.75e-06,
        "cache_read_input_token_cost": 3e-07,
        "input_cost_per_token": 3e-06,
        "litellm_provider": "cortecs",
        "max_input_tokens": 200000,
        "max_output_tokens": 64000,
        "max_tokens": 64000,
        "mode": "chat",
        "output_cost_per_token": 1.5e-05,
        "source": "https://api.cortecs.ai/v1/models",
        "supports_audio_input": True,
        "supports_function_calling": True,
        "supports_pdf_input": True,
        "supports_prompt_caching": True,
        "supports_reasoning": True,
        "supports_tool_choice": True,
        "supports_vision": True,
    }


def test_entry_omits_flags_the_catalog_does_not_advertise() -> None:
    entry = sync.entry_for(_model(inputs=("text",), features=()))

    assert not any(key.startswith("supports_") for key in entry)
    assert "cache_read_input_token_cost" not in entry


def test_entry_only_claims_prompt_caching_when_the_catalog_prices_a_cache_read() -> None:
    without_cache = sync.entry_for(_model())
    with_cache = sync.entry_for(_model(cached_read=0.3))

    assert "supports_prompt_caching" not in without_cache
    assert with_cache["supports_prompt_caching"] is True
    assert with_cache["cache_read_input_token_cost"] == 3e-07


def test_load_catalog_rejects_a_catalog_that_stopped_pricing_in_usd() -> None:
    catalog = json.loads(FIXTURES.joinpath("models.json").read_text())
    catalog["data"][0]["pricing"]["currency"] = "EUR"

    with pytest.raises(sync.SyncError, match="prices in"):
        sync.load_catalog(json.dumps(catalog).encode())


def test_load_catalog_rejects_a_shape_change() -> None:
    with pytest.raises(sync.SyncError, match="no longer matches the expected shape"):
        sync.load_catalog(b'[{"id": "x", "type": "chat"}]')


def test_load_catalog_rejects_an_empty_catalog() -> None:
    with pytest.raises(sync.SyncError, match="refusing to sync an empty catalog"):
        sync.load_catalog(b'{"object": "list", "data": []}')


def test_sync_adds_every_catalog_model() -> None:
    outcome = sync.compute_sync({}, RECORDED_CATALOG)

    assert outcome.added == tuple(sorted(f"cortecs/{model.id}" for model in RECORDED_CATALOG))
    assert outcome.stale == ()
    assert all(entry["litellm_provider"] == "cortecs" for entry in outcome.cost_map.values())


def test_sync_is_idempotent() -> None:
    first = sync.compute_sync({}, RECORDED_CATALOG)
    second = sync.compute_sync(first.cost_map, RECORDED_CATALOG)

    assert second.has_changes is False
    assert second.cost_map == first.cost_map


def test_sync_refreshes_owned_fields_and_keeps_hand_added_ones() -> None:
    existing = {
        "input_cost_per_token": 9e-06,
        "litellm_provider": "cortecs",
        "max_tokens": 100,
        "mode": "chat",
        "output_cost_per_token": 9e-06,
        "supports_function_calling": True,
        "supports_response_schema": True,
    }
    model = _model(model_id="known", input_price=3, output_price=15, max_output=4096)

    outcome = sync.compute_sync(_cost_map(("cortecs/known", existing)), [model])

    assert outcome.updated == ("cortecs/known",)
    assert outcome.cost_map["cortecs/known"] == sync.entry_for(model) | {"supports_response_schema": True}
    assert outcome.cost_map["cortecs/known"] != existing


def test_sync_drops_a_flag_the_catalog_stopped_advertising() -> None:
    existing = sync.entry_for(_model(model_id="known", features=("tools",)))
    outcome = sync.compute_sync(_cost_map(("cortecs/known", existing)), [_model(model_id="known")])

    assert outcome.updated == ("cortecs/known",)
    assert "supports_function_calling" not in outcome.cost_map["cortecs/known"]


def test_sync_reports_a_delisted_model_without_removing_it() -> None:
    delisted = sync.entry_for(_model(model_id="gone"))
    outcome = sync.compute_sync(_cost_map(("cortecs/gone", delisted)), [_model(model_id="kept")])

    assert outcome.stale == ("cortecs/gone",)
    assert outcome.cost_map["cortecs/gone"] == delisted
    assert any("deprecation call" in warning for warning in outcome.warnings)


def test_sync_warns_about_terms_it_cannot_map() -> None:
    outcome = sync.compute_sync({}, [_model(inputs=("text", "hologram"), features=("json_mode", "tools"))])

    assert any("hologram" in warning for warning in outcome.warnings)


def test_entry_writes_the_catalog_audio_second_price() -> None:
    model = _model()
    audio_priced = model.model_copy(update={"pricing": model.pricing.model_copy(update={"audio_cost": 2.5e-06})})

    entry = sync.entry_for(audio_priced)

    assert entry["input_cost_per_audio_per_second"] == 2.5e-06
    assert "input_cost_per_audio_per_second" not in sync.entry_for(model)


def test_sync_skips_models_with_non_text_output_instead_of_registering_them_as_chat() -> None:
    image_model = _model("image-maker", outputs=("image",))
    outcome = sync.compute_sync({}, [_model(), image_model])

    assert outcome.added == ("cortecs/some-model",)
    assert "cortecs/image-maker" not in outcome.cost_map
    assert any("image-maker" in warning for warning in outcome.warnings)


def test_every_field_the_sync_can_write_is_refreshed_and_cleared_on_update() -> None:
    everything = _model(
        inputs=("text", "image", "audio", "file", "video"),
        features=("tools", "reasoning"),
        cached_read=0.3,
        cached_write=3.75,
    )
    everything = everything.model_copy(
        update={"pricing": everything.pricing.model_copy(update={"audio_cost": 2.5e-06})}
    )
    full_entry = sync.entry_for(everything)
    stale_entry = {field: "stale" for field in full_entry}

    refreshed = sync.compute_sync({"cortecs/some-model": stale_entry}, [everything]).cost_map["cortecs/some-model"]
    cleared = sync.compute_sync({"cortecs/some-model": full_entry}, [_model()]).cost_map["cortecs/some-model"]

    assert refreshed == full_entry
    assert set(cleared) == set(sync.entry_for(_model()))


def test_sync_warns_about_prices_it_does_not_write() -> None:
    model = _model()
    speech_priced = model.model_copy(update={"pricing": model.pricing.model_copy(update={"speech_cost": 15.0})})
    outcome = sync.compute_sync({}, [speech_priced])

    assert any("speech_cost" in warning for warning in outcome.warnings)
    assert "input_cost_per_character" not in outcome.cost_map["cortecs/some-model"]


def test_render_replaces_only_the_updated_entry() -> None:
    original = (
        "{\n"
        '    "sample_spec": {"input_cost_per_token": "USD per prompt token"},\n'
        '    "cortecs/known": {\n'
        '        "input_cost_per_token": 9e-06,\n'
        '        "litellm_provider": "cortecs"\n'
        "    },\n"
        '    "openrouter/inline": {"us":1.1,"fast":2}\n'
        "}\n"
    )
    outcome = sync.compute_sync(json.loads(original), [_model(model_id="known", input_price=3)])
    rendered = sync.render_cost_map_text(original, outcome)

    assert json.loads(rendered) == outcome.cost_map
    assert '"openrouter/inline": {"us":1.1,"fast":2}' in rendered
    assert '"input_cost_per_token": 3e-06' in rendered
    assert rendered.count("cortecs/known") == 1


def test_render_appends_new_entries_before_the_closing_brace() -> None:
    original = '{\n    "openrouter/inline": {"us":1.1,"fast":2}\n}\n'
    outcome = sync.compute_sync(json.loads(original), [_model(model_id="known", input_price=3)])
    rendered = sync.render_cost_map_text(original, outcome)

    assert rendered == (
        "{\n"
        '    "openrouter/inline": {"us":1.1,"fast":2},\n'
        '    "cortecs/known": {\n'
        '        "input_cost_per_token": 3e-06,\n'
        '        "litellm_provider": "cortecs",\n'
        '        "max_input_tokens": 128000,\n'
        '        "max_output_tokens": 8192,\n'
        '        "max_tokens": 8192,\n'
        '        "mode": "chat",\n'
        '        "output_cost_per_token": 2e-06,\n'
        '        "source": "https://api.cortecs.ai/v1/models"\n'
        "    }\n"
        "}\n"
    )


def test_render_is_a_no_op_when_nothing_changed() -> None:
    original = '{\n    "cortecs/known": {\n        "litellm_provider": "cortecs"\n    }\n}\n'
    outcome = sync.SyncOutcome(cost_map=json.loads(original))

    assert sync.render_cost_map_text(original, outcome) == original


def test_registry_entries_match_the_recorded_catalog() -> None:
    outcome = sync.compute_sync(COST_MAP, RECORDED_CATALOG)

    assert outcome.added == ()
    assert outcome.updated == ()


def test_every_registry_entry_is_a_complete_chat_model() -> None:
    entries = {key: entry for key, entry in COST_MAP.items() if key.startswith("cortecs/")}

    assert len(entries) == len(RECORDED_CATALOG)
    for key, entry in entries.items():
        assert entry["litellm_provider"] == "cortecs", key
        assert entry["mode"] == "chat", key
        assert entry["max_tokens"] == entry["max_output_tokens"], key
        assert entry["input_cost_per_token"] >= 0, key
        assert entry["output_cost_per_token"] >= 0, key


def test_registry_entries_route_and_price_through_litellm() -> None:
    info = litellm.get_model_info("cortecs/gpt-6.1-sol")
    row = COST_MAP["cortecs/gpt-6.1-sol"]
    prompt_cost, completion_cost = litellm.cost_per_token(
        model="cortecs/gpt-6.1-sol", prompt_tokens=10, completion_tokens=10
    )

    assert info["litellm_provider"] == "cortecs"
    assert prompt_cost == pytest.approx(row["input_cost_per_token"] * 10)
    assert completion_cost == pytest.approx(row["output_cost_per_token"] * 10)
