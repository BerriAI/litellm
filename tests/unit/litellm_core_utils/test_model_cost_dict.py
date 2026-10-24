import copy
import json
import operator
import pickle
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Final, cast

import pytest
from pydantic import BaseModel, ConfigDict, JsonValue

import litellm
from litellm import ModelCostDict
from litellm import utils as litellm_utils
from litellm.litellm_core_utils.get_model_cost_map import (
    _MODEL_COST_LAYERS,
    GetModelCostMap,
    LoadedCatalog,
    adopt_model_cost_map,
    get_model_cost_map_source_info,
    install_loaded_catalog,
)
from litellm.litellm_core_utils.model_cost_dict import ServedRow
from litellm.rust_bridge._native import ModelCostLayers, ModelRegistry


def test_model_cost_dict_preserves_served_rows_and_alias_identity() -> None:
    row = ServedRow({"input_cost_per_token": 0.1}, ("model", "alias"), frozenset({"input_cost_per_token"}))
    model_cost = ModelCostDict({"model": row, "alias": row})

    assert isinstance(litellm.model_cost, litellm.ModelCostDict)
    assert isinstance(model_cost, dict)
    assert isinstance(model_cost["model"], ServedRow)
    assert model_cost["model"] is model_cost["alias"]
    assert model_cost["model"].names == ("model", "alias")
    assert model_cost["model"].registered_fields == frozenset({"input_cost_per_token"})


def test_runtime_registered_model_cost_is_live_read_only_sparse_view() -> None:
    previous_layers: Final = _MODEL_COST_LAYERS.snapshot()
    original_model_cost: Final = cast(ModelCostDict, litellm.model_cost)
    original_key: Final = "openai/gpt-4o"
    litellm.model_cost = ModelCostDict(original_model_cost)
    try:
        _MODEL_COST_LAYERS.forget_writes()
        runtime_cost: Final = litellm_utils._runtime_registered_model_cost  # pyright: ignore[reportPrivateUsage]  # Exercise the requested private view
        litellm.register_model(
            {
                original_key: {
                    "input_cost_per_token": 0,
                    "supports_function_calling": False,
                    "max_input_tokens": None,
                    "nested_cost_metadata": {"first": 1},
                }
            },
            add_to_provider_model_sets=False,
        )

        assert tuple(runtime_cost) == (original_key,)
        assert runtime_cost[original_key] == {
            "input_cost_per_token": 0,
            "supports_function_calling": False,
            "nested_cost_metadata": {"first": 1},
        }

        litellm.register_model(
            {
                original_key: {
                    "input_cost_per_token": None,
                    "supports_function_calling": None,
                    "max_input_tokens": None,
                    "max_output_tokens": 0,
                    "nested_cost_metadata": {"second": 2},
                }
            },
            add_to_provider_model_sets=False,
        )

        assert tuple(runtime_cost) == (original_key,)
        assert "gpt-4o" not in runtime_cost
        assert runtime_cost[original_key] == {
            "input_cost_per_token": 0,
            "supports_function_calling": False,
            "max_output_tokens": 0,
            "nested_cost_metadata": {"first": 1, "second": 2},
        }
        with pytest.raises(TypeError):
            operator.setitem(
                cast(MutableMapping[str, Mapping[str, object]], runtime_cost),
                original_key,
                cast(Mapping[str, object], {}),
            )
        with pytest.raises(TypeError):
            operator.setitem(
                cast(MutableMapping[str, object], runtime_cost[original_key]),
                "input_cost_per_token",
                1,
            )

        nested: Final = cast(dict[str, object], runtime_cost[original_key]["nested_cost_metadata"])
        nested["first"] = 99
        assert runtime_cost[original_key]["nested_cost_metadata"] == {"first": 1, "second": 2}

        _MODEL_COST_LAYERS.forget_writes()
        assert tuple(runtime_cost) == ()
    finally:
        _MODEL_COST_LAYERS.restore(previous_layers)
        litellm.model_cost = original_model_cost


def test_model_cost_dict_copies_are_plain_dicts_with_plain_rows() -> None:
    row = ServedRow({"input_cost_per_token": 0.1}, ("model", "alias"))
    model_cost = ModelCostDict({"model": row, "alias": row})

    copies = (
        copy.copy(model_cost),
        model_cost.copy(),
        copy.deepcopy(model_cost),
        pickle.loads(pickle.dumps(model_cost)),
    )

    assert all(type(value) is dict for value in copies)
    assert all(type(value["model"]) is dict for value in copies)
    assert all(value["model"] is value["alias"] for value in copies)


def test_model_cost_dict_json_preserves_order_and_values() -> None:
    model_cost = ModelCostDict(
        {
            "first": ServedRow({"input_cost_per_token": 0.1}, ("first",)),
            "second": ServedRow({"output_cost_per_token": 0.2}, ("second",)),
        }
    )

    assert json.dumps(model_cost) == (
        '{"first": {"input_cost_per_token": 0.1}, "second": {"output_cost_per_token": 0.2}}'
    )


class RegistrationCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    model_cost: dict[str, dict[str, JsonValue]]
    custom_llm_provider: str | None = None
    add_to_provider_model_sets: bool = True
    persist_across_reloads: bool = True


class ReplayCase(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    base: dict[str, dict[str, JsonValue]]
    calls: tuple[RegistrationCall, ...]
    reload: dict[str, dict[str, JsonValue]]
    before_reload: dict[str, dict[str, JsonValue]]
    after_reload: dict[str, dict[str, JsonValue]]


class ReplayReference(BaseModel):
    model_config = ConfigDict(frozen=True)
    reference_revision: str
    cases: tuple[ReplayCase, ...]


_REFERENCE: Final = ReplayReference.model_validate_json(
    Path(__file__).with_name("registration_replay_reference.json").read_text()
)


@pytest.mark.parametrize("case", _REFERENCE.cases, ids=lambda case: case.name)
def test_registration_replay_matches_pre_migration_python(case: ReplayCase) -> None:
    original: Final = GetModelCostMap._base_registry
    provenance: Final = get_model_cost_map_source_info()
    _MODEL_COST_LAYERS.forget_writes()
    try:
        adopt_model_cost_map(copy.deepcopy(case.base))
        for call in case.calls:
            litellm.register_model(
                copy.deepcopy(call.model_cost),
                custom_llm_provider=call.custom_llm_provider,
                add_to_provider_model_sets=call.add_to_provider_model_sets,
                persist_across_reloads=call.persist_across_reloads,
            )
        assert list(litellm.model_cost) == list(case.before_reload)
        assert litellm.model_cost == case.before_reload

        adopt_model_cost_map(copy.deepcopy(case.reload))

        assert list(litellm.model_cost) == list(case.after_reload)
        assert litellm.model_cost == case.after_reload
    finally:
        install_loaded_catalog(
            LoadedCatalog(original, provenance["source_revision"] or "", provenance["etag"]),
            provenance=provenance,
        )


def test_replay_metadata_preserves_native_fields_aliases_and_missing_token_prices() -> None:
    base = {
        "gpt-4o": {
            "litellm_provider": "openai",
            "mode": "chat",
            "aliases": ["parity-alias"],
            "input_cost_per_token": 0.7,
            "supports_function_calling": True,
        },
        "untouched": {"litellm_provider": "custom", "mode": "chat"},
    }
    adopt_model_cost_map(base)
    litellm.register_model({"gpt-4o": {"input_cost_per_token": 0.0, "supports_function_calling": False}})
    provenance: Final = get_model_cost_map_source_info()
    installed: Final = install_loaded_catalog(
        LoadedCatalog(
            GetModelCostMap.loaded_registry(),
            provenance["source_revision"] or "",
            provenance["etag"],
        ),
        provenance=provenance,
    )
    installed_row: Final = installed["gpt-4o"]
    assert installed_row["max_input_tokens"] is None
    assert "output_cost_per_token" not in installed_row
    assert installed_row is installed["parity-alias"]
    adopt_model_cost_map(base)

    row = litellm.model_cost["gpt-4o"]
    native_rows = {key: fields for key, fields, _ in _MODEL_COST_LAYERS.compose(GetModelCostMap.loaded_registry())}
    assert {key: row[key] for key in native_rows["gpt-4o"]} == native_rows["gpt-4o"]
    assert row["input_cost_per_token"] == 0.0
    assert row["supports_function_calling"] is False
    assert "output_cost_per_token" not in row
    assert row["max_input_tokens"] is None
    assert row["key"] == "gpt-4o"
    assert row["supported_openai_params"] == litellm.get_supported_openai_params("gpt-4o")
    assert row is litellm.model_cost["parity-alias"]
    assert row.names == ("gpt-4o", "parity-alias")
    assert row.registered_fields == frozenset({"input_cost_per_token", "supports_function_calling"})
    assert litellm.model_cost["untouched"] == base["untouched"]
    assert "supported_openai_params" not in GetModelCostMap.loaded_model_cost_map()["gpt-4o"]


def test_registration_replay_fallback_keeps_source_aliases_and_inherits_refreshed_fields() -> None:
    original: Final = GetModelCostMap._base_registry
    provenance: Final = get_model_cost_map_source_info()
    previous_layers: Final = _MODEL_COST_LAYERS.snapshot()
    source_key: Final = "anthropic.parity-tiered-v1:0"
    alias_key: Final = "fallback-source-alias"
    target_key: Final = "bedrock/ap-northeast-1.anthropic.parity-tiered-v1:0"
    base: Final = {
        source_key: {
            "litellm_provider": "bedrock",
            "mode": "chat",
            "aliases": [alias_key],
            "supports_function_calling": True,
            "tiered_pricing": [
                {"input_cost_per_token": 0.01, "output_cost_per_token": 0.02, "range": [0, 256000]},
                {"input_cost_per_token": 0.03, "output_cost_per_token": 0.04, "range": [256000, 1000000]},
            ],
        }
    }
    refreshed: Final = {
        source_key: {
            **base[source_key],
            "supports_function_calling": False,
            "tiered_pricing": [
                {"input_cost_per_token": 0.11, "output_cost_per_token": 0.12, "range": [0, 256000]},
                {"input_cost_per_token": 0.13, "output_cost_per_token": 0.14, "range": [256000, 1000000]},
            ],
        }
    }
    _MODEL_COST_LAYERS.forget_writes()
    try:
        adopt_model_cost_map(copy.deepcopy(base))
        source_before: Final = copy.deepcopy(cast(dict[str, object], litellm.model_cost[source_key]))
        litellm.register_model(
            {
                target_key: {
                    "litellm_provider": "bedrock",
                    "input_cost_per_token": 0.5,
                }
            },
            add_to_provider_model_sets=False,
        )

        assert litellm.model_cost[source_key] == source_before
        assert litellm.model_cost[alias_key] == source_before
        assert litellm.model_cost[source_key] is litellm.model_cost[alias_key]

        adopt_model_cost_map(copy.deepcopy(refreshed))

        source_after: Final = litellm.model_cost[source_key]
        target_after: Final = litellm.model_cost[target_key]
        assert source_after is litellm.model_cost[alias_key]
        assert source_after["tiered_pricing"] == refreshed[source_key]["tiered_pricing"]
        assert source_after["supports_function_calling"] is False
        assert target_after["tiered_pricing"] == [
            {"input_cost_per_token": 0.5, "output_cost_per_token": 0.12, "range": [0, 256000]},
            {"input_cost_per_token": 0.5, "output_cost_per_token": 0.14, "range": [256000, 1000000]},
        ]
        assert target_after["supports_function_calling"] is False
    finally:
        _MODEL_COST_LAYERS.restore(previous_layers)
        install_loaded_catalog(
            LoadedCatalog(original, provenance["source_revision"] or "", provenance["etag"]),
            provenance=provenance,
        )


@pytest.mark.parametrize("target", ("base", "alias"))
@pytest.mark.parametrize("write_kind", ("direct", "owner"))
def test_composed_alias_rows_match_compose_keys_for_overrides(target: str, write_kind: str) -> None:
    base: Final = ModelRegistry.parse(
        b'{"base":{"litellm_provider":"test","aliases":["alias"],"input_cost_per_token":1}}'
    )
    layers: Final = ModelCostLayers()
    if write_kind == "direct":
        layers.record_direct(target, {"input_cost_per_token": 2})
    else:
        layers.set_owner("override", "override", {target: {"input_cost_per_token": 2}})

    composed: Final = {key: (row, registered) for key, row, registered in layers.compose(base)}
    selected: Final = {
        key: (row, registered) for key, row, registered in layers.compose_keys(base, frozenset({"base", "alias"}))
    }
    expected_base: Final = (
        {"input_cost_per_token": 2}
        if write_kind == "direct" and target == "base"
        else {"litellm_provider": "test", "input_cost_per_token": 2 if target == "base" else 1}
    )
    expected_alias: Final = (
        {"input_cost_per_token": 2}
        if write_kind == "direct" and target == "alias"
        else {"litellm_provider": "test", "input_cost_per_token": 2 if target == "alias" else 1}
    )

    assert composed == selected
    assert composed == {
        "base": (expected_base, frozenset()),
        "alias": (expected_alias, frozenset()),
    }


@pytest.mark.parametrize("target", ("base", "alias"))
def test_composed_alias_rows_split_when_registration_provenance_differs(target: str) -> None:
    base: Final = ModelRegistry.parse(
        b'{"base":{"litellm_provider":"test","aliases":["alias"],"input_cost_per_token":1}}'
    )
    layers: Final = ModelCostLayers()
    layers.record_registration(target, "base", {"input_cost_per_token": 1})

    composed: Final = {key: (row, registered) for key, row, registered in layers.compose(base)}
    selected: Final = {
        key: (row, registered) for key, row, registered in layers.compose_keys(base, frozenset({"base", "alias"}))
    }

    assert composed == selected
    assert composed["base"] == (
        {"litellm_provider": "test", "input_cost_per_token": 1},
        frozenset({"input_cost_per_token"}) if target == "base" else frozenset(),
    )
    assert composed["alias"] == (
        {"litellm_provider": "test", "input_cost_per_token": 1},
        frozenset({"input_cost_per_token"}) if target == "alias" else frozenset(),
    )
    assert composed["base"][0] is not composed["alias"][0]


def test_layers_preserve_order_tombstones_and_independent_alias_writes() -> None:
    base: Final = ModelRegistry.parse(
        b'{"z":{"litellm_provider":"test","aliases":["b"]},"a":{"litellm_provider":"test"}}'
    )
    layers: Final = ModelCostLayers()
    assert [key for key, _, _ in layers.compose(base)] == ["z", "a", "b"]
    layers.record_registration("b", "z", {"input_cost_per_token": 1.0})
    layers.record_tombstone("z")
    composed: Final = layers.compose(base)
    assert [key for key, _, _ in composed] == ["a", "b"]
    assert composed[1][1]["input_cost_per_token"] == 1.0
    assert composed[1][2] == frozenset({"input_cost_per_token"})
    selected: Final = {
        key: (row, registered) for key, row, registered in layers.compose_keys(base, frozenset({"z", "b"}))
    }
    assert selected["z"] == (None, frozenset())
    assert selected["b"] == (composed[1][1], composed[1][2])


def test_registration_prefix_includes_only_preceding_application_writes() -> None:
    base: Final = ModelRegistry.parse(b'{"base":{"litellm_provider":"test"},"removed":{"mode":"chat"}}')
    layers: Final = ModelCostLayers()
    layers.record_direct("direct", {"mode": "chat"})
    layers.record_tombstone("removed")
    handle: Final = layers.record_registration("target", "base", {"input_cost_per_token": 0.1})
    layers.record_direct("future", {"mode": "embedding"})

    prefix: Final = layers.compose_prefix(base, handle)

    assert [key for key, _, _ in prefix] == ["base", "direct"]
    assert prefix[1][1] == {"mode": "chat"}


def test_owner_rows_preserve_insertion_order_and_restore_self() -> None:
    base: Final = ModelRegistry.parse(b'{"base":{"litellm_provider":"test"}}')
    layers: Final = ModelCostLayers()
    layers.set_owner("owner", "override", {"z": {"mode": "chat"}, "a": {"mode": "embedding"}})
    layers.restore(layers)
    assert [key for key, _, _ in layers.compose(base)] == ["base", "z", "a"]
