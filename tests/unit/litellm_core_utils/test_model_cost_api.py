import asyncio
import copy
import json
import queue
import threading
from collections.abc import Generator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import count
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

import pytest

import litellm
from litellm.litellm_core_utils import model_cost_api
from litellm.litellm_core_utils.fallback_generalizations import (
    get_fallback_generalization_rules,  # pyright: ignore[reportUnknownVariableType]  # legacy fallback rules lack annotations
    set_fallback_generalizations,  # pyright: ignore[reportUnknownVariableType]  # legacy fallback rules lack annotations
)
from litellm.litellm_core_utils.get_model_cost_map import (
    _MODEL_COST_LAYERS,  # pyright: ignore[reportPrivateUsage]  # native layer snapshots isolate global state
    GetModelCostMap,
)
from litellm.litellm_core_utils.model_cost_api import (
    Entries,
    FetchedModelCostMap,
    File,
    ModelCostMapUnavailable,
    ModelCostReloaded,
    Remote,
    fetch_model_cost_map,
    model_cost_snapshot,
    reload_model_cost_map,
    set_model_cost_overrides,
    validate_model_cost_entries,
)
from litellm.litellm_core_utils.model_cost_dict import ModelCostDict
from litellm.rust_bridge._native import ModelRegistry, model_row_errors


@dataclass(frozen=True, slots=True)
class _LocalResponses:
    url: str
    requests: queue.Queue[None]


@contextmanager
def _local_responses(responses: tuple[tuple[int, bytes], ...]) -> Generator[_LocalResponses, None, None]:
    response_indexes: Final = count()
    request_count: Final[queue.Queue[None]] = queue.Queue()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            request_count.put(None)
            index: Final = min(next(response_indexes), len(responses) - 1)
            status, body = responses[index]
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread: Final = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield _LocalResponses(f"http://127.0.0.1:{server.server_port}/prices.json", request_count)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@contextmanager
def _isolated_model_cost_state() -> Generator[None, None, None]:
    from litellm.litellm_core_utils import get_model_cost_map as get_model_cost_map_module

    original_map: Final[ModelCostDict] = cast(  # cast-ok: dynamic module export is the live model-cost dict
        ModelCostDict, litellm.model_cost
    )
    original_layers: Final = _MODEL_COST_LAYERS.snapshot()
    original_registry: Final = GetModelCostMap._base_registry  # pyright: ignore[reportPrivateUsage]  # test state restore
    original_catalog: Final = GetModelCostMap._loaded_catalog  # pyright: ignore[reportPrivateUsage]  # test state restore
    original_fields: Final = GetModelCostMap._loaded_catalog_fields  # pyright: ignore[reportPrivateUsage]  # test state restore
    original_owners: Final = model_cost_api._owners  # pyright: ignore[reportPrivateUsage]  # test state restore
    original_source: Final = model_cost_api._current_source  # pyright: ignore[reportPrivateUsage]  # test state restore
    source_info: Final = get_model_cost_map_module._cost_map_source_info  # pyright: ignore[reportPrivateUsage]  # test state restore
    source_values: Final = (
        source_info.source,
        source_info.url,
        source_info.is_env_forced,
        source_info.fallback_reason,
        source_info.loaded_at,
        source_info.source_revision,
        source_info.etag,
    )
    original_fallback: Final[list[object]] = copy.deepcopy(cast(list[object], get_fallback_generalization_rules()))
    module_values: Final[dict[str, object]] = cast(dict[str, object], vars(litellm))
    original_provider_sets: Final[dict[str, set[str]]] = {
        name: cast(set[str], value)  # pyright: ignore[reportAny]  # dynamic module exports are runtime sets
        for name, value in module_values.items()
        if name.endswith("_models") and isinstance(value, set)
    }
    original_models_by_provider: Final[dict[str, object]] = cast(dict[str, object], litellm.models_by_provider).copy()
    try:
        yield
    finally:
        _MODEL_COST_LAYERS.restore(original_layers)
        GetModelCostMap._base_registry = original_registry  # pyright: ignore[reportPrivateUsage]  # test state restore
        GetModelCostMap._loaded_catalog = original_catalog  # pyright: ignore[reportPrivateUsage]  # test state restore
        GetModelCostMap._loaded_catalog_fields = original_fields  # pyright: ignore[reportPrivateUsage]  # test state restore
        model_cost_api._owners = original_owners  # pyright: ignore[reportPrivateUsage]  # test state restore
        model_cost_api._current_source = original_source  # pyright: ignore[reportPrivateUsage]  # test state restore
        source_info.source, source_info.url, source_info.is_env_forced = source_values[:3]
        source_info.fallback_reason, source_info.loaded_at, source_info.source_revision, source_info.etag = (
            source_values[3:]
        )
        set_fallback_generalizations(original_fallback)
        litellm.model_cost = original_map
        for name, values in original_provider_sets.items():
            current = cast(set[str], getattr(litellm, name))
            current.clear()
            current.update(values)
        models_by_provider = cast(dict[str, object], litellm.models_by_provider)
        models_by_provider.clear()
        models_by_provider.update(original_models_by_provider)
        from litellm import utils

        utils.invalidate_model_cost_caches()


@pytest.fixture(autouse=True)
def isolated_catalog() -> Generator[None, None, None]:
    with _isolated_model_cost_state():
        litellm.model_cost = ModelCostDict(copy.deepcopy(dict(_live_model_cost())))
        yield


def _mutable_mapping(
    value: object,
) -> MutableMapping[str, object]:  # mutable-ok: tests attempt writes to prove results are immutable
    return cast(  # cast-ok: tests deliberately attempt mutation through a read-only result
        MutableMapping[str, object], value
    )


def _live_model_cost() -> Mapping[str, Mapping[str, object]]:
    return cast(  # cast-ok: model_cost is a string-keyed mapping exposed dynamically by litellm
        Mapping[str, Mapping[str, object]], litellm.model_cost
    )


def test_entries_fetch_normalizes_generic_mappings_and_freezes_reads() -> None:
    source: Final = Entries(
        MappingProxyType(
            {
                "custom/model": MappingProxyType(
                    {
                        "litellm_provider": "openai",
                        "nested": MappingProxyType({"values": [1, {"stable": True}]}),
                    }
                )
            }
        )
    )

    first: Final = fetch_model_cost_map(source)
    second: Final = fetch_model_cost_map(Entries({"custom/model": dict(source.entries["custom/model"])}))

    assert isinstance(first, FetchedModelCostMap)
    assert isinstance(second, FetchedModelCostMap)
    assert first.revision == second.revision
    assert first.entries == second.entries
    with pytest.raises(TypeError):
        _mutable_mapping(first.entries["custom/model"])["nested"] = {}
    values: Final = cast(  # cast-ok: the nested test input is a sequence frozen as a tuple
        Sequence[object],
        _mutable_mapping(_mutable_mapping(first.entries["custom/model"])["nested"])["values"],
    )
    with pytest.raises(TypeError):
        _mutable_mapping(values[1])["stable"] = False


def test_bundled_model_cost_map_is_cached_and_deeply_immutable() -> None:
    first: Final = model_cost_api.bundled_model_cost_map()
    second: Final = model_cost_api.bundled_model_cost_map()

    assert first is second
    assert first == model_cost_api._freeze_entries(  # pyright: ignore[reportPrivateUsage]  # compare public bundled contents
        ModelRegistry.bundled().served()
    )
    first_row: Final = next(iter(first.values()))
    with pytest.raises(TypeError):
        _mutable_mapping(first_row)["input_cost_per_token"] = 0


def test_validation_reports_rejecting_and_non_rejecting_native_diagnostics() -> None:
    issues: Final = validate_model_cost_entries(
        {
            "invalid": {
                "litellm_provider": "openai",
                "input_cost_per_token": -1,
                "output_cost_per_token": True,
            },
            "providerless": {"input_cost_per_token": 0.1, "output_cost_per_token": 0.2},
            "nonfinite": {
                "litellm_provider": "openai",
                "input_cost_per_token": float("nan"),
                "output_cost_per_token": float("inf"),
            },
        }
    )

    assert any(issue.key == "invalid" and issue.rejects for issue in issues)
    assert any(issue.key == "providerless" and not issue.rejects for issue in issues)


def test_validation_matches_native_diagnostic_tuples() -> None:
    entries: Final = {
        "negative": {
            "litellm_provider": "openai",
            "input_cost_per_token": -1,
            "output_cost_per_token": -2,
        },
        "string": {"litellm_provider": "openai", "input_cost_per_token": "bad"},
        "providerless": {"input_cost_per_token": 0.1},
        "boolean": {"litellm_provider": "openai", "input_cost_per_token": True},
        "mappingproxy": MappingProxyType({"litellm_provider": "openai", "input_cost_per_token": -1}),
        "nan": {"litellm_provider": "openai", "input_cost_per_token": float("nan")},
        "infinity": {"litellm_provider": "openai", "input_cost_per_token": float("inf")},
    }
    actual: Final = validate_model_cost_entries(entries)
    expected: Final = tuple(
        (key, error.field, error.kind, error.rejects, error.message)
        for key, row in entries.items()
        for error in model_row_errors(key, dict(row) if isinstance(row, MappingProxyType) else row)
    )
    assert tuple((issue.key, issue.field, issue.kind, issue.rejects, issue.message) for issue in actual) == expected


def test_native_diagnostics_reject_nonfinite_prices() -> None:
    issues: Final = validate_model_cost_entries(
        {
            "nan": {"litellm_provider": "openai", "input_cost_per_token": float("nan")},
            "infinity": {"litellm_provider": "openai", "input_cost_per_token": float("inf")},
        }
    )
    assert {issue.key for issue in issues if issue.rejects} == {"nan", "infinity"}


def test_owner_validation_keeps_good_siblings_and_does_not_store_rejected_rows() -> None:
    with _isolated_model_cost_state():
        owner: Final = "owner-per-row"
        result: Final = set_model_cost_overrides(
            owner,
            {
                "openai/gpt-4o": MappingProxyType({"input_cost_per_token": 0.123}),
                "invalid/negative": {"litellm_provider": "openai", "input_cost_per_token": -1},
                "invalid/object": {"litellm_provider": "openai", "custom": object()},
            },
        )
        assert "openai/gpt-4o" in result.applied
        assert "invalid/negative" not in result.applied
        assert "invalid/object" not in result.applied
        snapshot: Final = model_cost_snapshot()
        assert "invalid/negative" not in snapshot.owners
        assert "invalid/object" not in snapshot.owners
        set_model_cost_overrides(owner, {})


def test_reload_preserves_registration_and_owner_rows_but_counts_only_base_rows() -> None:
    with _isolated_model_cost_state():
        key: Final = "openai/registered"
        litellm.register_model(
            model_cost={key: {"litellm_provider": "openai", "mode": "chat", "custom_metadata": "registered"}}
        )
        set_model_cost_overrides("reload-owner", {key: {"input_cost_per_token": 0.123}})
        result: Final = reload_model_cost_map(
            Entries({key: {"litellm_provider": "openai", "mode": "chat", "output_cost_per_token": 0.2}})
        )
        assert isinstance(result, ModelCostReloaded)
        assert result.row_count == 1
        assert _live_model_cost()[key]["input_cost_per_token"] == 0.123
        assert _live_model_cost()[key]["custom_metadata"] == "registered"


def test_explicit_entries_reload_is_not_env_forced() -> None:
    with _isolated_model_cost_state():
        result: Final = reload_model_cost_map(
            Entries({"explicit/source": {"litellm_provider": "openai", "mode": "chat"}})
        )
        assert isinstance(result, ModelCostReloaded)
        assert model_cost_api.get_model_cost_map_source_info()["is_env_forced"] is False


def test_async_fetch_and_reload_have_behavioral_results() -> None:
    with _isolated_model_cost_state():
        result: Final = asyncio.run(
            model_cost_api.areload_model_cost_map(
                Entries({"async/source": {"litellm_provider": "openai", "mode": "chat"}})
            )
        )
        assert isinstance(result, ModelCostReloaded)
        assert result.row_count == 1
        assert "async/source" in _live_model_cost()


def test_explicit_remote_fetch_retries_inline_and_does_not_adopt() -> None:
    from litellm.litellm_core_utils import get_model_cost_map as loader

    payload: Final = json.dumps(
        {
            f"local/model-{index}": {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
            }
            for index in range(2200)
        }
    ).encode()
    unavailable: Final = ((503, b"temporary failure"), (200, payload))

    original_map: Final[Mapping[str, Mapping[str, object]]] = cast(  # cast-ok: model_cost is a string-keyed catalog
        Mapping[str, Mapping[str, object]], litellm.model_cost
    )
    original_source: Final = model_cost_api._current_source  # pyright: ignore[reportPrivateUsage]  # purity assertion
    provenance: Final = loader.get_model_cost_map_source_info()
    source_state: Final = loader._cost_map_source_info  # pyright: ignore[reportPrivateUsage]  # verify loader identity purity
    fallback: Final[list[object]] = cast(list[object], get_fallback_generalization_rules())
    catalog: Final = GetModelCostMap._loaded_catalog  # pyright: ignore[reportPrivateUsage]  # verify catalog identity purity
    before_threads: Final = frozenset(threading.enumerate())
    with _local_responses(unavailable) as responses:
        fetched: Final = fetch_model_cost_map(Remote(responses.url), max_attempts=2)
        assert responses.requests.qsize() == 2

    assert isinstance(fetched, FetchedModelCostMap)
    assert "local/model-0" in fetched.entries
    assert _live_model_cost() is original_map
    assert model_cost_api._current_source is original_source  # pyright: ignore[reportPrivateUsage]  # purity assertion
    assert loader._cost_map_source_info is source_state  # pyright: ignore[reportPrivateUsage]  # verify loader identity purity
    assert loader.get_model_cost_map_source_info() == provenance
    assert get_fallback_generalization_rules() is fallback
    assert GetModelCostMap._loaded_catalog is catalog  # pyright: ignore[reportPrivateUsage]  # verify catalog identity purity
    assert frozenset(threading.enumerate()) <= before_threads


def test_explicit_default_url_beats_local_flag_and_failed_reload_is_atomic(monkeypatch: pytest.MonkeyPatch) -> None:
    payload: Final = json.dumps(ModelRegistry.bundled().served()).encode()
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    with _local_responses(((200, payload), (200, payload), (404, b"missing"))) as responses:
        monkeypatch.setattr(litellm, "model_cost_map_url", responses.url)
        assert isinstance(reload_model_cost_map(), ModelCostReloaded)
        assert responses.requests.qsize() == 0
        assert isinstance(asyncio.run(model_cost_api.afetch_model_cost_map(Remote(responses.url))), FetchedModelCostMap)
        assert isinstance(reload_model_cost_map(Remote(responses.url)), ModelCostReloaded)
        assert responses.requests.qsize() == 2
        assert model_cost_snapshot().source == Remote(responses.url)
        assert model_cost_api.get_model_cost_map_source_info()["is_env_forced"] is False
        original: Final[ModelCostDict] = cast(  # cast-ok: dynamic module export is the live model-cost dict
            ModelCostDict, litellm.model_cost
        )
        provenance: Final = model_cost_api.get_model_cost_map_source_info()
        fallback: Final[list[object]] = cast(list[object], get_fallback_generalization_rules())
        catalog: Final = GetModelCostMap._loaded_catalog  # pyright: ignore[reportPrivateUsage]  # verify failed reload atomicity
        assert isinstance(reload_model_cost_map(Remote(responses.url)), ModelCostMapUnavailable)
        assert _live_model_cost() is original
        assert model_cost_api.get_model_cost_map_source_info() == provenance
        assert get_fallback_generalization_rules() is fallback
        assert GetModelCostMap._loaded_catalog is catalog  # pyright: ignore[reportPrivateUsage]  # verify failed reload atomicity


def test_owner_precedence_deletion_replacement_and_copy_safe_snapshot() -> None:
    key: Final = "api/precedence"
    new_key: Final = "api/new"
    base: Final = {
        "litellm_provider": "openai",
        "mode": "chat",
        "input_cost_per_token": 0.1,
        "cache_read_input_token_cost": 0.05,
    }
    assert isinstance(reload_model_cost_map(Entries({key: base})), ModelCostReloaded)
    result: Final = set_model_cost_overrides(
        "fallback", {key: {"input_cost_per_token": -1}, new_key: base}, precedence="fallback"
    )
    assert result.rejected == ()
    assert _live_model_cost()[key]["input_cost_per_token"] == 0.1
    assert new_key in cast(set[str], litellm.open_ai_chat_completion_models)
    set_model_cost_overrides("override", {key: {"input_cost_per_token": 0.2, "cache_read_input_token_cost": None}})
    assert _live_model_cost()[key]["input_cost_per_token"] == 0.2
    assert "cache_read_input_token_cost" not in _live_model_cost()[key]
    snapshot: Final = model_cost_snapshot()
    assert snapshot.owners[key] == ("fallback", "override")
    assert dict(snapshot.entries[key]) == dict(_live_model_cost()[key])
    assert isinstance(
        reload_model_cost_map(Entries({key: base, new_key: base | {"input_cost_per_token": 0.3}})), ModelCostReloaded
    )
    assert _live_model_cost()[new_key]["input_cost_per_token"] == 0.3
    set_model_cost_overrides("override", {})
    assert _live_model_cost()[key]["input_cost_per_token"] == 0.1
    assert _live_model_cost()[key]["cache_read_input_token_cost"] == 0.05
    assert snapshot.entries[key]["input_cost_per_token"] == 0.2
    set_model_cost_overrides("fallback", {"api/providerless": {"input_cost_per_token": 0.4}}, precedence="fallback")
    assert key not in model_cost_snapshot().owners
    providerless: Final = set_model_cost_overrides("providerless", {"api/providerless": {"output_cost_per_token": 0.2}})
    assert providerless.applied == ("api/providerless",)
    assert providerless.rejected and all(not issue.rejects for issue in providerless.rejected)


def test_owner_suppression_on_reload_is_retried_and_later_owners_can_repair() -> None:
    key: Final = "api/tiered"
    base: Final = {"litellm_provider": "openai", "mode": "chat"}
    tiered: Final = [{"range": [0, 10], "input_cost_per_token": 0.2}]
    assert isinstance(reload_model_cost_map(Entries({key: base})), ModelCostReloaded)
    assert set_model_cost_overrides("tiered", {key: {"tiered_pricing": tiered}}).applied == (key,)
    assert isinstance(reload_model_cost_map(Entries({key: base | {"input_cost_per_token": 0.1}})), ModelCostReloaded)
    assert "tiered_pricing" not in _live_model_cost()[key]
    assert model_cost_snapshot().owners[key] == ("tiered",)
    assert isinstance(reload_model_cost_map(Entries({key: base})), ModelCostReloaded)
    assert _live_model_cost()[key]["tiered_pricing"] == tiered
    set_model_cost_overrides("repair", {key: {"input_cost_per_token": None}})
    assert isinstance(reload_model_cost_map(Entries({key: base | {"input_cost_per_token": 0.1}})), ModelCostReloaded)
    assert _live_model_cost()[key]["tiered_pricing"] == tiered
    assert "input_cost_per_token" not in _live_model_cost()[key]


def test_overrides_replace_rows_validate_siblings_and_snapshot_owners() -> None:
    original_map: Final[Mapping[str, Mapping[str, object]]] = cast(  # cast-ok: model_cost is a string-keyed catalog
        Mapping[str, Mapping[str, object]], litellm.model_cost
    )
    original_layers: Final = _MODEL_COST_LAYERS.snapshot()
    owner: Final = "model-cost-api-test"
    try:
        result: Final = set_model_cost_overrides(
            owner,
            {
                "openai/gpt-4o": {"input_cost_per_token": 0.123},
                "invalid/model": {"litellm_provider": "openai", "input_cost_per_token": -1},
            },
        )

        assert "openai/gpt-4o" in result.applied
        assert any(issue.key == "invalid/model" and issue.rejects for issue in result.rejected)
        assert _live_model_cost()["openai/gpt-4o"]["input_cost_per_token"] == 0.123
        snapshot: Final = model_cost_snapshot()
        assert owner in snapshot.owners["openai/gpt-4o"]
        with pytest.raises(TypeError):
            _mutable_mapping(snapshot.entries["openai/gpt-4o"])["input_cost_per_token"] = 0
    finally:
        set_model_cost_overrides(owner, {})
        litellm.model_cost = original_map
        _MODEL_COST_LAYERS.restore(original_layers)


def test_reload_entries_returns_base_row_count_and_preserves_registration_layer() -> None:
    original_map: Final[Mapping[str, Mapping[str, object]]] = cast(  # cast-ok: model_cost is a string-keyed catalog
        Mapping[str, Mapping[str, object]], litellm.model_cost
    )
    original_layers: Final = _MODEL_COST_LAYERS.snapshot()
    original_registry: Final = GetModelCostMap._base_registry  # pyright: ignore[reportPrivateUsage]  # restore native fixture state
    original_catalog: Final = GetModelCostMap._loaded_catalog  # pyright: ignore[reportPrivateUsage]  # restore native fixture state
    original_fields: Final = GetModelCostMap._loaded_catalog_fields  # pyright: ignore[reportPrivateUsage]  # restore native fixture state
    original_source: Final = model_cost_api._current_source  # pyright: ignore[reportPrivateUsage]  # restore API state
    source: Final = Entries(
        {
            "api/reloaded": {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
            }
        }
    )
    try:
        result: Final = reload_model_cost_map(source)

        assert isinstance(result, ModelCostReloaded)
        assert result.row_count == 1
        assert result.source == source
        assert "api/reloaded" in _live_model_cost()
    finally:
        litellm.model_cost = original_map
        _MODEL_COST_LAYERS.restore(original_layers)
        GetModelCostMap._base_registry = original_registry  # pyright: ignore[reportPrivateUsage]  # restore native fixture state
        GetModelCostMap._loaded_catalog = original_catalog  # pyright: ignore[reportPrivateUsage]  # restore native fixture state
        GetModelCostMap._loaded_catalog_fields = original_fields  # pyright: ignore[reportPrivateUsage]  # restore native fixture state
        model_cost_api._current_source = original_source  # pyright: ignore[reportPrivateUsage]  # restore API state


def test_reload_failure_is_a_value_and_does_not_change_current_map(tmp_path: Path) -> None:
    original_map: Final[Mapping[str, Mapping[str, object]]] = cast(  # cast-ok: model_cost is a string-keyed catalog
        Mapping[str, Mapping[str, object]], litellm.model_cost
    )
    result: Final = reload_model_cost_map(File(tmp_path / "missing.json"), max_attempts=1)

    assert isinstance(result, ModelCostMapUnavailable)
    assert litellm.model_cost is original_map  # pyright: ignore[reportUnknownMemberType]  # dynamic module export
