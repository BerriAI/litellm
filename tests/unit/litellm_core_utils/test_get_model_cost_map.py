"""
Tests for model-cost-map loading: the model-count integrity check (which must
count actual model entries, not reserved meta keys) and the extraction of the
``fallback_generalizations`` block out of the raw map.
"""

import asyncio
import copy
import importlib
import json
import logging
import os
import select
import subprocess
import sys
import threading
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Final

import boto3
import pytest
from typing_extensions import NotRequired, ReadOnly, TypedDict

import litellm
from litellm.litellm_core_utils import get_model_cost_map as get_model_cost_map_module
from litellm.litellm_core_utils.fallback_generalizations import (
    get_fallback_generalization_rules,
    match_routing_generalization,
    set_fallback_generalizations,
)
from litellm.litellm_core_utils.get_model_cost_map import (
    _MODEL_COST_LAYERS,
    FALLBACK_GENERALIZATIONS_KEY,
    CostMapSourceInfo,
    GetModelCostMap,
    LoadedCatalog,
    ModelCostMapUnavailable,
    NativeSource,
    _classify_fetch_content,
    _finalize_model_cost_map,
    _load,
    adopt_model_cost_map,
    get_model_cost_map_provenance,
    get_model_cost_map_source_info,
    git_blob_id,
    install_loaded_catalog,
)
from litellm.rust_bridge._native import ModelCostLayers, ModelRegistry, model_row_errors

_ROOT_COST_MAP_PATH: Final = os.path.join(os.path.dirname(__file__), "../../../model_prices_and_context_window.json")


def test_native_registry_rejects_nested_nonfinite_owner_rows() -> None:
    key: Final = "nested-nonfinite-owner"
    row: Final = {
        "litellm_provider": "openai",
        "metadata": {"prices": [float("nan")]},
    }
    errors: Final = model_row_errors(key, row)
    layers: Final = ModelCostLayers()
    with pytest.raises(ValueError):
        layers.set_owner("nonfinite-owner", "override", {key: row})
    assert tuple(layers.compose_keys(ModelRegistry.bundled(), (key,))) == ((key, None, frozenset()),)
    assert tuple((error.key, error.field, error.kind, error.rejects) for error in errors) == (
        (key, "metadata/prices/0", "invalid", True),
    )


@pytest.fixture
def restore_installed_catalog_state() -> Iterator[None]:
    previous_registry: Final = GetModelCostMap.loaded_registry()
    previous_catalog: Final = GetModelCostMap.loaded_model_cost_map()
    source_info: Final = get_model_cost_map_module._cost_map_source_info
    previous_source: Final = (
        source_info.source,
        source_info.url,
        source_info.is_env_forced,
        source_info.fallback_reason,
        source_info.loaded_at,
        source_info.checked_at,
        source_info.source_revision,
        source_info.etag,
    )
    previous_fallbacks: Final = list(get_fallback_generalization_rules())
    previous_layers: Final = _MODEL_COST_LAYERS.snapshot()
    api_module: Final = sys.modules.get("litellm.litellm_core_utils.model_cost_api")
    api_state: Final = (
        None
        if api_module is None
        else (getattr(api_module, "_owners", None), getattr(api_module, "_current_source", None))
    )

    def restore() -> None:
        GetModelCostMap._base_registry = previous_registry
        GetModelCostMap._snapshot_loaded_catalog(previous_catalog)
        set_fallback_generalizations(previous_fallbacks)
        (
            source_info.source,
            source_info.url,
            source_info.is_env_forced,
            source_info.fallback_reason,
            source_info.loaded_at,
            source_info.checked_at,
            source_info.source_revision,
            source_info.etag,
        ) = previous_source
        _MODEL_COST_LAYERS.restore(previous_layers)
        if api_module is not None and api_state is not None:
            setattr(api_module, "_owners", api_state[0])
            setattr(api_module, "_current_source", api_state[1])

    yield
    restore()


def _load_root_cost_map() -> dict:
    with open(_ROOT_COST_MAP_PATH) as f:
        return json.load(f)


def _bundled_blob_id() -> str:
    with open(_ROOT_COST_MAP_PATH, "rb") as f:
        return git_blob_id(f.read())


def test_git_blob_id_is_what_git_hash_object_prints():
    assert git_blob_id(b'{"gpt-5.4-mini": {"mode": "chat"}}\n') == "18b9a8381e13a3b38a2128f184f631f95829e987"


def test_git_blob_id_matches_shared_native_fixture() -> None:
    fixture: Final = json.loads(
        (Path(__file__).resolve().parents[3] / "litellm-rust/crates/model-catalog/tests/git_blob_id.json").read_text()
    )
    body: Final = fixture["git_blob_id"]["example_body_utf8"].encode()

    assert git_blob_id(body) == fixture["git_blob_id"]["example_revision"]


def _make_models(n: int) -> dict:
    return {f"model-{i}": {"litellm_provider": "openai", "mode": "chat"} for i in range(n)}


def test_native_integrity_check_excludes_reserved_keys() -> None:
    shrunk = _make_models(5)
    shrunk["sample_spec"] = {"foo": "bar"}
    shrunk[FALLBACK_GENERALIZATIONS_KEY] = {"rules": [{"name": "x"}]}

    registry: Final = ModelRegistry.parse(json.dumps(shrunk).encode())

    assert registry.check_size(reference=2000, minimum=50, ratio=0.5) is False


def test_native_integrity_check_accepts_healthy_registry() -> None:
    healthy = _make_models(2000)
    healthy["sample_spec"] = {"foo": "bar"}
    healthy[FALLBACK_GENERALIZATIONS_KEY] = {"rules": []}

    registry: Final = ModelRegistry.parse(json.dumps(healthy).encode())

    assert registry.check_size(reference=2000, minimum=50, ratio=0.5) is True


def test_native_integrity_check_rejects_significant_shrink() -> None:
    shrunk = _make_models(600)
    shrunk[FALLBACK_GENERALIZATIONS_KEY] = {"rules": []}
    registry: Final = ModelRegistry.parse(json.dumps(shrunk).encode())

    assert registry.check_size(reference=2000, minimum=50, ratio=0.5) is False


def test_native_registry_serves_aliases_in_order_and_compose_keeps_triples() -> None:
    registry: Final = ModelRegistry.parse(b'{"first":{"litellm_provider":"openai","aliases":["alias"],"mode":"chat"}}')
    served: Final = registry.served()
    layers: Final = ModelCostLayers()
    composed: Final = layers.compose(registry)

    assert list(served) == ["first", "alias"]
    assert served["first"] is served["alias"]
    assert isinstance(served, dict)
    assert all(len(row) == 3 for row in composed)
    assert [row[0] for row in composed] == ["first", "alias"]
    assert composed[0][1] is composed[1][1]


def test_native_compose_keys_keeps_tombstone_key() -> None:
    registry: Final = ModelRegistry.parse(b'{"model":{"litellm_provider":"openai","mode":"chat"}}')
    layers: Final = ModelCostLayers()
    layers.record_tombstone("removed")

    assert layers.compose_keys(registry, ["removed"]) == [("removed", None, frozenset())]


def test_bundled_raw_map_keeps_reserved_metadata() -> None:
    raw: Final = GetModelCostMap.load_local_model_cost_map()
    served: Final = _finalize_model_cost_map(raw)

    assert "sample_spec" in raw
    assert FALLBACK_GENERALIZATIONS_KEY in raw
    assert "sample_spec" not in served
    assert FALLBACK_GENERALIZATIONS_KEY not in served


def test_load_returns_catalog_without_mutating_process_state() -> None:
    base_registry: Final = GetModelCostMap._base_registry
    loaded_snapshot: Final = GetModelCostMap.loaded_model_cost_map()
    provenance: Final = get_model_cost_map_source_info()
    fallback_rules: Final = tuple(get_fallback_generalization_rules())
    loaded: Final = _load(NativeSource("bundled"))

    assert isinstance(loaded, LoadedCatalog)
    assert GetModelCostMap._base_registry is base_registry
    assert GetModelCostMap.loaded_model_cost_map() is loaded_snapshot
    assert get_model_cost_map_source_info() == provenance
    assert tuple(get_fallback_generalization_rules()) == fallback_rules


def test_remote_install_filters_before_integrity_and_keeps_state_unchanged(
    restore_installed_catalog_state: None,
) -> None:
    previous_registry: Final = GetModelCostMap.loaded_registry()
    previous_catalog: Final = GetModelCostMap.loaded_model_cost_map()
    previous_source_info: Final = get_model_cost_map_source_info()
    previous_fallbacks: Final = tuple(get_fallback_generalization_rules())
    previous_model_cost: Final = litellm.model_cost
    previous_entries: Final = dict(previous_model_cost)
    body: Final = json.dumps(
        {
            "accepted": {"litellm_provider": "openai", "mode": "chat"},
            "rejected": {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": "invalid"},
        }
    ).encode()
    catalog: Final = LoadedCatalog(ModelRegistry.parse(body), git_blob_id(body), '"etag"')
    provenance: Final[CostMapSourceInfo] = {
        **previous_source_info,
        "source": "remote",
        "url": "https://example.invalid/model_cost.json",
        "is_env_forced": False,
        "fallback_reason": None,
        "source_revision": catalog.revision,
        "etag": catalog.etag,
    }

    result: Final = install_loaded_catalog(catalog, provenance=provenance)

    assert isinstance(result, ModelCostMapUnavailable)
    assert result.reason == "model cost map failed integrity validation"
    assert litellm.model_cost is previous_model_cost
    assert dict(litellm.model_cost) == previous_entries
    assert GetModelCostMap.loaded_registry() is previous_registry
    assert GetModelCostMap.loaded_model_cost_map() is previous_catalog
    assert get_model_cost_map_source_info() == previous_source_info
    assert tuple(get_fallback_generalization_rules()) == previous_fallbacks


def test_live_direct_write_survives_reload_and_refreshes_caches_and_providers(
    restore_installed_catalog_state: None,
) -> None:
    live_map: Final = litellm.model_cost
    catalog_rows: Final = dict(live_map)
    key: Final = "litellm-plan04-direct-write-cache-model"
    openai_row: Final = {
        "litellm_provider": "openai",
        "mode": "chat",
        "input_cost_per_token": 0.000001,
    }
    anthropic_row: Final = {
        "litellm_provider": "anthropic",
        "mode": "chat",
        "input_cost_per_token": 0.000002,
    }
    assert key not in catalog_rows

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", litellm.ModelCostDirectWriteWarning)
        litellm.model_cost[key] = openai_row
    assert key in litellm.open_ai_chat_completion_models
    assert litellm.get_model_info(key)["input_cost_per_token"] == openai_row["input_cost_per_token"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", litellm.ModelCostDirectWriteWarning)
        litellm.model_cost[key] = anthropic_row
    assert key not in litellm.open_ai_chat_completion_models
    assert key in litellm.anthropic_models
    assert litellm.get_model_info(key)["litellm_provider"] == "anthropic"
    assert litellm.get_model_info(key)["input_cost_per_token"] == anthropic_row["input_cost_per_token"]

    assert adopt_model_cost_map(catalog_rows) > 0
    assert litellm.model_cost is live_map
    assert litellm.model_cost[key] == anthropic_row
    assert litellm.get_model_info(key)["input_cost_per_token"] == anthropic_row["input_cost_per_token"]


def test_direct_writes_and_registrations_replay_in_chronological_order(
    restore_installed_catalog_state: None,
) -> None:
    catalog_rows: Final = dict(litellm.model_cost)
    direct_before_registration: Final = "litellm-plan04-direct-before-registration"
    registration_before_direct: Final = "litellm-plan04-registration-before-direct"
    direct_first_row: Final = {
        "litellm_provider": "openai",
        "mode": "chat",
        "input_cost_per_token": 0.000011,
    }
    direct_last_row: Final = {
        "litellm_provider": "openai",
        "mode": "chat",
        "input_cost_per_token": 0.000044,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", litellm.ModelCostDirectWriteWarning)
        litellm.model_cost[direct_before_registration] = direct_first_row
    litellm.register_model(
        {
            direct_before_registration: {
                "litellm_provider": "openai",
                "output_cost_per_token": 0.000022,
            }
        },
        persist_across_reloads=True,
    )
    litellm.register_model(
        {
            registration_before_direct: {
                "litellm_provider": "openai",
                "output_cost_per_token": 0.000033,
            }
        },
        persist_across_reloads=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", litellm.ModelCostDirectWriteWarning)
        litellm.model_cost[registration_before_direct] = direct_last_row

    assert adopt_model_cost_map(catalog_rows) > 0
    assert litellm.model_cost[direct_before_registration]["input_cost_per_token"] == 0.000011
    assert litellm.model_cost[direct_before_registration]["output_cost_per_token"] == 0.000022
    assert litellm.model_cost[registration_before_direct] == direct_last_row


class ForeignModelCostMap(dict[str, dict[str, object]]):
    def __getitem__(self, key: str) -> dict[str, object]:
        return super().__getitem__(key) if key in self else {"foreign_default": True}


@pytest.mark.parametrize("foreign_subclass", [False, True])
def test_catalog_adoption_preserves_plain_and_foreign_map_identity(
    foreign_subclass: bool,
    monkeypatch: pytest.MonkeyPatch,
    restore_installed_catalog_state: None,
) -> None:
    cost_map: Final = dict(litellm.model_cost)
    target: Final = ForeignModelCostMap(cost_map) if foreign_subclass else cost_map
    monkeypatch.setattr(litellm, "model_cost", target)

    adopt_model_cost_map(cost_map)

    assert litellm.model_cost is target
    assert type(litellm.model_cost) is type(target)
    assert dict(litellm.model_cost) == cost_map
    if foreign_subclass:
        assert litellm.model_cost["missing-foreign-key"] == {"foreign_default": True}


def test_explicit_file_source_ignores_local_map_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    source_path: Final = Path(__file__).resolve().parents[3] / "model_prices_and_context_window.json"
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "true")

    loaded: Final = _load(NativeSource("file", str(source_path)))

    assert isinstance(loaded, LoadedCatalog)
    assert loaded.revision == git_blob_id(source_path.read_bytes())


def test_finalize_omits_reserved_keys_without_mutating_global_rules():
    previous = list(get_fallback_generalization_rules())
    try:
        set_fallback_generalizations(
            [{"name": "existing", "pattern": r"^existing-", "model_info": {"litellm_provider": "openai"}}]
        )
        raw = _make_models(2)
        raw[FALLBACK_GENERALIZATIONS_KEY] = {
            "rules": [
                {
                    "name": "rule",
                    "pattern": r"^widget-",
                    "model_info": {"litellm_provider": "openai"},
                }
            ]
        }
        finalized = _finalize_model_cost_map(raw)

        assert FALLBACK_GENERALIZATIONS_KEY not in finalized
        assert "sample_spec" not in finalized
        assert match_routing_generalization("existing-model") == "openai"
        assert match_routing_generalization("widget-9") is None
    finally:
        set_fallback_generalizations(previous)


def test_finalize_with_no_block_preserves_global_rules():
    previous = list(get_fallback_generalization_rules())
    try:
        set_fallback_generalizations(
            [{"name": "existing", "pattern": r"^x", "model_info": {"litellm_provider": "openai"}}]
        )
        _finalize_model_cost_map(_make_models(2))
        assert match_routing_generalization("x-1") == "openai"
    finally:
        set_fallback_generalizations(previous)


def test_shipped_backup_carries_the_claude_routing_rules():
    """The bundled backup must ship the Claude routing rules so a fresh install
    (or an offline fallback) routes unknown Claude models without code changes.
    Bedrock-syntax ids must hit the bedrock rule before the bare-id Anthropic rule."""
    backup = GetModelCostMap.load_local_model_cost_map()
    rules = backup.get(FALLBACK_GENERALIZATIONS_KEY, {}).get("rules", [])
    names = [r.get("name") for r in rules]
    assert names.index("bedrock-claude-ids") < names.index("anthropic-claude-ids")

    previous = list(get_fallback_generalization_rules())
    try:
        set_fallback_generalizations(rules)
        assert match_routing_generalization("claude-opus-4-9") == "anthropic"
        assert match_routing_generalization("global.anthropic.claude-opus-4-9") == "bedrock"
    finally:
        set_fallback_generalizations(previous)


def test_shipped_routing_rules_never_match_through_an_unrecognized_namespace():
    """Routing rules decide ``litellm_provider`` for otherwise-unknown ids, and the
    proxy's wildcard access check (``can_key_call_model`` with a ``bedrock/*`` key)
    trusts that inference: it rebuilds ``{provider}/{model}`` and matches it against
    the key's patterns. A routing pattern that matches as a substring lets
    ``bedrockz/anthropic.claude-...`` resolve to bedrock and slip through a
    ``bedrock/*`` key, so every shipped routing rule must anchor to the start of
    the name and never match an id carrying an unrecognized namespace prefix."""
    backup = GetModelCostMap.load_local_model_cost_map()
    rules = backup[FALLBACK_GENERALIZATIONS_KEY]["rules"]

    routing_rules = [r for r in rules if "litellm_provider" in r["model_info"]]
    assert routing_rules
    assert all(r["pattern"].startswith("^") for r in routing_rules)

    previous = list(get_fallback_generalization_rules())
    try:
        set_fallback_generalizations(rules)
        for bedrock_id in [
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "anthropic.claude-v2:1",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "us-gov.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "global.anthropic.claude-fable-5-20260120-v1:0",
        ]:
            assert match_routing_generalization(bedrock_id) == "bedrock", bedrock_id
        for namespaced in [
            "bedrockz/anthropic.claude-3-5-sonnet-20240620",
            "bedrockz/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "bedrockz/claude-3-5-sonnet-20240620",
        ]:
            assert match_routing_generalization(namespaced) is None, namespaced
    finally:
        set_fallback_generalizations(previous)


def test_shipped_backup_marks_claude_4_6_plus_adaptive_not_4_0():
    """Adaptive thinking is data, not code. The bundled backup must carry
    supports_adaptive_thinking on genuine Claude >= 4.6 entries (every provider
    route) and on the version-gated anthropic-claude-adaptive-thinking rule for
    unmapped future Claudes, while leaving the dated Claude 4.0 names
    ("...-4-20250514") unflagged so a date can never be mistaken for a 4.6+ minor
    version. The version-neutral claude-family-baseline capability rule must not flag
    it, so an unmapped sub-4.6 name resolves but stays non-adaptive. The adaptive rule
    carries only its delta; capability unioning stacks it onto the baseline, so the
    baseline block is never duplicated across rules and no rule needs ``extends``."""
    backup = GetModelCostMap.load_local_model_cost_map()

    rules = backup[FALLBACK_GENERALIZATIONS_KEY]["rules"]
    baseline_rule = next(r for r in rules if r.get("name") == "claude-family-baseline")
    adaptive_rule = next(r for r in rules if r.get("name") == "claude-adaptive-thinking")
    assert "supports_adaptive_thinking" not in baseline_rule["model_info"]
    assert "litellm_provider" not in baseline_rule["model_info"]
    assert adaptive_rule["model_info"] == {"supports_adaptive_thinking": True}
    assert all("extends" not in r for r in rules)

    for adaptive in [
        "anthropic.claude-opus-4-8",
        "vertex_ai/claude-opus-4-6@default",
        "us.anthropic.claude-sonnet-4-6",
        "azure_ai/claude-opus-4-7",
    ]:
        assert backup[adaptive]["supports_adaptive_thinking"] is True, adaptive

    for non_adaptive in [
        "us.anthropic.claude-opus-4-20250514-v1:0",
        "claude-opus-4-5",
    ]:
        assert "supports_adaptive_thinking" not in backup[non_adaptive], non_adaptive


def test_get_model_cost_map_stamps_loaded_at():
    """The load time feeds each pod's reload-due decision; a load that does not stamp it
    would make manual reload requests race the proxy's startup"""
    from datetime import datetime, timezone

    from litellm.litellm_core_utils import get_model_cost_map as module

    client, _calls = _mock_client([httpx.Response(200, content=_real_map_bytes())], client_cls=httpx.Client)

    before = datetime.now(timezone.utc)
    module.load_model_cost_map_at_import(url="https://example.invalid/cost_map.json", client=client)
    loaded_at = module.get_model_cost_map_loaded_at()

    assert loaded_at is not None
    assert before <= loaded_at <= datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# refetch_model_cost_map: retry/backoff behavior for runtime reloads
# ---------------------------------------------------------------------------

import functools
import random
from datetime import datetime, timezone

import httpx

from litellm.litellm_core_utils.get_model_cost_map import (
    ModelCostMapReloaded,
    ModelCostMapReloadUnavailable,
    refetch_model_cost_map,
)

_URL = "https://example.invalid/model_prices.json"


@functools.lru_cache(maxsize=1)
def _real_map_bytes() -> bytes:
    return json.dumps(_load_root_cost_map()).encode()


class _SleepRecorder:
    """Injected in place of asyncio.sleep so tests assert waits without real delay."""

    def __init__(self):
        self.waits = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


@pytest.fixture(autouse=True)
def _unset_local_cost_map_env(monkeypatch):
    """CI exports LITELLM_LOCAL_MODEL_COST_MAP=True; clear it so fetch behavior is deterministic."""
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)


def _mock_client(outcomes, client_cls=httpx.AsyncClient):
    """httpx client over a MockTransport serving one outcome per request; an exception instance is raised."""
    calls = {"count": 0}

    def handler(request):
        idx = min(calls["count"], len(outcomes) - 1)
        calls["count"] += 1
        outcome = outcomes[idx]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return client_cls(transport=httpx.MockTransport(handler)), calls


@pytest.mark.asyncio
async def test_refetch_retries_429_honoring_retry_after():
    """Two 429s with Retry-After then success: waits follow the header, not backoff."""
    client, calls = _mock_client(
        [
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, content=_real_map_bytes()),
        ]
    )
    sleeper = _SleepRecorder()
    result = await refetch_model_cost_map(url=_URL, sleep=sleeper, rng=random.Random(0), client=client)
    assert isinstance(result, ModelCostMapReloaded)
    assert len(result.model_cost_map) > 100
    assert calls["count"] == 3
    assert sleeper.waits == [7.0, 7.0]


@pytest.mark.asyncio
async def test_refetch_gives_up_after_max_attempts_with_exponential_backoff():
    """All 429 without Retry-After: exponential backoff waits, then a failure value."""
    client, calls = _mock_client([httpx.Response(429)])
    sleeper = _SleepRecorder()
    result = await refetch_model_cost_map(url=_URL, sleep=sleeper, rng=random.Random(0), client=client)
    assert isinstance(result, ModelCostMapReloadUnavailable)
    assert "429" in result.reason
    assert result.reason == "HTTP status 429"
    assert calls["count"] == 3
    assert len(sleeper.waits) == 2
    assert 2.0 <= sleeper.waits[0] < 3.0
    assert 4.0 <= sleeper.waits[1] < 5.0


@pytest.mark.asyncio
async def test_refetch_caps_retry_after_wait():
    """A hostile/huge Retry-After is capped so reloads never sleep unbounded."""
    client, _calls = _mock_client(
        [
            httpx.Response(429, headers={"Retry-After": "9999"}),
            httpx.Response(200, content=_real_map_bytes()),
        ]
    )
    sleeper = _SleepRecorder()
    result = await refetch_model_cost_map(url=_URL, sleep=sleeper, rng=random.Random(0), client=client)
    assert isinstance(result, ModelCostMapReloaded)
    assert sleeper.waits == [30.0]


@pytest.mark.asyncio
async def test_refetch_retries_transport_errors():
    """Connection failures are transient: retried like 5xx, succeeding when the network heals."""
    client, calls = _mock_client(
        [
            httpx.ConnectError("connection refused"),
            httpx.Response(200, content=_real_map_bytes()),
        ]
    )
    sleeper = _SleepRecorder()
    result = await refetch_model_cost_map(url=_URL, sleep=sleeper, rng=random.Random(0), client=client)
    assert isinstance(result, ModelCostMapReloaded)
    assert calls["count"] == 2
    assert len(sleeper.waits) == 1


@pytest.mark.asyncio
async def test_refetch_non_retryable_status_fails_immediately():
    """A 404 is permanent: one attempt, no sleeps, failure value."""
    client, calls = _mock_client([httpx.Response(404)])
    sleeper = _SleepRecorder()
    result = await refetch_model_cost_map(url=_URL, sleep=sleeper, rng=random.Random(0), client=client)
    assert isinstance(result, ModelCostMapReloadUnavailable)
    assert "404" in result.reason
    assert calls["count"] == 1
    assert sleeper.waits == []


@pytest.mark.asyncio
async def test_refetch_invalid_json_fails_immediately():
    client, calls = _mock_client([httpx.Response(200, content=b"not json")])
    sleeper = _SleepRecorder()
    result = await refetch_model_cost_map(url=_URL, sleep=sleeper, rng=random.Random(0), client=client)
    assert isinstance(result, ModelCostMapReloadUnavailable)
    assert "invalid JSON" in result.reason
    assert calls["count"] == 1
    assert sleeper.waits == []


@pytest.mark.asyncio
async def test_refetch_shrunk_map_fails_integrity_not_swapped_in():
    """A drastically shrunk upstream file is rejected instead of being adopted."""
    tiny = json.dumps(_make_models(60)).encode()
    client, _calls = _mock_client([httpx.Response(200, content=tiny)])
    result = await refetch_model_cost_map(url=_URL, sleep=_SleepRecorder(), rng=random.Random(0), client=client)
    assert isinstance(result, ModelCostMapReloadUnavailable)
    assert "below 0.5 of reference count" in result.reason


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["LITELLM_LOCAL_MODEL_COST_MAP", "LITELLM_OFFLINE_MODE"])
async def test_refetch_respects_local_env_override(monkeypatch, flag):
    """LITELLM_LOCAL_MODEL_COST_MAP=True or LITELLM_OFFLINE_MODE=True short-circuits to the bundled backup, zero HTTP."""
    monkeypatch.setenv(flag, "True")

    def _fail(request):
        raise AssertionError("no HTTP request should be made when local map is forced")

    result = await refetch_model_cost_map(
        url=_URL,
        sleep=_SleepRecorder(),
        rng=random.Random(0),
        client=httpx.AsyncClient(transport=httpx.MockTransport(_fail)),
    )
    assert isinstance(result, ModelCostMapReloaded)
    assert len(result.model_cost_map) > 100


@pytest.mark.asyncio
async def test_refetch_records_the_blob_id_of_the_bytes_served_and_the_fetch_etag():
    body = _real_map_bytes()
    client, _ = _mock_client([httpx.Response(200, headers={"ETag": 'W/"abc123"'}, content=body)])

    result = await refetch_model_cost_map(url=_URL, sleep=_SleepRecorder(), rng=random.Random(0), client=client)

    assert isinstance(result, ModelCostMapReloaded)
    assert result.revision == git_blob_id(body)
    assert result.etag == 'W/"abc123"'
    assert get_model_cost_map_provenance() == {"source_revision": git_blob_id(body), "etag": 'W/"abc123"'}


@pytest.mark.asyncio
async def test_refetch_client_sends_etag_only_for_matching_source() -> None:
    request_etags: Final[list[str | None]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        request_etags.append(request.headers.get("If-None-Match"))
        return httpx.Response(200, headers={"ETag": '"conditional"'}, content=_real_map_bytes())

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        first: Final = await refetch_model_cost_map(
            url=_URL, sleep=_SleepRecorder(), rng=random.Random(0), client=client
        )
        same_source: Final = await refetch_model_cost_map(
            url=_URL, sleep=_SleepRecorder(), rng=random.Random(0), client=client
        )
        different_source: Final = await refetch_model_cost_map(
            url="https://different.example/model_prices.json",
            sleep=_SleepRecorder(),
            rng=random.Random(0),
            client=client,
        )

    assert isinstance(first, ModelCostMapReloaded)
    assert isinstance(same_source, ModelCostMapReloaded)
    assert isinstance(different_source, ModelCostMapReloaded)
    assert request_etags[-2:] == ['"conditional"', None]


@pytest.mark.asyncio
async def test_loaded_catalog_snapshot_follows_the_fetched_map_and_ignores_later_registrations(monkeypatch):
    import litellm

    edited = json.loads(_real_map_bytes())
    edited["gpt-5.4-mini"]["max_input_tokens"] = 777
    client, _ = _mock_client([httpx.Response(200, content=json.dumps(edited).encode())])

    result = await refetch_model_cost_map(url=_URL, sleep=_SleepRecorder(), rng=random.Random(0), client=client)

    assert isinstance(result, ModelCostMapReloaded)
    monkeypatch.setattr(litellm, "model_cost", result.model_cost_map)
    litellm.register_model({"gpt-5.4-mini": {"max_input_tokens": 2048}}, persist_across_reloads=False)
    assert litellm.model_cost["gpt-5.4-mini"]["max_input_tokens"] == 2048
    assert GetModelCostMap.loaded_model_cost_map()["gpt-5.4-mini"]["max_input_tokens"] == 777


@pytest.mark.asyncio
async def test_loaded_catalog_fields_follow_each_fetched_map():
    added = json.loads(_real_map_bytes())
    added["gpt-5.4-mini"]["supports_litellm_qa_capability"] = True
    with_field, _ = _mock_client([httpx.Response(200, content=json.dumps(added).encode())])
    without_field, _ = _mock_client([httpx.Response(200, content=_real_map_bytes())])

    await refetch_model_cost_map(url=_URL, sleep=_SleepRecorder(), rng=random.Random(0), client=with_field)
    fields_after_adding: Final = GetModelCostMap.loaded_model_cost_map_fields()
    await refetch_model_cost_map(url=_URL, sleep=_SleepRecorder(), rng=random.Random(0), client=without_field)

    assert "supports_litellm_qa_capability" in fields_after_adding
    assert "supports_litellm_qa_capability" not in GetModelCostMap.loaded_model_cost_map_fields()
    assert "max_input_tokens" in GetModelCostMap.loaded_model_cost_map_fields()


@pytest.mark.asyncio
async def test_refetch_revision_follows_the_bytes_not_the_url():
    edited = json.loads(_real_map_bytes())
    edited["gpt-5.4-mini"]["input_cost_per_token"] = 0.5
    client, _ = _mock_client(
        [httpx.Response(200, content=_real_map_bytes()), httpx.Response(200, content=json.dumps(edited).encode())]
    )

    first = await refetch_model_cost_map(url=_URL, sleep=_SleepRecorder(), rng=random.Random(0), client=client)
    second = await refetch_model_cost_map(url=_URL, sleep=_SleepRecorder(), rng=random.Random(0), client=client)

    assert isinstance(first, ModelCostMapReloaded) and isinstance(second, ModelCostMapReloaded)
    assert first.revision != second.revision
    assert get_model_cost_map_provenance()["source_revision"] == second.revision


@pytest.mark.asyncio
async def test_refetch_local_override_reports_the_bundled_blob_id_without_an_etag(monkeypatch):
    remote, _ = _mock_client([httpx.Response(200, headers={"ETag": 'W/"remote"'}, content=_real_map_bytes())])
    await refetch_model_cost_map(url=_URL, sleep=_SleepRecorder(), rng=random.Random(0), client=remote)
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    result = await refetch_model_cost_map(url=_URL, sleep=_SleepRecorder(), rng=random.Random(0))

    assert isinstance(result, ModelCostMapReloaded)
    assert result.revision == _bundled_blob_id()
    assert get_model_cost_map_provenance() == {"source_revision": _bundled_blob_id(), "etag": None}


@pytest.mark.asyncio
async def test_refetch_stamps_loaded_at_on_remote_and_local_reloads(monkeypatch):
    from litellm.litellm_core_utils import get_model_cost_map as module

    client, _ = _mock_client([httpx.Response(200, content=_real_map_bytes())])
    before_remote = datetime.now(timezone.utc)
    await refetch_model_cost_map(url=_URL, sleep=_SleepRecorder(), rng=random.Random(0), client=client)
    remote_loaded_at = module.get_model_cost_map_loaded_at()
    assert remote_loaded_at is not None
    assert before_remote <= remote_loaded_at <= datetime.now(timezone.utc)

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    before_local = datetime.now(timezone.utc)
    await refetch_model_cost_map(url=_URL, sleep=_SleepRecorder(), rng=random.Random(0))
    local_loaded_at = module.get_model_cost_map_loaded_at()
    assert local_loaded_at is not None
    assert before_local <= local_loaded_at <= datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# load_model_cost_map_at_import: boot-time retries transient failures like a reload does
# ---------------------------------------------------------------------------

from unittest.mock import patch

from litellm.litellm_core_utils.get_model_cost_map import (
    load_model_cost_map_at_import,
)
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.utils import _invalidate_model_cost_lowercase_map
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome
from tests.unit.litellm_core_utils.test_model_cost_api import _local_responses


class _FakeS3Body:
    def __init__(self, content: bytes) -> None:
        self.content: Final = content

    def read(self) -> bytes:
        return self.content


class _FakeS3GetObjectResponse(TypedDict):
    Body: ReadOnly[_FakeS3Body]
    ETag: NotRequired[ReadOnly[str]]


class _FakeS3Client:
    def __init__(self, content: bytes) -> None:
        self.content: Final = content
        self.calls: list[tuple[str, str]] = []

    def get_object(self, *, Bucket: str, Key: str) -> _FakeS3GetObjectResponse:
        self.calls.append((Bucket, Key))
        return {"Body": _FakeS3Body(self.content), "ETag": '"fake-etag"'}


def test_s3_integrity_excludes_rejected_rows_but_preserves_report_rows() -> None:
    invalid_row: Final = {
        "litellm_provider": "openai",
        "mode": "chat",
        "input_cost_per_token": "invalid",
    }
    raw_large_map: Final = {
        **_make_models(2100),
        **{f"invalid-{index}": invalid_row for index in range(2100)},
    }

    rejected: Final = _classify_fetch_content(json.dumps(raw_large_map).encode(), "s3://bucket/model_prices.json", None)

    assert isinstance(rejected, ModelCostMapUnavailable)
    assert "integrity validation" in rejected.reason

    report_map: Final = {**_make_models(2200), "invalid-report-row": invalid_row}
    loaded: Final = _classify_fetch_content(json.dumps(report_map).encode(), "s3://bucket/model_prices.json", None)

    assert isinstance(loaded, LoadedCatalog)
    assert "invalid-report-row" in loaded.registry.served()


def test_s3_boot_load_keeps_bundled_catalog_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.litellm_core_utils import get_model_cost_map as module

    source_info = module._cost_map_source_info
    for name in ("source", "url", "is_env_forced", "fallback_reason", "loaded_at", "source_revision", "etag"):
        monkeypatch.setattr(source_info, name, getattr(source_info, name))

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    s3_content: Final = _make_models(len(GetModelCostMap.load_local_model_cost_map()))
    s3_client: Final = _FakeS3Client(json.dumps(s3_content).encode())
    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: s3_client)

    cost_map: Final = load_model_cost_map_at_import("s3://my-bucket/prices/model_prices.json")

    assert cost_map == _finalize_model_cost_map(GetModelCostMap.load_local_model_cost_map())
    assert get_model_cost_map_source_info()["source"] == "local"
    assert get_model_cost_map_source_info()["fallback_reason"] == "remote revalidation pending"
    assert s3_client.calls == []


class _SyncSleepRecorder:
    """Injected in place of time.sleep so the boot path's waits are asserted without delay."""

    def __init__(self, block=False):
        self.waits = []
        self.block = block
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, seconds: float) -> None:
        if self.block:
            self.started.set()
            self.release.wait(timeout=10)
        self.waits.append(seconds)


def _retry_threads():
    return [thread for thread in threading.enumerate() if thread.name == "litellm-model-cost-map-retry"]


@pytest.mark.parametrize("status", [200, 503])
def test_get_model_cost_map_side_read_preserves_installed_state(
    monkeypatch: pytest.MonkeyPatch,
    restore_installed_catalog_state: None,
    status: int,
) -> None:
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    monkeypatch.delenv("LITELLM_OFFLINE_MODE", raising=False)
    live_model_cost: Final = litellm.model_cost
    live_entries: Final = copy.deepcopy(dict(live_model_cost))
    source_info: Final = get_model_cost_map_source_info()
    loaded_registry: Final = GetModelCostMap.loaded_registry()
    loaded_catalog: Final = GetModelCostMap.loaded_model_cost_map()
    fallback_rules: Final = copy.deepcopy(get_fallback_generalization_rules())
    layer_state: Final = _MODEL_COST_LAYERS.compose(ModelRegistry.bundled())
    retry_threads: Final = frozenset(thread.ident for thread in _retry_threads())
    bundled: Final = ModelRegistry.bundled().served()
    side_key: Final = "side-read-only-model"
    side_row: Final = {
        "litellm_provider": "openai",
        "mode": "chat",
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000002,
        "metadata": {"labels": [{"source": "side-read"}]},
    }
    payload: Final = json.dumps({**bundled, side_key: side_row}).encode()
    response_body: Final = payload if status == 200 else b"unavailable"

    with _local_responses(((status, response_body),)) as responses:
        result: Final = litellm.get_model_cost_map(url=responses.url, max_attempts=9)
        assert responses.requests.get(timeout=5) is None
        assert responses.requests.empty()

    assert type(result) is dict
    assert all(type(row) is dict for row in result.values())
    if status == 200:
        assert result[side_key]["metadata"] == {"labels": [{"source": "side-read"}]}
        metadata: Final = result[side_key]["metadata"]
        assert type(metadata) is dict
        labels: Final = metadata["labels"]
        assert type(labels) is list
        assert type(labels[0]) is dict
        labels[0]["source"] = "mutated"  # mutable-ok: prove side-read nested rows are independent
        assert side_key not in ModelRegistry.bundled().served()
    else:
        assert result == bundled
        bundled_price: Final = bundled["gpt-5.4-mini"]["input_cost_per_token"]
        result["gpt-5.4-mini"]["input_cost_per_token"] = "mutated"  # mutable-ok: fallback rows are independent
        assert ModelRegistry.bundled().served()["gpt-5.4-mini"]["input_cost_per_token"] == bundled_price

    result["side-read-mutation"] = {}  # mutable-ok: verify the returned map is writable
    assert "side-read-mutation" not in ModelRegistry.bundled().served()
    assert litellm.model_cost is live_model_cost
    assert dict(litellm.model_cost) == live_entries
    assert get_model_cost_map_source_info() == source_info
    assert GetModelCostMap.loaded_registry() is loaded_registry
    assert GetModelCostMap.loaded_model_cost_map() is loaded_catalog
    assert get_fallback_generalization_rules() == fallback_rules
    assert _MODEL_COST_LAYERS.compose(ModelRegistry.bundled()) == layer_state
    assert frozenset(thread.ident for thread in _retry_threads()) <= retry_threads


def test_url_registration_fetches_the_requested_map_when_local_flag_is_set(
    monkeypatch: pytest.MonkeyPatch,
    restore_installed_catalog_state: None,
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    row: Final = copy.deepcopy(_load_root_cost_map()["gpt-5.4-mini"])
    side_key: Final = "side-url-registration-model"
    payload: Final = json.dumps({**_load_root_cost_map(), side_key: row}).encode()
    live_model_cost: Final = litellm.model_cost
    source_info: Final = get_model_cost_map_source_info()
    loaded_registry: Final = GetModelCostMap.loaded_registry()
    loaded_catalog: Final = GetModelCostMap.loaded_model_cost_map()
    fallback_rules: Final = copy.deepcopy(get_fallback_generalization_rules())

    with _local_responses(((200, payload),)) as responses:
        litellm.register_model(responses.url)
        assert responses.requests.get(timeout=5) is None
        assert responses.requests.empty()

    assert litellm.model_cost is live_model_cost
    assert side_key in litellm.model_cost
    assert litellm.model_cost[side_key]["litellm_provider"] == row["litellm_provider"]
    assert get_model_cost_map_source_info() == source_info
    assert GetModelCostMap.loaded_registry() is loaded_registry
    assert GetModelCostMap.loaded_model_cost_map() is loaded_catalog
    assert get_fallback_generalization_rules() == fallback_rules


def test_url_registration_404_does_not_change_model_cost_state(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    restore_installed_catalog_state: None,
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    live_model_cost: Final = litellm.model_cost
    live_entries: Final = copy.deepcopy(dict(live_model_cost))
    source_info: Final = get_model_cost_map_source_info()
    loaded_registry: Final = GetModelCostMap.loaded_registry()
    loaded_catalog: Final = GetModelCostMap.loaded_model_cost_map()
    fallback_rules: Final = copy.deepcopy(get_fallback_generalization_rules())
    layer_state: Final = _MODEL_COST_LAYERS.compose(ModelRegistry.bundled())
    warning_start: Final = len(caplog.records)

    with _local_responses(((404, b"missing"),)) as responses:
        litellm.register_model(responses.url)
        assert responses.requests.get(timeout=5) is None
        assert responses.requests.empty()

    warning_records: Final = tuple(
        record
        for record in caplog.records[warning_start:]
        if record.name == "LiteLLM" and record.levelno == logging.WARNING
    )
    assert len(warning_records) == 1
    assert "model cost URL registration failed" in warning_records[0].getMessage()
    assert "404" in warning_records[0].getMessage()
    assert litellm.model_cost is live_model_cost
    assert dict(litellm.model_cost) == live_entries
    assert get_model_cost_map_source_info() == source_info
    assert GetModelCostMap.loaded_registry() is loaded_registry
    assert GetModelCostMap.loaded_model_cost_map() is loaded_catalog
    assert get_fallback_generalization_rules() == fallback_rules
    assert _MODEL_COST_LAYERS.compose(ModelRegistry.bundled()) == layer_state


def test_import_serves_bundled_catalog_before_held_response_then_adopts_remote() -> None:
    from tests.unit.litellm_core_utils.test_model_cost_api import _local_responses, _LocalResponse

    remote_entries = json.loads(_real_map_bytes())
    remote_entries["swr/subprocess-test"] = {
        "litellm_provider": "openai",
        "mode": "chat",
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000002,
    }
    response_started: Final = threading.Event()
    release_response: Final = threading.Event()
    payload: Final = json.dumps(remote_entries).encode()
    script: Final = """
import litellm
import warnings
from litellm.litellm_core_utils.model_cost_revalidate import REVALIDATOR

warnings.filterwarnings("ignore", category=litellm.ModelCostDirectWriteWarning)
live_map = litellm.model_cost
print("bundled", len(live_map) > 100, flush=True)
print("lookup", litellm.get_model_info("gpt-4o")["litellm_provider"], flush=True)
live_map["direct/write-survives-revalidation"] = {"litellm_provider": "openai"}
print("direct-write", flush=True)
worker = REVALIDATOR._thread
if worker is not None:
    worker.join(timeout=30)
print("adopted", litellm.model_cost is live_map and "swr/subprocess-test" in live_map, flush=True)
print("direct-write-preserved", "direct/write-survives-revalidation" in live_map, flush=True)
"""

    with _local_responses(
        (
            _LocalResponse(
                200,
                payload,
                started=response_started,
                release=release_response,
            ),
        )
    ) as responses:
        child: Final = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={
                **os.environ,
                "LITELLM_LOCAL_MODEL_COST_MAP": "False",
                "LITELLM_MODEL_COST_MAP_URL": responses.url,
            },
        )

        def read_line() -> str:
            assert child.stdout is not None
            ready, _, _ = select.select([child.stdout], [], [], 30)
            assert ready, "timed out waiting for the import subprocess"
            return child.stdout.readline().strip()

        try:
            assert response_started.wait(timeout=30)
            assert read_line() == "bundled True"
            assert read_line() == "lookup openai"
            assert read_line() == "direct-write"
            release_response.set()
            assert read_line() == "adopted True"
            assert read_line() == "direct-write-preserved True"
            child.wait(timeout=30)
        finally:
            release_response.set()
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=10)

    assert responses.requests.qsize() == 1


def test_boot_load_installs_bundled_map_and_records_pending_revalidation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    client, calls = _mock_client([httpx.Response(200, content=_real_map_bytes())], client_cls=httpx.Client)
    cost_map = load_model_cost_map_at_import(url=_URL, client=client)
    assert calls["count"] == 0
    assert cost_map.keys() >= _load_root_cost_map().keys() - {"sample_spec", FALLBACK_GENERALIZATIONS_KEY}
    source = get_model_cost_map_source_info()
    assert source["source"] == "local"
    assert source["url"] == _URL
    assert source["fallback_reason"] == "remote revalidation pending"


def test_boot_load_does_not_block_on_remote_revalidation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    client, calls = _mock_client([httpx.Response(503)], client_cls=httpx.Client)
    cost_map = load_model_cost_map_at_import(url=_URL, client=client)
    assert calls["count"] == 0
    assert cost_map
    assert get_model_cost_map_source_info()["fallback_reason"] == "remote revalidation pending"


def test_boot_load_keeps_pending_state_before_background_worker(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    client, calls = _mock_client([httpx.Response(404)], client_cls=httpx.Client)
    load_model_cost_map_at_import(url=_URL, client=client)
    assert calls["count"] == 0
    assert get_model_cost_map_source_info()["fallback_reason"] == "remote revalidation pending"


def test_boot_load_accepts_url_without_synchronous_validation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    client, calls = _mock_client([httpx.Response(200, content=_real_map_bytes())], client_cls=httpx.Client)
    cost_map = load_model_cost_map_at_import(url="http://[bad/model_prices.json", client=client)
    assert calls["count"] == 0
    assert cost_map.keys() >= _load_root_cost_map().keys() - {"sample_spec", FALLBACK_GENERALIZATIONS_KEY}
    source = get_model_cost_map_source_info()
    assert source["source"] == "local"
    assert source["fallback_reason"] == "remote revalidation pending"


@pytest.mark.parametrize("flag", ["LITELLM_LOCAL_MODEL_COST_MAP", "LITELLM_OFFLINE_MODE"])
def test_boot_load_respects_local_env_override(monkeypatch, flag):
    """LITELLM_LOCAL_MODEL_COST_MAP=True or LITELLM_OFFLINE_MODE=True still short-circuits to the backup with zero HTTP."""
    monkeypatch.setenv(flag, "True")

    def _fail(request):
        raise AssertionError("no HTTP request should be made when local map is forced")

    cost_map = load_model_cost_map_at_import(
        url=_URL,
        sleep=_SyncSleepRecorder(),
        client=httpx.Client(transport=httpx.MockTransport(_fail)),
    )
    assert len(cost_map) > 100
    assert get_model_cost_map_source_info()["fallback_reason"] == "remote revalidation pending"


@pytest.mark.parametrize(
    ("argv0", "request_count"),
    [
        ("/some/venv/bin/lite", 0),
        ("/some/venv/bin/lite.exe", 0),
        ("/some/venv/bin/python", 1),
    ],
)
def test_boot_load_skips_remote_fetch_for_cli_processes(
    monkeypatch: pytest.MonkeyPatch, argv0: str, request_count: int
) -> None:
    monkeypatch.setattr(sys, "argv", [argv0, "--version"])
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    client, calls = _mock_client([httpx.Response(200, content=_real_map_bytes())], client_cls=httpx.Client)

    cost_map = load_model_cost_map_at_import(url=_URL, client=client)

    assert calls["count"] == 0
    assert cost_map
    source = get_model_cost_map_source_info()
    assert source["source"] == "local"
    assert source["fallback_reason"] == "remote revalidation pending"


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def setup_and_teardown(event_loop):
    import litellm

    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    importlib.reload(litellm)
    asyncio.set_event_loop(event_loop)
    yield
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    pending = asyncio.all_tasks(event_loop)
    for task in pending:
        task.cancel()
    if pending:
        event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


_SCALAR_DEFAULTS = {
    "num_retries": getattr(litellm, "num_retries", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
    "cohere_key": getattr(litellm, "cohere_key", None),
}


@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param(httpx.Response(200, content=b"not json"), id="invalid-json"),
        pytest.param(httpx.ConnectError("Connection refused"), id="network-error"),
        pytest.param(httpx.Response(200, content=b"{}"), id="empty-map"),
        pytest.param(httpx.Response(200, content=b"[]"), id="non-object-json"),
        pytest.param(
            httpx.Response(
                200,
                content=json.dumps({f"model-{index}": {"litellm_provider": "test"} for index in range(11)}).encode(),
            ),
            id="undersized-map",
        ),
    ],
)
def test_boot_loader_serves_bundled_catalog_without_waiting_for_remote_outcome(
    monkeypatch: pytest.MonkeyPatch,
    restore_installed_catalog_state: None,
    outcome: object,
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    client, calls = _mock_client([outcome], client_cls=httpx.Client)

    cost_map: Final = load_model_cost_map_at_import(url=_URL, max_attempts=1, client=client)

    assert calls["count"] == 0
    assert cost_map.keys() == _finalize_model_cost_map(GetModelCostMap.load_local_model_cost_map()).keys()
    source: Final = get_model_cost_map_source_info()
    assert source["source"] == "local"
    assert source["fallback_reason"] == "remote revalidation pending"


def test_should_use_local_map_when_env_var_set(monkeypatch):
    """LITELLM_LOCAL_MODEL_COST_MAP=True should skip remote fetch entirely."""
    with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}):
        with _local_responses(((200, _real_map_bytes()),)) as responses:
            result = litellm.get_model_cost_map(url=responses.url)
            assert responses.requests.empty()

    assert result == ModelRegistry.bundled().served()


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestBackupModelCostMapExists:
    """Validates the local backup file is always present and valid."""

    def test_should_have_backup_file(self):
        """The backup model cost map must exist and be loadable."""
        backup = GetModelCostMap.load_local_model_cost_map()
        assert isinstance(backup, dict)
        assert len(backup) > 0, "Backup model cost map is empty"

    def test_should_have_minimum_models_in_backup(self):
        """The backup must contain a reasonable number of models."""
        backup = GetModelCostMap.load_local_model_cost_map()
        assert len(backup) > 100, f"Backup has only {len(backup)} models, expected > 100"


@pytest.mark.usefixtures("setup_and_teardown")
class TestBundledCatalogAtImport:
    def test_model_info_uses_bundled_catalog(self):
        bundled_map = load_model_cost_map_at_import("https://fake-url.com/model_prices.json")
        assert get_model_cost_map_source_info()["fallback_reason"] == "remote revalidation pending"
        original = litellm.model_cost
        litellm.model_cost = bundled_map
        try:
            info = litellm.get_model_info("gpt-4o")
            assert info is not None
            assert info["input_cost_per_token"] > 0
        finally:
            litellm.model_cost = original

    def test_completion_uses_bundled_catalog(self):
        bundled_map = load_model_cost_map_at_import("https://fake-url.com/model_prices.json")
        assert get_model_cost_map_source_info()["fallback_reason"] == "remote revalidation pending"
        original = litellm.model_cost
        litellm.model_cost = bundled_map
        try:
            # mock_response goes through the real completion path —
            # routing, cost calculator, logging — but skips the HTTP call
            response = litellm.completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "say hi"}],
                mock_response="hello from mock",
            )
            assert response is not None
            assert response.choices[0].message.content == "hello from mock"
        finally:
            litellm.model_cost = original


@pytest.fixture()
def _vcr_outcome_gate_local_testing(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="function")
def isolate_litellm_state():
    """
    Per-function isolation fixture.

    Resets litellm globals to their true defaults before each test and
    restores them afterward, so tests don't leak side effects.
    Works safely under pytest-xdist parallel execution.
    """
    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in ("pre_call_rules", "post_call_rules"):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS_local_testing:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
        "pre_call_rules",
        "post_call_rules",
    ):
        if hasattr(litellm, attr):
            setattr(litellm, attr, [])
    for attr, default_val in _SCALAR_DEFAULTS_local_testing.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, default_val)
    yield
    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    _invalidate_model_cost_lowercase_map()


_SCALAR_DEFAULTS_local_testing = {
    "num_retries": getattr(litellm, "num_retries", None),
    "num_retries_per_request": getattr(litellm, "num_retries_per_request", None),
    "request_timeout": getattr(litellm, "request_timeout", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "default_fallbacks": getattr(litellm, "default_fallbacks", None),
    "enable_azure_ad_token_refresh": getattr(litellm, "enable_azure_ad_token_refresh", None),
    "tag_budget_config": getattr(litellm, "tag_budget_config", None),
    "model_cost": getattr(litellm, "model_cost", None),
    "token_counter": getattr(litellm, "token_counter", None),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
}


@pytest.fixture(scope="module")
def setup_and_teardown_local_testing():
    """
    Module-scoped setup. Reloads litellm only in single-process mode
    (skipped under xdist to avoid cross-worker interference).
    """
    import litellm

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", None)
    if worker_id is None:
        importlib.reload(litellm)
        try:
            if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
                import litellm.proxy.proxy_server

                importlib.reload(litellm.proxy.proxy_server)
        except Exception:
            pass
        if hasattr(litellm, "in_memory_llm_clients_cache"):
            litellm.in_memory_llm_clients_cache.flush_cache()
    yield


@pytest.mark.usefixtures(
    "_vcr_outcome_gate_local_testing",
    "isolate_litellm_state",
    "setup_and_teardown_local_testing",
)
def test_get_model_cost_map():
    try:
        print(litellm.get_model_cost_map(url="fake-url"))
    except Exception as e:
        pytest.fail(f"An exception occurred: {e}")


def test_the_embedded_map_is_the_root_json_exactly_as_python_reads_it():
    loaded: Final = GetModelCostMap.load_local_model_cost_map()
    expected: Final = _load_root_cost_map()

    assert type(loaded) is dict
    assert all(type(entry) is dict for entry in loaded.values())
    assert loaded == expected
    assert json.dumps(loaded) == json.dumps(expected)
    assert GetModelCostMap.read_local_model_cost_map_bytes() == Path(_ROOT_COST_MAP_PATH).read_bytes()


def _read_every_key(model_cost: dict) -> dict[str, tuple[object, object, object]]:
    get_model_cost_map_module.adopt_model_cost_map(model_cost)
    litellm.get_model_info.cache_clear()

    def outcome(read: Callable[[], object]) -> object:
        try:
            return read()
        except Exception as error:  # noqa: BLE001  # the exception type is the behavior being compared
            return type(error).__name__

    return {
        key: (
            outcome(lambda key=key: litellm.get_model_info(key)),
            outcome(lambda key=key: litellm.cost_per_token(model=key, prompt_tokens=1000, completion_tokens=100)),
            model_cost[key],
        )
        for key in model_cost
    }


@pytest.mark.usefixtures("isolate_litellm_state")
def test_every_key_reads_the_same_as_the_json_parsed_map(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from_json: Final = _read_every_key(_finalize_model_cost_map(_load_root_cost_map()))
    native: Final = _read_every_key(litellm.get_model_cost_map(url=_URL))

    assert native.keys() == from_json.keys()
    assert [key for key in native if native[key] != from_json[key]] == []


@pytest.mark.asyncio
async def test_a_fetched_map_drops_a_bad_entry_and_logs_it_by_model_and_key():
    edited = json.loads(_real_map_bytes())
    edited["gpt-5.4-mini"]["mode"] = "telepathy"
    client, _ = _mock_client([httpx.Response(200, content=json.dumps(edited).encode())])
    records: Final[list[logging.LogRecord]] = []
    handler: Final = logging.Handler()
    handler.emit = records.append
    logger: Final = logging.getLogger("LiteLLM")
    logger.addHandler(handler)
    try:
        result = await refetch_model_cost_map(url=_URL, sleep=_SleepRecorder(), rng=random.Random(0), client=client)
    finally:
        logger.removeHandler(handler)

    assert isinstance(result, ModelCostMapReloaded)
    assert "gpt-5.4-mini" not in result.model_cost_map
    assert any("gpt-5.4-mini.mode" in record.getMessage() for record in records)
