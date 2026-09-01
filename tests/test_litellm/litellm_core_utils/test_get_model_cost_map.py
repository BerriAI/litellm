"""
Tests for model-cost-map loading: the model-count integrity check (which must
count actual model entries, not reserved meta keys) and the extraction of the
``fallback_generalizations`` block out of the raw map.
"""

import json
import os

import pytest


from litellm.litellm_core_utils.fallback_generalizations import (
    get_fallback_generalization_rules,
    match_capability_generalizations,
    match_routing_generalization,
    set_fallback_generalizations,
)
from litellm.litellm_core_utils.get_model_cost_map import (
    FALLBACK_GENERALIZATIONS_KEY,
    GetModelCostMap,
    _count_model_entries,
    _finalize_model_cost_map,
)


def _load_root_cost_map() -> dict:
    path = os.path.join(
        os.path.dirname(__file__), "../../../model_prices_and_context_window.json"
    )
    with open(path) as f:
        return json.load(f)


def _make_models(n: int) -> dict:
    return {
        f"model-{i}": {"litellm_provider": "openai", "mode": "chat"} for i in range(n)
    }


def test_count_model_entries_excludes_reserved_keys():
    m = _make_models(3)
    m["sample_spec"] = {"foo": "bar"}
    m[FALLBACK_GENERALIZATIONS_KEY] = {"rules": []}
    assert _count_model_entries(m) == 3


def test_validation_rejects_truly_shrunk_file_even_with_meta_keys():
    """A file with only a handful of real models must be rejected as corrupt,
    and the extra meta keys must not inflate the count past the minimum."""
    shrunk = _make_models(5)
    shrunk["sample_spec"] = {"foo": "bar"}
    shrunk[FALLBACK_GENERALIZATIONS_KEY] = {"rules": [{"name": "x"}]}

    assert (
        GetModelCostMap.validate_model_cost_map(
            fetched_map=shrunk,
            backup_model_count=2000,
            min_model_count=50,
        )
        is False
    )


def test_validation_accepts_healthy_file_with_meta_keys():
    healthy = _make_models(2000)
    healthy["sample_spec"] = {"foo": "bar"}
    healthy[FALLBACK_GENERALIZATIONS_KEY] = {"rules": []}

    assert (
        GetModelCostMap.validate_model_cost_map(
            fetched_map=healthy,
            backup_model_count=2000,
            min_model_count=50,
        )
        is True
    )


def test_validation_rejects_significant_shrink_vs_backup():
    # 600 real models vs a 2000-model backup is below the 50% shrink threshold.
    shrunk = _make_models(600)
    shrunk[FALLBACK_GENERALIZATIONS_KEY] = {"rules": []}
    assert (
        GetModelCostMap.validate_model_cost_map(
            fetched_map=shrunk,
            backup_model_count=2000,
            min_model_count=50,
            max_shrink_ratio=0.5,
        )
        is False
    )


def test_finalize_pops_key_and_installs_rules():
    previous = list(get_fallback_generalization_rules())
    try:
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

        # The reserved key is removed from the returned model map ...
        assert FALLBACK_GENERALIZATIONS_KEY not in finalized
        # ... and its rules are installed into the generalizations module.
        assert match_routing_generalization("widget-9") == "openai"
    finally:
        set_fallback_generalizations(previous)


def test_finalize_with_no_block_clears_rules():
    previous = list(get_fallback_generalization_rules())
    try:
        set_fallback_generalizations(
            [{"name": "stale", "pattern": r"^x", "model_info": {"a": 1}}]
        )
        _finalize_model_cost_map(_make_models(2))
        assert match_capability_generalizations("x-1") is None
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
        "openrouter/anthropic/claude-opus-4.7",
        "azure_ai/claude-opus-4-7",
    ]:
        assert backup[adaptive]["supports_adaptive_thinking"] is True, adaptive

    for non_adaptive in [
        "claude-opus-4-20250514",
        "us.anthropic.claude-opus-4-20250514-v1:0",
        "claude-opus-4-5",
    ]:
        assert "supports_adaptive_thinking" not in backup[non_adaptive], non_adaptive


@pytest.mark.parametrize(
    "cost_map",
    [_load_root_cost_map(), GetModelCostMap.load_local_model_cost_map()],
    ids=["root", "bundled_backup"],
)
def test_azure_ai_claude_1m_context_entries(cost_map: dict):
    """Microsoft Foundry serves a 1M-token context window for Opus 4.6+ and Sonnet
    4.6+, so the ``azure_ai`` entries must not advertise the 200k cap that made
    context-aware clients compact prompts early (LIT-4406). Both the root map (used
    by default network loading) and the bundled fallback are checked so the two can
    never drift apart."""
    for model in [
        "azure_ai/claude-opus-4-6",
        "azure_ai/claude-opus-4-7",
        "azure_ai/claude-opus-4-8",
        "azure_ai/claude-opus-5",
        "azure_ai/claude-sonnet-5",
        "azure_ai/claude-sonnet-4-6",
    ]:
        assert cost_map[model]["max_input_tokens"] == 1000000, model

    for model in [
        "azure_ai/claude-opus-4-1",
        "azure_ai/claude-opus-4-5",
        "azure_ai/claude-sonnet-4-5",
        "azure_ai/claude-haiku-4-5",
    ]:
        assert cost_map[model]["max_input_tokens"] == 200000, model


def test_get_model_cost_map_stamps_loaded_at(monkeypatch):
    """The load time feeds each pod's reload-due decision; a load that does not stamp it
    would make manual reload requests race the proxy's startup"""
    from datetime import datetime, timezone

    from litellm.litellm_core_utils import get_model_cost_map as module

    monkeypatch.setattr(module._cost_map_source_info, "loaded_at", None)
    monkeypatch.setattr(
        module.GetModelCostMap,
        "fetch_remote_model_cost_map",
        staticmethod(lambda url, timeout=5: _load_root_cost_map()),
    )

    before = datetime.now(timezone.utc)
    module.get_model_cost_map(url="https://example.invalid/cost_map.json")
    loaded_at = module.get_model_cost_map_loaded_at()

    assert loaded_at is not None
    assert before <= loaded_at <= datetime.now(timezone.utc)

# ---------------------------------------------------------------------------
# refetch_model_cost_map: retry/backoff behavior for runtime reloads
# ---------------------------------------------------------------------------

import functools
import random

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


def _mock_client(outcomes):
    """httpx client over a MockTransport serving one outcome per request; an exception instance is raised."""
    calls = {"count": 0}

    def handler(request):
        idx = min(calls["count"], len(outcomes) - 1)
        calls["count"] += 1
        outcome = outcomes[idx]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), calls


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
    result = await refetch_model_cost_map(
        url=_URL, sleep=sleeper, rng=random.Random(0), client=client
    )
    assert isinstance(result, ModelCostMapReloaded)
    assert len(result.model_cost_map) > 100
    assert calls["count"] == 3
    assert sleeper.waits == [7.0, 7.0]


@pytest.mark.asyncio
async def test_refetch_gives_up_after_max_attempts_with_exponential_backoff():
    """All 429 without Retry-After: exponential backoff waits, then a failure value."""
    client, calls = _mock_client([httpx.Response(429)])
    sleeper = _SleepRecorder()
    result = await refetch_model_cost_map(
        url=_URL, sleep=sleeper, rng=random.Random(0), client=client
    )
    assert isinstance(result, ModelCostMapReloadUnavailable)
    assert "429" in result.reason
    assert "after 3 attempts" in result.reason
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
    result = await refetch_model_cost_map(
        url=_URL, sleep=sleeper, rng=random.Random(0), client=client
    )
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
    result = await refetch_model_cost_map(
        url=_URL, sleep=sleeper, rng=random.Random(0), client=client
    )
    assert isinstance(result, ModelCostMapReloaded)
    assert calls["count"] == 2
    assert len(sleeper.waits) == 1


@pytest.mark.asyncio
async def test_refetch_non_retryable_status_fails_immediately():
    """A 404 is permanent: one attempt, no sleeps, failure value."""
    client, calls = _mock_client([httpx.Response(404)])
    sleeper = _SleepRecorder()
    result = await refetch_model_cost_map(
        url=_URL, sleep=sleeper, rng=random.Random(0), client=client
    )
    assert isinstance(result, ModelCostMapReloadUnavailable)
    assert "404" in result.reason
    assert calls["count"] == 1
    assert sleeper.waits == []


@pytest.mark.asyncio
async def test_refetch_invalid_json_fails_immediately():
    client, calls = _mock_client([httpx.Response(200, content=b"not json")])
    sleeper = _SleepRecorder()
    result = await refetch_model_cost_map(
        url=_URL, sleep=sleeper, rng=random.Random(0), client=client
    )
    assert isinstance(result, ModelCostMapReloadUnavailable)
    assert "invalid JSON" in result.reason
    assert calls["count"] == 1
    assert sleeper.waits == []


@pytest.mark.asyncio
async def test_refetch_shrunk_map_fails_integrity_not_swapped_in():
    """A drastically shrunk upstream file is rejected instead of being adopted."""
    tiny = json.dumps(_make_models(60)).encode()
    client, _calls = _mock_client([httpx.Response(200, content=tiny)])
    result = await refetch_model_cost_map(
        url=_URL, sleep=_SleepRecorder(), rng=random.Random(0), client=client
    )
    assert isinstance(result, ModelCostMapReloadUnavailable)
    assert "integrity validation" in result.reason


@pytest.mark.asyncio
async def test_refetch_respects_local_env_override(monkeypatch):
    """LITELLM_LOCAL_MODEL_COST_MAP=True short-circuits to the bundled backup, zero HTTP."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

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
