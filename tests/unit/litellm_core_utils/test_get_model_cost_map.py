"""
Tests for model-cost-map loading: the model-count integrity check (which must
count actual model entries, not reserved meta keys) and the extraction of the
``fallback_generalizations`` block out of the raw map.
"""

import asyncio, importlib, importlib.resources as importlib_get_model, json, litellm
import os
import sys
import threading
from pathlib import Path
from typing import Final

import pytest
from pydantic import ValidationError

from litellm.litellm_core_utils import get_model_cost_map as get_model_cost_map_module
from litellm.litellm_core_utils.fallback_generalizations import (
    get_fallback_generalization_rules,
    match_capability_generalizations,
    match_routing_generalization,
    set_fallback_generalizations,
)
from litellm.litellm_core_utils.get_model_cost_map import (
    FALLBACK_GENERALIZATIONS_KEY,
    GetModelCostMap,
    _finalize_model_cost_map,
    count_model_entries,
    get_model_cost_map_provenance,
    git_blob_id,
)


def _load_root_cost_map() -> dict:
    path = os.path.join(os.path.dirname(__file__), "../../../model_prices_and_context_window.json")
    with open(path) as f:
        return json.load(f)


def _bundled_blob_id() -> str:
    path = os.path.join(os.path.dirname(__file__), "../../../litellm/model_prices_and_context_window_backup.json")
    with open(path, "rb") as f:
        return git_blob_id(f.read())


def test_git_blob_id_is_what_git_hash_object_prints():
    assert git_blob_id(b'{"gpt-5.4-mini": {"mode": "chat"}}\n') == "18b9a8381e13a3b38a2128f184f631f95829e987"


def _make_models(n: int) -> dict:
    return {f"model-{i}": {"litellm_provider": "openai", "mode": "chat"} for i in range(n)}


def test_count_model_entries_excludes_reserved_keys():
    m = _make_models(3)
    m["sample_spec"] = {"foo": "bar"}
    m[FALLBACK_GENERALIZATIONS_KEY] = {"rules": []}
    assert count_model_entries(m) == 3


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
        set_fallback_generalizations([{"name": "stale", "pattern": r"^x", "model_info": {"a": 1}}])
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
        "azure_ai/claude-opus-4-7",
    ]:
        assert backup[adaptive]["supports_adaptive_thinking"] is True, adaptive

    for non_adaptive in [
        "us.anthropic.claude-opus-4-20250514-v1:0",
        "claude-opus-4-5",
    ]:
        assert "supports_adaptive_thinking" not in backup[non_adaptive], non_adaptive


def test_get_model_cost_map_stamps_loaded_at(scripted_http):
    """The load time feeds each pod's reload-due decision; a load that does not stamp it
    would make manual reload requests race the proxy's startup"""
    from datetime import datetime, timezone

    from litellm.litellm_core_utils import get_model_cost_map as module

    server = scripted_http([httpx.Response(200, content=_real_map_bytes())])

    before = datetime.now(timezone.utc)
    module.get_model_cost_map(url=server.url)
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
from litellm.rust_bridge._native import install_model_cost_map_provenance, model_cost_map_provenance


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


@pytest.fixture(autouse=True)
def _restore_cost_map_provenance():
    """Loads install a new provenance, so each test puts the one it found back."""
    found = model_cost_map_provenance()
    yield
    if found is not None:
        install_model_cost_map_provenance(
            source=found.source,
            url=found.url,
            is_env_forced=found.is_env_forced,
            fallback_reason=found.fallback_reason,
            revision=found.revision,
            etag=found.etag,
        )


@pytest.mark.asyncio
async def test_refetch_retries_429_honoring_retry_after(scripted_http):
    """Two 429s with Retry-After then success: waits follow the header, not backoff."""
    server = scripted_http(
        [
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, content=_real_map_bytes()),
        ]
    )
    sleeper = _SleepRecorder()
    result = await refetch_model_cost_map(url=server.url, sleep=sleeper, rng=random.Random(0))
    assert isinstance(result, ModelCostMapReloaded)
    assert len(result.model_cost_map) > 100
    assert server.count == 3
    assert sleeper.waits == [7.0, 7.0]


@pytest.mark.asyncio
async def test_refetch_gives_up_after_max_attempts_with_exponential_backoff(scripted_http):
    """All 429 without Retry-After: exponential backoff waits, then a failure value."""
    server = scripted_http([httpx.Response(429)])
    sleeper = _SleepRecorder()
    result = await refetch_model_cost_map(url=server.url, sleep=sleeper, rng=random.Random(0))
    assert isinstance(result, ModelCostMapReloadUnavailable)
    assert result.reason == f"HTTP 429 from {server.url} (after 3 attempts)"
    assert server.count == 3
    assert sleeper.waits == [2.0 + _jitter(2)[0], 4.0 + _jitter(2)[1]]


def _jitter(count: int) -> list[float]:
    rng = random.Random(0)
    return [rng.uniform(0.0, 1.0) for _ in range(count)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("replies", "waits", "reloaded", "reason"),
    [
        pytest.param(
            [httpx.Response(429, headers={"Retry-After": "3"}), httpx.Response(503), None],
            [3.0, 4.0 + _jitter(1)[0]],
            True,
            None,
            id="jitter_is_drawn_only_for_a_backoff",
        ),
        pytest.param(
            [
                httpx.Response(429, headers={"Retry-After": "-1"}),
                httpx.Response(429, headers={"Retry-After": "nan"}),
                None,
            ],
            [2.0 + _jitter(2)[0], 4.0 + _jitter(2)[1]],
            True,
            None,
            id="negative_and_nan_retry_after_back_off",
        ),
        pytest.param(
            [httpx.Response(429, headers={"Retry-After": "inf"}), None],
            [30.0],
            True,
            None,
            id="infinite_retry_after_is_capped",
        ),
        pytest.param(
            [httpx.Response(503, headers={"Retry-After": " 2_0 "}), None],
            [20.0],
            True,
            None,
            id="retry_after_is_read_as_python_float_reads_it",
        ),
        pytest.param(
            [httpx.Response(503, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}), None],
            [2.0 + _jitter(1)[0]],
            True,
            None,
            id="an_http_date_retry_after_backs_off",
        ),
        pytest.param([httpx.Response(501)], [], False, "HTTP 501 from ", id="other_5xx_is_final"),
        pytest.param([httpx.Response(599)], [], False, "HTTP 599 from ", id="599_is_final"),
        pytest.param([httpx.Response(400)], [], False, "HTTP 400 from ", id="400_is_final"),
        pytest.param([httpx.Response(302, content=b"moved")], [], False, "invalid JSON", id="a_redirect_is_read"),
        pytest.param([httpx.Response(304)], [], False, "invalid JSON", id="an_unconditional_304_is_read"),
        pytest.param([httpx.Response(600, content=b"[]")], [], False, "expected a JSON object", id="600_is_read"),
    ],
)
async def test_refetch_retry_policy_matches_main(
    scripted_http, replies: list[httpx.Response | None], waits: list[float], reloaded: bool, reason: str | None
):
    """The same status cases as ``status_rule_matches_main`` in ``model-catalog-refresh/src/status.rs``"""
    server = scripted_http(
        [reply if reply is not None else httpx.Response(200, content=_real_map_bytes()) for reply in replies]
    )
    sleeper = _SleepRecorder()

    result = await refetch_model_cost_map(url=server.url, sleep=sleeper, rng=random.Random(0))

    assert sleeper.waits == waits
    assert isinstance(result, ModelCostMapReloaded) is reloaded
    if reason is not None:
        assert isinstance(result, ModelCostMapReloadUnavailable)
        assert result.reason.startswith(reason)


@pytest.mark.asyncio
async def test_refetch_caps_retry_after_wait(scripted_http):
    """A hostile/huge Retry-After is capped so reloads never sleep unbounded."""
    server = scripted_http(
        [
            httpx.Response(429, headers={"Retry-After": "9999"}),
            httpx.Response(200, content=_real_map_bytes()),
        ]
    )
    sleeper = _SleepRecorder()
    result = await refetch_model_cost_map(url=server.url, sleep=sleeper, rng=random.Random(0))
    assert isinstance(result, ModelCostMapReloaded)
    assert sleeper.waits == [30.0]


@pytest.mark.asyncio
async def test_refetch_retries_transport_errors(scripted_http):
    """Connection failures are transient: retried like 5xx, succeeding when the network heals."""
    server = scripted_http(
        [
            httpx.ConnectError("connection refused"),
            httpx.Response(200, content=_real_map_bytes()),
        ]
    )
    sleeper = _SleepRecorder()
    result = await refetch_model_cost_map(url=server.url, sleep=sleeper, rng=random.Random(0))
    assert isinstance(result, ModelCostMapReloaded)
    assert server.count == 2
    assert len(sleeper.waits) == 1


@pytest.mark.asyncio
async def test_refetch_without_an_injected_sleep_waits_in_rust(scripted_http):
    server = scripted_http(
        [httpx.Response(503, headers={"Retry-After": "0"}), httpx.Response(200, content=_real_map_bytes())]
    )
    result = await refetch_model_cost_map(url=server.url, rng=random.Random(0))
    assert isinstance(result, ModelCostMapReloaded)
    assert server.count == 2


@pytest.mark.asyncio
async def test_refetch_raises_what_the_injected_sleep_raises_and_stops_fetching(scripted_http):
    server = scripted_http([httpx.Response(503, headers={"Retry-After": "0"})])

    async def broken_sleep(seconds: float) -> None:
        raise RuntimeError(f"no sleeping for {seconds}")

    with pytest.raises(RuntimeError, match=r"no sleeping for 0\.0"):
        await refetch_model_cost_map(url=server.url, sleep=broken_sleep, rng=random.Random(0))
    assert server.count == 1


@pytest.mark.asyncio
async def test_refetch_warns_that_an_injected_client_is_ignored(scripted_http):
    server = scripted_http([httpx.Response(200, content=_real_map_bytes())])
    with pytest.warns(DeprecationWarning, match="client= is ignored"):
        result = await refetch_model_cost_map(url=server.url, client=object())
    assert isinstance(result, ModelCostMapReloaded)
    assert server.count == 1


@pytest.mark.asyncio
async def test_refetch_non_retryable_status_fails_immediately(scripted_http):
    """A 404 is permanent: one attempt, no sleeps, failure value."""
    server = scripted_http([httpx.Response(404)])
    sleeper = _SleepRecorder()
    result = await refetch_model_cost_map(url=server.url, sleep=sleeper, rng=random.Random(0))
    assert isinstance(result, ModelCostMapReloadUnavailable)
    assert result.reason == f"HTTP 404 from {server.url}"
    assert server.count == 1
    assert sleeper.waits == []


@pytest.mark.asyncio
async def test_refetch_invalid_json_fails_immediately(scripted_http):
    server = scripted_http([httpx.Response(200, content=b"not json")])
    sleeper = _SleepRecorder()
    result = await refetch_model_cost_map(url=server.url, sleep=sleeper, rng=random.Random(0))
    assert isinstance(result, ModelCostMapReloadUnavailable)
    assert "invalid JSON" in result.reason
    assert server.count == 1
    assert sleeper.waits == []


@pytest.mark.asyncio
async def test_refetch_shrunk_map_fails_integrity_not_swapped_in(scripted_http):
    """A drastically shrunk upstream file is rejected instead of being adopted."""
    tiny = json.dumps(_make_models(60)).encode()
    server = scripted_http([httpx.Response(200, content=tiny)])
    result = await refetch_model_cost_map(url=server.url, sleep=_SleepRecorder(), rng=random.Random(0))
    assert isinstance(result, ModelCostMapReloadUnavailable)
    assert "integrity validation" in result.reason


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["LITELLM_LOCAL_MODEL_COST_MAP", "LITELLM_OFFLINE_MODE"])
async def test_refetch_respects_local_env_override(monkeypatch, scripted_http, flag):
    """LITELLM_LOCAL_MODEL_COST_MAP=True or LITELLM_OFFLINE_MODE=True short-circuits to the bundled backup, zero HTTP."""
    monkeypatch.setenv(flag, "True")
    server = scripted_http([httpx.Response(200, content=_real_map_bytes())])

    result = await refetch_model_cost_map(url=server.url, sleep=_SleepRecorder(), rng=random.Random(0))

    assert isinstance(result, ModelCostMapReloaded)
    assert len(result.model_cost_map) > 100
    assert server.count == 0


@pytest.mark.asyncio
async def test_refetch_records_the_blob_id_of_the_bytes_served_and_the_fetch_etag(scripted_http):
    body = _real_map_bytes()
    server = scripted_http([httpx.Response(200, headers={"ETag": 'W/"abc123"'}, content=body)])

    result = await refetch_model_cost_map(url=server.url, sleep=_SleepRecorder(), rng=random.Random(0))

    assert isinstance(result, ModelCostMapReloaded)
    assert result.revision == git_blob_id(body)
    assert result.etag == 'W/"abc123"'
    assert get_model_cost_map_provenance() == {"source_revision": git_blob_id(body), "etag": 'W/"abc123"'}


@pytest.mark.asyncio
async def test_refetch_reads_a_file_url(tmp_path):
    body = _real_map_bytes()
    path = tmp_path / "model_prices.json"
    path.write_bytes(body)

    result = await refetch_model_cost_map(url=path.as_uri())

    assert isinstance(result, ModelCostMapReloaded)
    assert result.revision == git_blob_id(body)
    assert get_model_cost_map_source_info() == {
        **get_model_cost_map_source_info(),
        "source": "remote",
        "url": path.as_uri(),
        "etag": None,
    }


@pytest.mark.asyncio
async def test_refetch_of_a_missing_file_is_final(tmp_path):
    missing = (tmp_path / "missing.json").as_uri()
    result = await refetch_model_cost_map(url=missing, sleep=_SleepRecorder())
    assert isinstance(result, ModelCostMapReloadUnavailable)
    assert result.reason.startswith(f"could not fetch {missing}: could not read the catalog")


@pytest.mark.asyncio
async def test_loaded_catalog_snapshot_follows_the_fetched_map_and_ignores_later_registrations(
    monkeypatch, scripted_http
):
    import litellm

    edited = json.loads(_real_map_bytes())
    edited["gpt-5.4-mini"]["max_input_tokens"] = 777
    server = scripted_http([httpx.Response(200, content=json.dumps(edited).encode())])

    result = await refetch_model_cost_map(url=server.url, sleep=_SleepRecorder(), rng=random.Random(0))

    assert isinstance(result, ModelCostMapReloaded)
    monkeypatch.setattr(litellm, "model_cost", result.model_cost_map)
    litellm.register_model({"gpt-5.4-mini": {"max_input_tokens": 2048}}, persist_across_reloads=False)
    assert litellm.model_cost["gpt-5.4-mini"]["max_input_tokens"] == 2048
    assert GetModelCostMap.loaded_model_cost_map()["gpt-5.4-mini"]["max_input_tokens"] == 777


@pytest.mark.asyncio
async def test_loaded_catalog_fields_follow_each_fetched_map(scripted_http):
    added = json.loads(_real_map_bytes())
    added["gpt-5.4-mini"]["supports_litellm_qa_capability"] = True
    with_field = scripted_http([httpx.Response(200, content=json.dumps(added).encode())])
    without_field = scripted_http([httpx.Response(200, content=_real_map_bytes())])

    await refetch_model_cost_map(url=with_field.url, sleep=_SleepRecorder(), rng=random.Random(0))
    fields_after_adding: Final = GetModelCostMap.loaded_model_cost_map_fields()
    await refetch_model_cost_map(url=without_field.url, sleep=_SleepRecorder(), rng=random.Random(0))

    assert "supports_litellm_qa_capability" in fields_after_adding
    assert "supports_litellm_qa_capability" not in GetModelCostMap.loaded_model_cost_map_fields()
    assert "max_input_tokens" in GetModelCostMap.loaded_model_cost_map_fields()


@pytest.mark.asyncio
async def test_refetch_revision_follows_the_bytes_not_the_url(scripted_http):
    edited = json.loads(_real_map_bytes())
    edited["gpt-5.4-mini"]["input_cost_per_token"] = 0.5
    server = scripted_http(
        [httpx.Response(200, content=_real_map_bytes()), httpx.Response(200, content=json.dumps(edited).encode())]
    )

    first = await refetch_model_cost_map(url=server.url, sleep=_SleepRecorder(), rng=random.Random(0))
    second = await refetch_model_cost_map(url=server.url, sleep=_SleepRecorder(), rng=random.Random(0))

    assert isinstance(first, ModelCostMapReloaded) and isinstance(second, ModelCostMapReloaded)
    assert first.revision != second.revision
    assert get_model_cost_map_provenance()["source_revision"] == second.revision


@pytest.mark.asyncio
async def test_refetch_local_override_reports_the_bundled_blob_id_without_an_etag(monkeypatch, scripted_http):
    remote = scripted_http([httpx.Response(200, headers={"ETag": 'W/"remote"'}, content=_real_map_bytes())])
    await refetch_model_cost_map(url=remote.url, sleep=_SleepRecorder(), rng=random.Random(0))
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    result = await refetch_model_cost_map(url=remote.url, sleep=_SleepRecorder(), rng=random.Random(0))

    assert isinstance(result, ModelCostMapReloaded)
    assert result.revision == _bundled_blob_id()
    assert get_model_cost_map_provenance() == {"source_revision": _bundled_blob_id(), "etag": None}


@pytest.mark.asyncio
async def test_refetch_stamps_loaded_at_on_remote_and_local_reloads(monkeypatch, scripted_http):
    from litellm.litellm_core_utils import get_model_cost_map as module

    server = scripted_http([httpx.Response(200, content=_real_map_bytes())])
    before_remote = datetime.now(timezone.utc)
    await refetch_model_cost_map(url=server.url, sleep=_SleepRecorder(), rng=random.Random(0))
    remote_loaded_at = module.get_model_cost_map_loaded_at()
    assert remote_loaded_at is not None
    assert before_remote <= remote_loaded_at <= datetime.now(timezone.utc)

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    before_local = datetime.now(timezone.utc)
    await refetch_model_cost_map(url=server.url, sleep=_SleepRecorder(), rng=random.Random(0))
    local_loaded_at = module.get_model_cost_map_loaded_at()
    assert local_loaded_at is not None
    assert before_local <= local_loaded_at <= datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# get_model_cost_map: the boot-time load retries transient failures like a reload does
# ---------------------------------------------------------------------------

from litellm.litellm_core_utils.get_model_cost_map import (
    get_model_cost_map,
    get_model_cost_map_source_info,
    load_model_cost_map_at_import,
)
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.utils import _invalidate_model_cost_lowercase_map
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome
from unittest.mock import call, patch


def test_get_model_cost_map_fetches_s3_catalog(monkeypatch: pytest.MonkeyPatch, scripted_http) -> None:
    s3_content: Final = _make_models(GetModelCostMap.get_backup_model_count())
    body: Final = json.dumps(s3_content).encode()
    server = scripted_http([httpx.Response(200, headers={"ETag": '"fake-etag"'}, content=body)], path="")
    for name, value in {
        "AWS_ENDPOINT_URL": server.url,
        "AWS_ACCESS_KEY_ID": "test-access-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key",
        "AWS_REGION_NAME": "us-east-1",
    }.items():
        monkeypatch.setenv(name, value)

    cost_map: Final = get_model_cost_map("s3://my-bucket/prices/model_prices.json")

    assert cost_map == s3_content
    assert [request["path"].split("?")[0] for request in server.requests] == ["/my-bucket/prices/model_prices.json"]
    assert server.requests[0]["authorization"].startswith("AWS4-HMAC-SHA256 Credential=test-access-key/")
    source = get_model_cost_map_source_info()
    assert (source["source"], source["etag"], source["source_revision"]) == ("remote", '"fake-etag"', git_blob_id(body))


def test_an_s3_failure_is_final(monkeypatch: pytest.MonkeyPatch, scripted_http) -> None:
    server = scripted_http([httpx.Response(503)], path="")
    for name, value in {
        "AWS_ENDPOINT_URL": server.url,
        "AWS_ACCESS_KEY_ID": "test-access-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key",
        "AWS_REGION_NAME": "us-east-1",
    }.items():
        monkeypatch.setenv(name, value)

    get_model_cost_map("s3://my-bucket/prices/model_prices.json")

    assert server.count == 1
    assert _retry_threads() == []
    assert get_model_cost_map_source_info()["fallback_reason"].startswith(
        "Remote fetch failed: could not fetch s3://my-bucket/prices/model_prices.json: S3 GetObject failed"
    )


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        pytest.param("s3://my-bucket/", "expected s3://bucket/key", id="s3_without_a_key"),
        pytest.param("ftp://example.invalid/prices.json", 'unsupported catalog URL scheme "ftp"', id="ftp"),
        pytest.param("", 'unsupported catalog URL scheme ""', id="empty"),
    ],
)
def test_a_location_no_source_serves_falls_back_at_once(url: str, reason: str) -> None:
    get_model_cost_map(url)

    assert _retry_threads() == []
    assert (
        get_model_cost_map_source_info()["fallback_reason"] == f"Remote fetch failed: could not fetch {url}: {reason}"
    )


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


def _refresh_threads():
    return [thread for thread in threading.enumerate() if thread.name == "litellm-model-cost-map-refresh"]


def test_boot_load_success_does_not_start_background_retry(scripted_http):
    server = scripted_http([httpx.Response(200, content=_real_map_bytes())])
    sleeper = _SyncSleepRecorder()

    cost_map = get_model_cost_map(url=server.url, sleep=sleeper, rng=random.Random(0))
    assert server.count == 1
    assert sleeper.waits == []
    assert _retry_threads() == []
    assert cost_map.keys() >= _load_root_cost_map().keys() - {"sample_spec", FALLBACK_GENERALIZATIONS_KEY}
    assert get_model_cost_map_source_info()["source"] == "remote"


def _isolated_catalog_state(monkeypatch):
    import litellm
    from litellm import utils as litellm_utils

    monkeypatch.setattr(litellm, "model_cost", dict(litellm.model_cost))
    for name, provider_models in tuple(vars(litellm).items()):
        if name.endswith("_models") and isinstance(provider_models, set):
            monkeypatch.setattr(litellm, name, set(provider_models))
    monkeypatch.setattr(litellm, "models_by_provider", dict(litellm.models_by_provider))
    monkeypatch.setattr(
        litellm_utils,
        "_runtime_registered_model_cost",
        dict(litellm_utils._runtime_registered_model_cost),
    )


def test_boot_load_transient_failure_returns_local_then_background_retry_adopts_remote(monkeypatch, scripted_http):
    import litellm

    _isolated_catalog_state(monkeypatch)
    remote_map = _load_root_cost_map()
    remote_map["claude-remote-only-test"] = {"litellm_provider": "anthropic", "mode": "chat"}
    server = scripted_http(
        [
            httpx.ConnectError("connection refused"),
            httpx.Response(200, content=json.dumps(remote_map).encode()),
        ]
    )
    sleeper = _SyncSleepRecorder(block=True)
    litellm.register_model({"my-runtime-model": {"litellm_provider": "custom", "max_input_tokens": 4321}})

    cost_map = get_model_cost_map(url=server.url, max_attempts=3, sleep=sleeper, rng=random.Random(0))

    assert server.count == 1
    assert sleeper.waits == []
    assert sleeper.started.wait(timeout=10)
    threads = _retry_threads()
    try:
        assert len(threads) == 1
        assert "claude-remote-only-test" not in cost_map
        assert cost_map.keys() == _finalize_model_cost_map(GetModelCostMap.load_local_model_cost_map()).keys()
        source = get_model_cost_map_source_info()
        assert source["source"] == "local"
        assert source["fallback_reason"].startswith("Remote fetch failed:")
        sleeper.release.set()
        for thread in threads:
            thread.join(timeout=10)
        assert all(not thread.is_alive() for thread in threads)
        assert sleeper.waits == [2.0 + _jitter(1)[0]]
        assert server.count == 2
        assert "claude-remote-only-test" in litellm.model_cost
        assert "claude-remote-only-test" in litellm.anthropic_models
        assert "my-runtime-model" in litellm.model_cost
        source = get_model_cost_map_source_info()
        assert source["source"] == "remote"
        assert source["fallback_reason"] is None
    finally:
        sleeper.release.set()
        for thread in _retry_threads():
            thread.join(timeout=10)


def test_boot_load_does_not_retry_non_retryable_failure(scripted_http):
    server = scripted_http([httpx.Response(404)])
    sleeper = _SyncSleepRecorder()

    get_model_cost_map(url=server.url, sleep=sleeper, rng=random.Random(0))
    assert server.count == 1
    assert sleeper.waits == []
    assert _retry_threads() == []
    source = get_model_cost_map_source_info()
    assert source["source"] == "local"
    assert source["fallback_reason"] == f"Remote fetch failed: HTTP 404 from {server.url}"


def test_boot_load_with_one_attempt_does_not_retry_a_retryable_failure(scripted_http):
    server = scripted_http([httpx.Response(503)])

    get_model_cost_map(url=server.url, max_attempts=1)

    assert server.count == 1
    assert _retry_threads() == []
    assert get_model_cost_map_source_info()["fallback_reason"] == f"Remote fetch failed: HTTP 503 from {server.url}"


def test_boot_load_falls_back_when_the_url_does_not_parse():
    sleeper = _SyncSleepRecorder()

    cost_map = get_model_cost_map(url="http://[bad/model_prices.json", sleep=sleeper, rng=random.Random(0))
    assert sleeper.waits == []
    assert _retry_threads() == []
    assert cost_map.keys() >= _load_root_cost_map().keys() - {"sample_spec", FALLBACK_GENERALIZATIONS_KEY}
    source = get_model_cost_map_source_info()
    assert source["source"] == "local"
    assert source["fallback_reason"].startswith(
        "Remote fetch failed: could not fetch http://[bad/model_prices.json: invalid catalog URL"
    )


@pytest.mark.parametrize("flag", ["LITELLM_LOCAL_MODEL_COST_MAP", "LITELLM_OFFLINE_MODE"])
def test_boot_load_respects_local_env_override(monkeypatch, scripted_http, flag):
    """LITELLM_LOCAL_MODEL_COST_MAP=True or LITELLM_OFFLINE_MODE=True still short-circuits to the backup with zero HTTP."""
    monkeypatch.setenv(flag, "True")
    server = scripted_http([httpx.Response(200, content=_real_map_bytes())])

    cost_map = get_model_cost_map(url=server.url, sleep=_SyncSleepRecorder())
    assert len(cost_map) > 100
    assert server.count == 0
    assert get_model_cost_map_source_info()["is_env_forced"] is True


def test_boot_load_records_the_blob_id_of_the_bytes_served_and_the_fetch_etag(scripted_http):
    body = _real_map_bytes()
    server = scripted_http([httpx.Response(200, headers={"ETag": 'W/"boot"'}, content=body)])

    get_model_cost_map(url=server.url, sleep=_SyncSleepRecorder(), rng=random.Random(0))

    source = get_model_cost_map_source_info()
    assert source["source"] == "remote"
    assert source["etag"] == 'W/"boot"'
    assert source["source_revision"] == git_blob_id(body)
    assert source["loaded_at"] is not None


def test_boot_load_joins_a_repeated_etag_as_httpx_reads_it(scripted_http):
    server = scripted_http(
        [httpx.Response(200, headers=[("ETag", 'W/"a"'), ("ETag", 'W/"b"')], content=_real_map_bytes())]
    )

    get_model_cost_map(url=server.url)

    assert get_model_cost_map_source_info()["etag"] == 'W/"a", W/"b"'


def test_boot_load_fallback_to_the_backup_reports_its_blob_id_and_drops_the_remote_etag(scripted_http):
    remote = scripted_http([httpx.Response(200, headers={"ETag": 'W/"boot"'}, content=_real_map_bytes())])
    get_model_cost_map(url=remote.url, sleep=_SyncSleepRecorder(), rng=random.Random(0))
    failing = scripted_http([httpx.Response(404)])

    get_model_cost_map(url=failing.url, sleep=_SyncSleepRecorder(), rng=random.Random(0))

    source = get_model_cost_map_source_info()
    assert source["source"] == "local"
    assert source["etag"] is None
    assert source["source_revision"] == _bundled_blob_id()


def test_boot_load_that_fails_the_integrity_check_reports_the_backup_not_the_rejected_fetch(scripted_http):
    remote = scripted_http([httpx.Response(200, headers={"ETag": 'W/"boot"'}, content=_real_map_bytes())])
    get_model_cost_map(url=remote.url, sleep=_SyncSleepRecorder(), rng=random.Random(0))
    shrunk_body = b'{"gpt-5.4-mini": {"mode": "chat", "input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06}}'
    shrunk = scripted_http([httpx.Response(200, headers={"ETag": 'W/"shrunk"'}, content=shrunk_body)])

    get_model_cost_map(url=shrunk.url, sleep=_SyncSleepRecorder(), rng=random.Random(0))

    source = get_model_cost_map_source_info()
    assert source["source"] == "local"
    assert source["fallback_reason"] == "Remote data failed integrity validation"
    assert source["etag"] is None
    assert source["source_revision"] == _bundled_blob_id()
    assert source["source_revision"] != git_blob_id(shrunk_body)


@pytest.mark.parametrize(
    ("argv0", "request_count"),
    [
        ("/some/venv/bin/lite", 0),
        ("/some/venv/bin/lite.exe", 0),
        ("/some/venv/bin/python", 1),
    ],
)
def test_boot_load_skips_remote_fetch_for_cli_processes(
    monkeypatch: pytest.MonkeyPatch, scripted_http, argv0: str, request_count: int
) -> None:
    monkeypatch.setattr(sys, "argv", [argv0, "--version"])
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    server = scripted_http([httpx.Response(200, content=_real_map_bytes())])

    cost_map = get_model_cost_map(url=server.url)

    assert server.count == request_count
    assert cost_map
    source = get_model_cost_map_source_info()
    if request_count == 0:
        assert source["source"] == "local"
    else:
        assert source["source"] == "remote"


def _join_refresh_threads() -> None:
    for thread in _refresh_threads():
        thread.join(timeout=30)


def test_import_serves_the_bundled_map_then_adopts_the_remote_one_in_the_background(monkeypatch, scripted_http):
    import litellm

    _isolated_catalog_state(monkeypatch)
    remote_map = _load_root_cost_map()
    remote_map["claude-remote-only-test"] = {"litellm_provider": "anthropic", "mode": "chat"}
    body = json.dumps(remote_map).encode()
    server = scripted_http([httpx.Response(200, headers={"ETag": 'W/"swr"'}, content=body)])
    litellm.register_model({"my-runtime-model": {"litellm_provider": "custom", "max_input_tokens": 4321}})

    served = load_model_cost_map_at_import(server.url)
    at_import = get_model_cost_map_source_info()
    monkeypatch.setattr(litellm, "model_cost", served)
    _join_refresh_threads()

    assert "claude-remote-only-test" not in served
    assert (at_import["source"], at_import["url"], at_import["fallback_reason"]) == ("local", server.url, None)
    assert at_import["source_revision"] == _bundled_blob_id()
    assert server.count == 1
    assert "claude-remote-only-test" in litellm.model_cost
    assert "claude-remote-only-test" in litellm.anthropic_models
    assert "my-runtime-model" in litellm.model_cost
    after = get_model_cost_map_source_info()
    assert (after["source"], after["etag"], after["source_revision"]) == ("remote", 'W/"swr"', git_blob_id(body))


@pytest.mark.parametrize(
    ("reply", "fallback_reason"),
    [
        pytest.param(httpx.Response(404), "Remote fetch failed: HTTP 404 from {url}", id="final_status"),
        pytest.param(
            httpx.Response(200, content=json.dumps(_make_models(60)).encode()),
            "Remote data failed integrity validation",
            id="shrunk",
        ),
    ],
)
def test_import_keeps_the_bundled_map_when_the_background_refresh_fails(
    monkeypatch, scripted_http, reply, fallback_reason
):
    import litellm

    _isolated_catalog_state(monkeypatch)
    server = scripted_http([reply])

    served = load_model_cost_map_at_import(server.url)
    monkeypatch.setattr(litellm, "model_cost", served)
    _join_refresh_threads()

    assert litellm.model_cost is served
    source = get_model_cost_map_source_info()
    assert (source["source"], source["url"], source["source_revision"]) == ("local", server.url, _bundled_blob_id())
    assert source["fallback_reason"] == fallback_reason.format(url=server.url)


@pytest.mark.parametrize("flag", ["LITELLM_LOCAL_MODEL_COST_MAP", "LITELLM_OFFLINE_MODE"])
def test_import_with_the_local_map_forced_starts_no_refresh(monkeypatch, scripted_http, flag):
    monkeypatch.setenv(flag, "True")
    server = scripted_http([httpx.Response(200, content=_real_map_bytes())])

    load_model_cost_map_at_import(server.url)

    assert _refresh_threads() == []
    assert server.count == 0
    assert get_model_cost_map_source_info()["is_env_forced"] is True


def test_import_in_a_cli_process_starts_no_refresh(monkeypatch, scripted_http):
    monkeypatch.setattr(sys, "argv", ["/some/venv/bin/litellm-proxy", "--version"])
    server = scripted_http([httpx.Response(200, content=_real_map_bytes())])

    load_model_cost_map_at_import(server.url)

    assert _refresh_threads() == []
    assert server.count == 0


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


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestCheckIsValidDict:
    """Unit tests for _check_is_valid_dict."""

    def test_should_reject_non_dict(self):
        """Non-dict should fail."""
        assert GetModelCostMap._check_is_valid_dict("not a dict") is False

    def test_should_reject_empty_dict(self):
        """Empty dict should fail."""
        assert GetModelCostMap._check_is_valid_dict({}) is False

    def test_should_reject_list(self):
        """List should fail."""
        assert GetModelCostMap._check_is_valid_dict([1, 2, 3]) is False

    def test_should_reject_none(self):
        """None should fail."""
        assert GetModelCostMap._check_is_valid_dict(None) is False

    def test_should_accept_non_empty_dict(self):
        """Non-empty dict should pass."""
        assert GetModelCostMap._check_is_valid_dict({"model": {}}) is True


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestCheckModelCountNotReduced:
    """Unit tests for _check_model_count_not_reduced."""

    def test_should_reject_too_few_models(self):
        """Fetched map with fewer models than min_model_count should fail."""
        small_map = {f"model-{i}": {} for i in range(5)}
        assert (
            GetModelCostMap._check_model_count_not_reduced(
                fetched_map=small_map, backup_model_count=0, min_model_count=10
            )
            is False
        )

    def test_should_reject_significant_shrinkage(self):
        """Fetched map that shrunk >50% vs backup should fail."""
        fetched = {f"model-{i}": {} for i in range(40)}  # 40% of 100
        assert (
            GetModelCostMap._check_model_count_not_reduced(
                fetched_map=fetched, backup_model_count=100, min_model_count=10
            )
            is False
        )

    def test_should_accept_when_above_threshold(self):
        """Fetched map at 60% of backup (above 50% threshold) should pass."""
        fetched = {f"model-{i}": {} for i in range(60)}
        assert (
            GetModelCostMap._check_model_count_not_reduced(
                fetched_map=fetched, backup_model_count=100, min_model_count=10
            )
            is True
        )

    def test_should_accept_growth(self):
        """Fetched map larger than backup should pass."""
        fetched = {f"model-{i}": {} for i in range(120)}
        assert (
            GetModelCostMap._check_model_count_not_reduced(
                fetched_map=fetched, backup_model_count=100, min_model_count=10
            )
            is True
        )

    def test_should_accept_with_empty_backup(self):
        """When backup is empty, only min_model_count matters."""
        fetched = {f"model-{i}": {} for i in range(15)}
        assert (
            GetModelCostMap._check_model_count_not_reduced(
                fetched_map=fetched, backup_model_count=0, min_model_count=10
            )
            is True
        )


@pytest.mark.parametrize(
    ("model_cost", "expected"),
    [
        pytest.param({"a": {}, "sample_spec": {}, FALLBACK_GENERALIZATIONS_KEY: {}}, 1, id="sections"),
        pytest.param({1: {}, None: {}, "Sample_Spec": {}}, 3, id="non_string_and_case_differing_keys"),
        pytest.param({}, 0, id="empty"),
    ],
)
def test_count_model_entries_matches_main(model_cost: dict[object, object], expected: int):
    assert count_model_entries(model_cost) == expected


_BELOW_MINIMUM = (
    "LiteLLM: Fetched model cost map has only %d models (minimum=%d). "
    "This may indicate a corrupted upstream file. "
    "Falling back to local backup."
)
_SHRUNK = (
    "LiteLLM: Fetched model cost map shrank significantly "
    "(fetched=%d, backup=%d, threshold=%.0f%%). "
    "This may indicate a corrupted upstream file. "
    "Falling back to local backup."
)


@pytest.mark.parametrize(
    ("models", "backup", "minimum", "ratio", "warning"),
    [
        pytest.param(5, 10, 6, 0.5, (_BELOW_MINIMUM, 5, 6), id="below_minimum_wins_over_shrunk"),
        pytest.param(5, 11, 0, 0.5, (_SHRUNK, 5, 11, 50.0), id="shrunk"),
        pytest.param(5, 10, 0, 0.5, None, id="exactly_at_the_ratio"),
        pytest.param(1, -5, -1, 0.5, None, id="negative_minimum_and_backup_pass"),
        pytest.param(0, 0, 0, 2.0, None, id="empty_backup_skips_the_ratio"),
        pytest.param(10, 10, 0, 1.5, (_SHRUNK, 10, 10, 150.0), id="ratio_above_one"),
        pytest.param(0, 10, 0, float("nan"), None, id="nan_ratio_never_shrinks"),
    ],
)
def test_model_count_check_matches_main(
    models: int, backup: int, minimum: int, ratio: float, warning: tuple[object, ...] | None
):
    with patch.object(get_model_cost_map_module.verbose_logger, "warning") as warn:
        passed = GetModelCostMap._check_model_count_not_reduced(
            fetched_map={**_make_models(models), "sample_spec": {}},
            backup_model_count=backup,
            min_model_count=minimum,
            max_shrink_ratio=ratio,
        )

    assert passed is (warning is None)
    assert warn.call_args_list == ([] if warning is None else [call(*warning)])


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestValidateModelCostMap:
    """Unit tests for validate_model_cost_map (combines both checks)."""

    def test_should_reject_non_dict(self):
        """Non-dict should fail at check 1."""
        assert GetModelCostMap.validate_model_cost_map(fetched_map="not a dict", backup_model_count=0) is False

    def test_should_reject_empty_map(self):
        """Empty dict should fail at check 1."""
        assert GetModelCostMap.validate_model_cost_map(fetched_map={}, backup_model_count=0) is False

    def test_should_reject_significant_shrinkage(self):
        """Should fail at check 2 (shrinkage)."""
        fetched = {f"model-{i}": {} for i in range(40)}
        assert (
            GetModelCostMap.validate_model_cost_map(fetched_map=fetched, backup_model_count=100, min_model_count=10)
            is False
        )

    def test_should_accept_valid_map(self):
        """Should pass both checks."""
        fetched = {f"model-{i}": {} for i in range(120)}
        assert (
            GetModelCostMap.validate_model_cost_map(fetched_map=fetched, backup_model_count=100, min_model_count=10)
            is True
        )

    def test_should_accept_equal_size_map(self):
        """Equal size should pass both checks."""
        fetched = {f"model-{i}": {} for i in range(100)}
        assert (
            GetModelCostMap.validate_model_cost_map(fetched_map=fetched, backup_model_count=100, min_model_count=10)
            is True
        )


def _fetch_with_single_outcome(serve, outcome):
    server = serve([outcome])
    result = get_model_cost_map(url=server.url, max_attempts=1)
    return result, server.count, get_model_cost_map_source_info(), server.url


def _backup_keys():
    return _finalize_model_cost_map(GetModelCostMap.load_local_model_cost_map()).keys()


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestGetModelCostMapFallback:
    """Tests for get_model_cost_map fallback behavior with bad upstream."""

    def test_should_fallback_to_backup_on_invalid_json(self, scripted_http):
        """When upstream returns invalid JSON, should fall back to local backup."""
        result, fetches, source, url = _fetch_with_single_outcome(
            scripted_http, httpx.Response(200, content=b"not json")
        )

        assert fetches == 1
        assert result.keys() == _backup_keys()
        assert source["source"] == "local"
        assert source["fallback_reason"].startswith("Remote fetch failed")

    def test_should_fallback_to_backup_on_network_error(self, scripted_http):
        """When upstream is unreachable, should fall back to local backup."""
        result, fetches, source, url = _fetch_with_single_outcome(
            scripted_http, httpx.ConnectError("Connection refused")
        )

        assert fetches == 1
        assert result.keys() == _backup_keys()
        assert source["source"] == "local"
        assert source["fallback_reason"].startswith("Remote fetch failed")

    def test_should_fallback_when_fetched_map_is_empty(self, scripted_http):
        """When upstream returns valid JSON but empty dict, should fall back."""
        result, fetches, source, url = _fetch_with_single_outcome(scripted_http, httpx.Response(200, content=b"{}"))

        assert fetches == 1
        assert result.keys() == _backup_keys()
        assert source["source"] == "local"

    def test_should_fallback_when_fetched_json_is_not_object(self, scripted_http):
        result, fetches, source, url = _fetch_with_single_outcome(scripted_http, httpx.Response(200, content=b"[]"))

        assert fetches == 1
        assert result.keys() == _backup_keys()
        assert source["source"] == "local"
        assert source["fallback_reason"] == f"Remote fetch failed: expected a JSON object from {url}, got list"

    def test_should_fallback_when_fetched_map_shrinks_dramatically(self, scripted_http):
        """When upstream returns far fewer models than backup, should fall back."""
        tiny_map = {f"model-{i}": {"litellm_provider": "test"} for i in range(11)}
        result, fetches, source, url = _fetch_with_single_outcome(
            scripted_http, httpx.Response(200, content=json.dumps(tiny_map).encode())
        )

        assert fetches == 1
        assert result.keys() == _backup_keys()
        assert source == {**source, "source": "local", "fallback_reason": "Remote data failed integrity validation"}

    def test_should_use_local_map_when_env_var_set(self, scripted_http):
        """LITELLM_LOCAL_MODEL_COST_MAP=True should skip remote fetch entirely."""
        server = scripted_http([httpx.Response(200, content=_real_map_bytes())])
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}):
            result = get_model_cost_map(server.url)

        assert server.count == 0
        assert result.keys() == _backup_keys()


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


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestBadHostedModelCostMap:
    """
    Simulates the hosted model cost map being bad (invalid JSON / corrupted).

    When the hosted map is bad, get_model_cost_map() falls back to the local
    backup. These tests verify that after fallback:
    - get_model_info() still works for models in the backup
    - litellm.completion() still works
    """

    def test_should_model_info_pass_after_bad_hosted_map(self, scripted_http):
        """
        If the hosted map is bad, get_model_cost_map falls back to the local
        backup. get_model_info should still work for models in the backup.
        """
        fallback_map = get_model_cost_map(scripted_http([httpx.Response(200, content=b"bad json")]).url)

        original = litellm.model_cost
        litellm.model_cost = fallback_map
        try:
            # gpt-4o is in every backup — should work fine
            info = litellm.get_model_info("gpt-4o")
            assert info is not None
            assert info["input_cost_per_token"] > 0
        finally:
            litellm.model_cost = original

    def test_should_completion_pass_after_bad_hosted_map(self, scripted_http):
        """
        If the hosted map is bad, litellm.completion() should still work.

        Uses litellm's built-in mock_response param so the real completion
        path is exercised (routing, cost calculator, logging) without
        needing API credentials.
        """
        fallback_map = get_model_cost_map(scripted_http([httpx.Response(200, content=b"bad json")]).url)

        original = litellm.model_cost
        litellm.model_cost = fallback_map
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


@pytest.mark.usefixtures(
    "_vcr_outcome_gate_local_testing",
    "isolate_litellm_state",
    "setup_and_teardown_local_testing",
)
def test_get_backup_model_cost_map():
    with importlib_get_model.open_text("litellm", "model_prices_and_context_window_backup.json") as f:
        print("inside backup")
        content = json.load(f)
        print("content", content)


@pytest.mark.parametrize("backup", [b'["only-in-the-backup"]', b'"only-in-the-backup"', b"null"])
def test_a_bundled_backup_that_is_not_a_json_object_is_rejected_without_quoting_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backup: bytes
):
    (tmp_path / "model_prices_and_context_window_backup.json").write_bytes(backup)
    monkeypatch.setattr(get_model_cost_map_module, "files", lambda _package: tmp_path)

    with pytest.raises(ValidationError) as rejected:
        GetModelCostMap.load_local_model_cost_map_with_revision()

    assert "only-in-the-backup" not in str(rejected.value)


@pytest.mark.asyncio
async def test_a_failed_import_refresh_leaves_a_map_installed_meanwhile_alone(scripted_http):
    from litellm.litellm_core_utils.get_model_cost_map import ModelCostMapRefresh, _refresh_in_background

    failing = scripted_http([httpx.Response(404)])
    load_model_cost_map_at_import(failing.url)
    _join_refresh_threads()
    reloaded = scripted_http([httpx.Response(200, headers={"ETag": 'W/"reload"'}, content=_real_map_bytes())])
    await refetch_model_cost_map(url=reloaded.url)

    _refresh_in_background(
        ModelCostMapRefresh(failing.url, timeout=5, max_attempts=1, rng=random.Random(), on_retry=print), failing.url
    )

    source = get_model_cost_map_source_info()
    assert (source["source"], source["url"], source["etag"], source["fallback_reason"]) == (
        "remote",
        reloaded.url,
        'W/"reload"',
        None,
    )


def test_a_refresh_landing_after_interpreter_exit_began_is_not_adopted(monkeypatch, scripted_http):
    import litellm
    from litellm.litellm_core_utils.get_model_cost_map import (
        ModelCostMapRefresh,
        _BackgroundAdoption,
        _refresh_in_background,
        _stop_background_adoption,
    )

    _isolated_catalog_state(monkeypatch)
    remote_map = _load_root_cost_map()
    remote_map["claude-remote-only-test"] = {"litellm_provider": "anthropic", "mode": "chat"}
    server = scripted_http([httpx.Response(200, content=json.dumps(remote_map).encode())])
    refresh = ModelCostMapRefresh(server.url, timeout=5, max_attempts=1, rng=random.Random(), on_retry=print)

    _stop_background_adoption()
    try:
        _refresh_in_background(refresh, server.url)
    finally:
        _BackgroundAdoption.exiting.clear()

    assert server.count == 1
    assert "claude-remote-only-test" not in litellm.model_cost
