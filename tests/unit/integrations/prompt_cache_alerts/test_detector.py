import asyncio
import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Final

import httpx
import pytest
import respx
from pydantic import JsonValue
from redis.exceptions import ConnectionError as RedisConnectionError

import litellm
from litellm.integrations.prompt_cache_alerts.detector import (
    PrefixEvent,
    PromptCacheAlertLogger,
    ScopeBreak,
    StreakEvent,
    StreakUpdate,
    Triggered,
    observe,
    prompt_cache_alert_logger,
)
from litellm.integrations.prompt_cache_alerts.evidence import fingerprint_key
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.models.alert import Alert, AlertDetection, PromptCacheEvidence
from litellm.responses.streaming_iterator import drain_logging_tasks

_KEY: Final = fingerprint_key("detector-test-salt")
_START: Final = 1_700_000_000.0
_CLOCK_MS: Final = int(_START * 1000) + 60_000
_API_BASE: Final = "https://api.anthropic.com/v1/messages"
_GUARD_SECONDS: Final = 5
_MARKED: Final[dict[str, JsonValue]] = {
    "model": "claude-sonnet-5-5",
    "system": [{"type": "text", "text": "shared policy", "cache_control": {"type": "ephemeral"}}],
    "messages": [{"role": "user", "content": "question"}],
}


def _usage(read: int | None, write: int | None) -> str:
    counters: Final = {
        key: value
        for key, value in (("cache_read_input_tokens", read), ("cache_creation_input_tokens", write))
        if value is not None
    }
    return json.dumps({"stop_reason": "end_turn", "usage": {"input_tokens": 9, **counters}})


def _kwargs(
    call_id: str = "call-1",
    at: float = _START,
    raw: object = None,
    body: object = _MARKED,
    provider: str = "anthropic",
    call_type: str = "acompletion",
    stream: bool = False,
    api_key: str = "sk-ant-secret",
    api_base: str = _API_BASE,
    model: str = "claude-sonnet-5-5",
    cache_hit: bool | None = None,
) -> dict[str, object]:
    return {
        "standard_logging_object": {
            "id": f"row-{call_id}",
            "litellm_call_id": call_id,
            "call_type": call_type,
            "stream": stream,
            "model": model,
            "custom_llm_provider": provider,
            "api_base": api_base,
            "model_id": "claude-east",
            "startTime": at,
            "endTime": at + 0.5,
        },
        "additional_args": {"complete_input_dict": body},
        "original_response": _usage(0, 0) if raw is None else raw,
        "api_key": api_key,
        "stream": stream,
        "cache_hit": cache_hit,
    }


def _update(kwargs: Mapping[str, object], succeeded: bool = True) -> StreakUpdate | None:
    observed: Final = observe(kwargs, _KEY, succeeded)
    return observed.update if observed is not None else None


def _event(update: StreakUpdate | None) -> tuple[StreakEvent, int] | str | None:
    match update:
        case PrefixEvent(event=event, write_tokens=tokens):
            return event, tokens
        case ScopeBreak():
            return "scope-break"
        case None:
            return None


@pytest.mark.parametrize(
    ("kwargs", "succeeded", "expected"),
    [
        (_kwargs(raw=_usage(0, 2048)), True, (StreakEvent.PROOF, 2048)),
        (_kwargs(raw=_usage(512, 0)), True, (StreakEvent.HIT, 0)),
        (_kwargs(raw=_usage(0, 0)), True, (StreakEvent.MISS, 0)),
        (_kwargs(raw=_usage(None, None)), True, (StreakEvent.UNKNOWN, 0)),
        (_kwargs(raw=_usage(0, 0), stream=True), True, (StreakEvent.UNKNOWN, 0)),
        (_kwargs(raw=_usage(0, 0)), False, (StreakEvent.UNKNOWN, 0)),
        (_kwargs(body={**_MARKED, "cache_control": {"type": "ephemeral"}}), True, "scope-break"),
        (_kwargs(body={**_MARKED, "system": "plain"}), True, None),
        (_kwargs(provider="bedrock"), True, None),
        (_kwargs(call_type="aresponses"), True, None),
        (_kwargs(cache_hit=True), True, None),
    ],
    ids=[
        "proving-write",
        "read",
        "miss",
        "missing-usage",
        "stream",
        "failure",
        "unsupported-breakpoint",
        "not-caching",
        "other-provider",
        "other-surface",
        "response-cache-hit",
    ],
)
def test_each_logged_attempt_becomes_its_streak_event(
    kwargs: Mapping[str, object], succeeded: bool, expected: tuple[StreakEvent, int] | str | None
) -> None:
    assert _event(_update(kwargs, succeeded)) == expected


@pytest.mark.parametrize(
    "changed",
    [{"api_base": "https://proxy.example/v1/messages"}, {"model": "claude-opus-5-5"}, {"api_key": "sk-ant-other"}],
)
def test_a_deployment_configuration_or_account_change_is_a_new_scope(changed: Mapping[str, str]) -> None:
    original: Final = _update(_kwargs())
    updated: Final = _update(_kwargs(**changed))
    assert original is not None and updated is not None
    assert original.scope.deployment_id == updated.scope.deployment_id
    assert original.scope.config_fingerprint != updated.scope.config_fingerprint
    assert "sk-ant-secret" not in repr(original)


class _ScriptedStore:
    def __init__(self, triggers: frozenset[int] = frozenset(), failing: frozenset[int] = frozenset()) -> None:
        self.triggers: Final = triggers
        self.failing: Final = failing
        self.calls = 0
        self.applied: tuple[StreakUpdate, ...] = ()
        self.held = asyncio.Event()
        self.release = asyncio.Event()
        self.hold_at: int | None = None

    async def apply(self, update: StreakUpdate, now_ms: int) -> Triggered | None:
        position: Final = self.calls
        self.calls += 1
        if position in self.failing:
            raise RedisConnectionError("redis is down")
        if position == self.hold_at:
            self.held.set()
            await self.release.wait()
        self.applied = (*self.applied, update)
        return Triggered(3, 2048, int((_START + 10) * 1000)) if position in self.triggers else None


class _Alerts:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure: Final = failure
        self.recorded: tuple[AlertDetection, ...] = ()

    async def record(self, detection: AlertDetection) -> Alert:
        if self.failure is not None:
            raise self.failure
        self.recorded = (*self.recorded, detection)
        return Alert.model_validate(
            {
                **detection.model_dump(),
                "id": "7",
                "status": "open",
                "created_at": detection.window_start,
                "updated_at": detection.window_start,
            }
        )


def _logger(store: _ScriptedStore | None, alerts: _Alerts) -> PromptCacheAlertLogger:
    return PromptCacheAlertLogger(_KEY, lambda: store, alerts, clock=lambda: _CLOCK_MS / 1000)


async def _log(logger: PromptCacheAlertLogger, kwargs: Mapping[str, object], succeeded: bool = True) -> None:
    if succeeded:
        await logger.async_log_success_event(kwargs, None, None, None)
    else:
        await logger.async_log_failure_event(kwargs, None, None, None)


async def test_a_triggered_streak_saves_one_deployment_incident_with_its_evidence() -> None:
    alerts: Final = _Alerts()
    logger: Final = _logger(_ScriptedStore(triggers=frozenset({0})), alerts)
    await _log(logger, _kwargs(call_id="c", at=_START + 12))
    (saved,) = alerts.recorded
    assert (saved.type, saved.resource_type, saved.resource_id) == ("prompt_cache", "deployment", "claude-east")
    assert saved.window_start == datetime.fromtimestamp(_START + 10, tz=timezone.utc)
    assert saved.window_end == datetime.fromtimestamp(_START + 12.5, tz=timezone.utc) + timedelta(milliseconds=1)
    assert isinstance(saved.evidence, PromptCacheEvidence)
    assert (saved.evidence.consecutive_misses, saved.evidence.proven_write_tokens) == (3, 2048)
    assert (saved.evidence.provider_model, saved.evidence.ttl_seconds) == ("claude-sonnet-5-5", 300)


async def test_evidence_lost_to_redis_breaks_its_scope_before_later_evidence_counts() -> None:
    store: Final = _ScriptedStore(failing=frozenset({1}))
    logger: Final = _logger(store, _Alerts())
    await _log(logger, _kwargs(call_id="a", at=_START + 10))
    await _log(logger, _kwargs(call_id="hit", at=_START + 11, raw=_usage(512, 0)))
    await _log(logger, _kwargs(call_id="c", at=_START + 12))
    assert [_event(update) for update in store.applied] == [
        (StreakEvent.MISS, 0),
        "scope-break",
        (StreakEvent.MISS, 0),
    ]
    (_, lost_break, _) = store.applied
    assert isinstance(lost_break, ScopeBreak) and lost_break.ended_ms == int((_START + 11.5) * 1000)


async def test_evidence_lost_while_an_older_break_is_in_flight_is_broken_before_later_evidence() -> None:
    store: Final = _ScriptedStore(failing=frozenset({0, 2}))
    store.hold_at = 1
    logger: Final = _logger(store, _Alerts())
    await _log(logger, _kwargs(call_id="lost-1", at=_START + 11, raw=_usage(512, 0)))
    flushing: Final = asyncio.create_task(_log(logger, _kwargs(call_id="next", at=_START + 20)))
    try:
        await asyncio.wait_for(store.held.wait(), timeout=_GUARD_SECONDS)
        await _log(logger, _kwargs(call_id="lost-2", at=_START + 15, raw=_usage(512, 0)))
        store.release.set()
        await asyncio.wait_for(flushing, timeout=_GUARD_SECONDS)
    finally:
        store.release.set()
        flushing.cancel()
        await asyncio.gather(flushing, return_exceptions=True)
    breaks: Final = [update.ended_ms for update in store.applied if isinstance(update, ScopeBreak)]
    assert breaks == [int((_START + 11.5) * 1000), int((_START + 15.5) * 1000)]
    assert _event(store.applied[-1]) == (StreakEvent.MISS, 0)


@pytest.mark.parametrize("store", [None, _ScriptedStore(failing=frozenset({0}))], ids=["no-redis", "redis-down"])
async def test_missing_or_failing_redis_saves_nothing_and_never_raises(store: _ScriptedStore | None) -> None:
    alerts: Final = _Alerts()
    await _log(_logger(store, alerts), _kwargs())
    assert alerts.recorded == ()


async def test_a_database_failure_while_saving_is_logged_and_never_raises() -> None:
    logger: Final = _logger(_ScriptedStore(triggers=frozenset({0})), _Alerts(failure=OSError("database unreachable")))
    await _log(logger, _kwargs())


@pytest.mark.parametrize(
    ("settings", "salt", "enabled"),
    [
        ({}, "salt", False),
        ({"prompt_cache_alerts": "true"}, "salt", False),
        ({"prompt_cache_alerts": True}, None, False),
        ({"prompt_cache_alerts": True}, "salt", True),
    ],
)
def test_alerts_are_off_unless_explicitly_enabled_with_a_shared_salt(
    settings: Mapping[str, object], salt: str | None, enabled: bool
) -> None:
    logger: Final = prompt_cache_alert_logger(settings, _Alerts(), lambda: None, salt)
    assert isinstance(logger, PromptCacheAlertLogger) is enabled


def test_alerts_are_off_without_a_database() -> None:
    assert prompt_cache_alert_logger({"prompt_cache_alerts": True}, None, lambda: None, "salt") is None


class _RecordingStore:
    def __init__(self) -> None:
        self.applied: tuple[StreakUpdate, ...] = ()
        self.done = asyncio.Event()

    async def apply(self, update: StreakUpdate, now_ms: int) -> Triggered | None:
        self.applied = (*self.applied, update)
        self.done.set()
        return None


def _anthropic_reply(usage: Mapping[str, int]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 9, "output_tokens": 2, **usage},
        },
    )


async def _completion(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, usage: Mapping[str, int], observed: bool
) -> tuple[litellm.ModelResponse, tuple[StreakUpdate, ...]]:
    store: Final = _RecordingStore()
    logger: Final = PromptCacheAlertLogger(_KEY, lambda: store, _Alerts())
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "callbacks", [logger] if observed else [])
    respx_mock.post(_API_BASE).mock(return_value=_anthropic_reply(usage))
    response: Final = await litellm.acompletion(
        model="anthropic/claude-sonnet-5-5",
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": "shared policy", "cache_control": {"type": "ephemeral"}}],
            },
            {"role": "user", "content": "question"},
        ],
        api_key="sk-ant-test",
        metadata={"model_info": {"id": "claude-east"}},
    )
    assert isinstance(response, litellm.ModelResponse)
    if observed:
        await drain_logging_tasks()
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=_GUARD_SECONDS)
        await asyncio.wait_for(store.done.wait(), timeout=_GUARD_SECONDS)
    return response, store.applied


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}, StreakEvent.MISS),
        ({}, StreakEvent.UNKNOWN),
    ],
    ids=["reported-zeros", "counters-absent"],
)
async def test_a_real_completion_reads_the_raw_counters_and_leaves_the_response_unchanged(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, usage: Mapping[str, int], expected: StreakEvent
) -> None:
    """Normalized usage reports 0 cache tokens either way, so only the raw body can tell them apart."""
    unobserved, _ = await _completion(monkeypatch, respx_mock, usage, observed=False)
    observed, updates = await _completion(monkeypatch, respx_mock, usage, observed=True)
    assert [_event(update) for update in updates] == [(expected, 0)]
    assert observed.choices[0].model_dump() == unobserved.choices[0].model_dump()
    assert observed.usage == unobserved.usage
