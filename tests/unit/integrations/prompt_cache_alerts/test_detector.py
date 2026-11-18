import asyncio
import json
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from datetime import datetime, timedelta, timezone
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest
import respx
from pydantic import JsonValue
from redis.exceptions import ConnectionError as RedisConnectionError

import litellm
from litellm.integrations.prompt_cache_alerts.detector import (
    Assessed,
    DeploymentReset,
    PromptCacheAlertLogger,
    Update,
    observe,
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


def _usage(read: int | None, write: int | None) -> dict[str, int]:
    return {
        key: value
        for key, value in (("cache_read_input_tokens", read), ("cache_creation_input_tokens", write))
        if value is not None
    }


def _raw(read: int | None, write: int | None) -> str:
    return json.dumps({"stop_reason": "end_turn", "usage": {"input_tokens": 9, **_usage(read, write)}})


def _kwargs(
    call_id: str = "call-1",
    at: float = _START,
    raw: str | None = None,
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
        "original_response": _raw(0, 0) if raw is None else raw,
        "api_key": api_key,
        "stream": stream,
        "cache_hit": cache_hit,
    }


def _kind(update: Update | None) -> str | None:
    match update:
        case Assessed(event=event, write_tokens=tokens):
            return f"{event}:{tokens}"
        case DeploymentReset():
            return "reset"
        case None:
            return None


@pytest.mark.parametrize(
    ("kwargs", "succeeded", "expected"),
    [
        (_kwargs(raw=_raw(0, 2048)), True, "proof:2048"),
        (_kwargs(raw=_raw(512, 0)), True, "hit:0"),
        (_kwargs(raw=_raw(0, 0)), True, "miss:0"),
        (_kwargs(raw=_raw(None, None)), True, "reset"),
        (_kwargs(stream=True), True, "reset"),
        (_kwargs(), False, "reset"),
        (_kwargs(body={**_MARKED, "cache_control": {"type": "ephemeral"}}), True, "reset"),
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
def test_each_logged_attempt_becomes_its_streak_update(
    kwargs: Mapping[str, object], succeeded: bool, expected: str | None
) -> None:
    assert _kind(observe(kwargs, _KEY, succeeded)) == expected


class _Script:
    """Stands in for the Lua script: records each call, answers by position and can fail or hold one."""

    def __init__(
        self,
        replies: Mapping[int, Sequence[str | int]] | None = None,
        failing: frozenset[int] = frozenset(),
        hold_at: int | None = None,
    ) -> None:
        self.replies: Final = replies or {}
        self.failing: Final = failing
        self.hold_at: Final = hold_at
        self.calls: tuple[tuple[tuple[str, ...], tuple[str | int, ...]], ...] = ()
        self.attempts = 0
        self.held = asyncio.Event()
        self.release = asyncio.Event()
        self.changed = asyncio.Event()

    async def __call__(self, keys: Sequence[str], args: Sequence[str | int]) -> object:
        position: Final = self.attempts
        self.attempts += 1
        if position in self.failing:
            raise RedisConnectionError("redis is down")
        if position == self.hold_at:
            self.held.set()
            await self.release.wait()
        self.calls = (*self.calls, (tuple(keys), tuple(args)))
        self.changed.set()
        return list(self.replies.get(position, []))

    def events(self) -> list[str | int]:
        return [args[0] for _, args in self.calls]

    async def until(self, count: int) -> None:
        while len(self.calls) < count:
            self.changed.clear()
            await asyncio.wait_for(self.changed.wait(), timeout=_GUARD_SECONDS)


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


def _logger(script: _Script | None, alerts: _Alerts, clock_ms: int = _CLOCK_MS) -> PromptCacheAlertLogger:
    return PromptCacheAlertLogger(_KEY, lambda: script, alerts, clock=lambda: clock_ms / 1000)


async def _log(logger: PromptCacheAlertLogger, kwargs: Mapping[str, object]) -> None:
    await logger.async_log_success_event(kwargs, None, None, None)


_TRIGGER: Final = (3, "2048", str(int((_START + 10) * 1000)))


async def test_a_triggered_streak_saves_one_incident_then_quiets_its_configuration() -> None:
    alerts: Final = _Alerts()
    script: Final = _Script(replies={0: _TRIGGER})
    await _log(_logger(script, alerts), _kwargs(call_id="c", at=_START + 12))
    (saved,) = alerts.recorded
    assert (saved.type, saved.resource_type, saved.resource_id) == ("prompt_cache", "deployment", "claude-east")
    assert saved.window_start == datetime.fromtimestamp(_START + 10, tz=timezone.utc)
    assert saved.window_end == datetime.fromtimestamp(_START + 12.5, tz=timezone.utc) + timedelta(milliseconds=1)
    assert isinstance(saved.evidence, PromptCacheEvidence)
    assert (saved.evidence.consecutive_misses, saved.evidence.proven_write_tokens) == (3, 2048)
    assert script.events() == ["miss", "quiet"]


async def test_an_unsaved_incident_leaves_its_configuration_free_to_trigger_again() -> None:
    script: Final = _Script(replies={0: _TRIGGER})
    await _log(_logger(script, _Alerts(failure=OSError("database unreachable"))), _kwargs())
    assert script.events() == ["miss"]


async def test_evidence_lost_to_redis_resets_its_deployment_before_later_evidence_counts() -> None:
    script: Final = _Script(failing=frozenset({1}))
    logger: Final = _logger(script, _Alerts())
    await _log(logger, _kwargs(call_id="a", at=_START + 10))
    await _log(logger, _kwargs(call_id="hit", at=_START + 11, raw=_raw(512, 0)))
    await _log(logger, _kwargs(call_id="c", at=_START + 12))
    assert script.events() == ["miss", "reset", "miss"]
    assert script.calls[1][1][2] == int((_START + 11.5) * 1000)


class _Clock:
    def __init__(self, now_ms: int) -> None:
        self.now_ms = now_ms

    def __call__(self) -> float:
        return self.now_ms / 1000


async def test_a_lost_reset_older_than_any_proof_is_dropped() -> None:
    script: Final = _Script(failing=frozenset({0}))
    clock: Final = _Clock(_CLOCK_MS)
    logger: Final = PromptCacheAlertLogger(_KEY, lambda: script, _Alerts(), clock=clock)
    await _log(logger, _kwargs(call_id="hit", at=_START, raw=_raw(512, 0)))
    clock.now_ms = _CLOCK_MS + 600_000
    await _log(logger, _kwargs(call_id="c", at=_START + 600))
    assert script.events() == ["miss"]


class _Reloadable:
    """The proxy's shared Redis getter, which a config reload can briefly leave empty."""

    def __init__(self, script: _Script) -> None:
        self.current: _Script | None = script

    def __call__(self) -> _Script | None:
        return self.current


async def test_evidence_seen_while_redis_is_reloading_resets_its_deployment_once_redis_is_back() -> None:
    """The read during the gap ended the streak, so the miss after it must not extend the misses before it."""
    script: Final = _Script()
    getter: Final = _Reloadable(script)
    logger: Final = PromptCacheAlertLogger(_KEY, getter, _Alerts(), clock=lambda: _CLOCK_MS / 1000)
    await _log(logger, _kwargs(call_id="a", at=_START + 10))
    getter.current = None
    await _log(logger, _kwargs(call_id="hit", at=_START + 11, raw=_raw(512, 0)))
    getter.current = script
    await _log(logger, _kwargs(call_id="c", at=_START + 12))
    assert script.events() == ["miss", "reset", "miss"]
    assert script.calls[1][1][2] == int((_START + 11.5) * 1000)


async def test_evidence_lost_while_an_older_reset_is_in_flight_is_reset_before_later_evidence() -> None:
    script: Final = _Script(failing=frozenset({0, 2}), hold_at=1)
    logger: Final = _logger(script, _Alerts())
    await _log(logger, _kwargs(call_id="lost-1", at=_START + 11, raw=_raw(512, 0)))
    flushing: Final = asyncio.create_task(_log(logger, _kwargs(call_id="next", at=_START + 20)))
    try:
        await asyncio.wait_for(script.held.wait(), timeout=_GUARD_SECONDS)
        await _log(logger, _kwargs(call_id="lost-2", at=_START + 15, raw=_raw(512, 0)))
        script.release.set()
        await asyncio.wait_for(flushing, timeout=_GUARD_SECONDS)
    finally:
        script.release.set()
        flushing.cancel()
        await asyncio.gather(flushing, return_exceptions=True)
    assert script.events() == ["reset", "reset", "miss"]
    assert [args[2] for _, args in script.calls[:2]] == [int((_START + 11.5) * 1000), int((_START + 15.5) * 1000)]


@pytest.mark.parametrize("script", [None, _Script(failing=frozenset({0}))], ids=["no-redis", "redis-down"])
async def test_missing_or_failing_redis_saves_nothing_and_never_raises(script: _Script | None) -> None:
    alerts: Final = _Alerts()
    await _log(_logger(script, alerts), _kwargs())
    assert alerts.recorded == ()


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


def _anthropic_stream(usage: Mapping[str, int]) -> httpx.Response:
    events: Final = (
        {"type": "message_start", "message": {"id": "msg_s", "role": "assistant", "content": [], "usage": usage}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    )
    body: Final = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


async def _call(
    respx_mock: respx.MockRouter,
    reply: httpx.Response,
    stream: bool = False,
    api_key: str = "sk-ant-test",
    api_base: str | None = None,
    no_log: bool = False,
) -> litellm.ModelResponse | None:
    respx_mock.post(api_base or _API_BASE).mock(return_value=reply)
    response: Final = await litellm.acompletion(
        model="anthropic/claude-sonnet-5-5",
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": "shared policy", "cache_control": {"type": "ephemeral"}}],
            },
            {"role": "user", "content": "question"},
        ],
        api_key=api_key,
        api_base=api_base,
        stream=stream,
        metadata={"model_info": {"id": "claude-east"}},
        **{"no-log": no_log},
    )
    if isinstance(response, litellm.CustomStreamWrapper):
        _ = [chunk async for chunk in response]
        return None
    assert isinstance(response, litellm.ModelResponse)
    return response


async def _settle(script: _Script, count: int) -> None:
    await drain_logging_tasks()
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=_GUARD_SECONDS)
    await script.until(count)


@pytest.fixture
def watched(monkeypatch: pytest.MonkeyPatch) -> _Script:
    script: Final = _Script()
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "callbacks", [PromptCacheAlertLogger(_KEY, lambda: script, _Alerts())])
    return script


_ZEROS: Final = _usage(0, 0)


@pytest.mark.parametrize(
    ("usage", "expected"), [(_ZEROS, "miss"), ({}, "reset")], ids=["reported-zeros", "counters-absent"]
)
async def test_a_real_completion_reads_the_raw_counters_and_leaves_the_response_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    watched: _Script,
    usage: Mapping[str, int],
    expected: str,
) -> None:
    """Normalized usage reports 0 cache tokens either way, so only the raw body can tell them apart."""
    observed: Final = await _call(respx_mock, _anthropic_reply(usage))
    await _settle(watched, 1)
    monkeypatch.setattr(litellm, "callbacks", [])
    unobserved: Final = await _call(respx_mock, _anthropic_reply(usage))
    assert watched.events() == [expected]
    assert observed is not None and unobserved is not None
    assert observed.choices[0].model_dump() == unobserved.choices[0].model_dump()
    assert observed.usage == unobserved.usage


async def test_a_streamed_request_between_misses_resets_the_deployment_those_misses_count_on(
    respx_mock: respx.MockRouter, watched: _Script
) -> None:
    """Streamed attempts log an empty API key, so they cannot share the assessed configuration. The reset lands on
    the deployment key every assessed streak on it reads."""
    await _call(respx_mock, _anthropic_reply(_usage(0, 2048)))
    await _settle(watched, 1)
    await _call(respx_mock, _anthropic_reply(_ZEROS))
    await _settle(watched, 2)
    await _call(respx_mock, _anthropic_stream(_usage(512, 0)), stream=True)
    await _settle(watched, 3)
    await _call(respx_mock, _anthropic_reply(_ZEROS))
    await _settle(watched, 4)
    assert watched.events() == ["proof", "miss", "reset", "miss"]
    reset_keys: Final = {keys[0] for keys, _ in watched.calls}
    assert len(reset_keys) == 1


async def test_the_logged_account_and_endpoint_split_configurations_and_redaction_keeps_the_prefix(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, watched: _Script
) -> None:
    await _call(respx_mock, _anthropic_reply(_ZEROS))
    await _call(respx_mock, _anthropic_reply(_ZEROS), api_key="sk-ant-other")
    await _call(respx_mock, _anthropic_reply(_ZEROS), api_base="https://gateway.example/v1/messages")
    monkeypatch.setattr(litellm, "turn_off_message_logging", True)
    await _call(respx_mock, _anthropic_reply(_ZEROS))
    await _settle(watched, 4)
    states: Final = [keys[1] for keys, _ in watched.calls]
    assert watched.events() == ["miss"] * 4
    assert len(set(states[:3])) == 3
    assert states[3] == states[0]
    assert all("sk-ant" not in key for key in states)


async def test_a_no_log_request_is_never_observed_even_when_it_read_the_cache(
    respx_mock: respx.MockRouter, watched: _Script
) -> None:
    """no-log requests stay out of the detector, so a hit among them is not seen and the alert stays advisory."""
    await _call(respx_mock, _anthropic_reply(_usage(0, 2048)))
    await _call(respx_mock, _anthropic_reply(_ZEROS))
    await _call(respx_mock, _anthropic_reply(_usage(2048, 0)), no_log=True)
    await _call(respx_mock, _anthropic_reply(_ZEROS))
    await _settle(watched, 3)
    assert watched.events() == ["proof", "miss", "miss"]


async def test_a_failed_completion_resets_its_deployment(respx_mock: respx.MockRouter, watched: _Script) -> None:
    with pytest.raises(litellm.InternalServerError):
        await _call(respx_mock, httpx.Response(500, json={"type": "error", "error": {"type": "api_error"}}))
    await _settle(watched, 1)
    assert watched.events() == ["reset"]


async def test_registration_is_opt_in_and_leaves_the_callback_lists_on_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy.proxy_server import register_prompt_cache_alerts

    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setenv("LITELLM_SALT_KEY", "registration-salt")
    async with AsyncExitStack() as resources:
        register_prompt_cache_alerts({}, MagicMock(), resources)
        assert litellm.callbacks == []
    async with AsyncExitStack() as resources:
        register_prompt_cache_alerts({"prompt_cache_alerts": True}, MagicMock(), resources)
        assert [type(callback) for callback in litellm.callbacks] == [PromptCacheAlertLogger]
    assert litellm.callbacks == []
