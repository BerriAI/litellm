from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Final, Literal

import pytest
import yaml
from integration._support.client import JSON_OBJECT, eventually, gateway_from_environment, object_value
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.vertex import service_account_json
from integration._support.wire import Wire, wire_server
from integration.providers._gemini_tts_voice_support import (
    AUDIO_TOKENS,
    BACKEND,
    GEMINI_API_KEY,
    PCM,
    PROMPT_TOKENS,
    REJECTED_PREFIX,
    VERTEX_LOCATION,
    VERTEX_PROJECT,
    gemini_peer,
    marker,
    received_text,
    wav_payload,
)
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

Provider = Literal["gemini", "vertex_ai"]
MODELS: Final[Mapping[Provider, str]] = {"gemini": "tts-billing-gemini", "vertex_ai": "tts-billing-vertex"}
TPM_MODEL: Final = "tts-billing-tpm"
TPM_LIMIT: Final = PROMPT_TOKENS + AUDIO_TOKENS
BINARY_RESPONSE: Final = "HttpxBinaryResponseContent"
CALLBACK_LOG: Final = "TTS_BILLING_CALLBACK_LOG"
CALLBACK_MODULE: Final = """
import json
import os

from litellm.integrations.custom_logger import CustomLogger


def _event(kind, kwargs, response_obj):
    payload = kwargs.get("standard_logging_object") or {}
    metadata = payload.get("metadata") or {}
    response = payload.get("response")
    usage = response.get("usage") if isinstance(response, dict) else None
    return {
        "kind": kind,
        "key": metadata.get("user_api_key_hash"),
        "call_type": kwargs.get("call_type"),
        "response_type": type(response_obj).__name__,
        "response_cost": payload.get("response_cost"),
        "prompt_tokens": payload.get("prompt_tokens"),
        "completion_tokens": payload.get("completion_tokens"),
        "cache_hit": payload.get("cache_hit"),
        "saved_cache_cost": payload.get("saved_cache_cost"),
        "cache_key": payload.get("cache_key"),
        "logged_usage_total": usage.get("total_tokens") if isinstance(usage, dict) else None,
    }


class SpeechEvents(CustomLogger):
    def _write(self, line):
        with open(os.environ["TTS_BILLING_CALLBACK_LOG"], "a") as sink:
            sink.write(json.dumps(line) + "\\n")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._write(_event("sync_success", kwargs, response_obj))

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._write(_event("async_success", kwargs, response_obj))

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._write(_event("sync_failure", kwargs, response_obj))

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._write(_event("async_failure", kwargs, response_obj))


speech_events = SpeechEvents()
"""
SPEND_ROWS_SQL: Final = (
    'SELECT call_type, spend, prompt_tokens, completion_tokens, cache_hit FROM "LiteLLM_SpendLogs"'
    ' WHERE api_key = %s ORDER BY "startTime"'
)
KEY_SPEND_SQL: Final = 'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token = %s'

Event = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class BillingProxy:
    proxy: OwnedProxy
    callback_log: Path


@dataclass(frozen=True, slots=True)
class Sent:
    status: int
    body: bytes
    cache_key: str


def _hashed(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def _model_list(token_url: str) -> list[JsonValue]:
    return [
        {"model_name": MODELS["gemini"], "litellm_params": {"model": f"gemini/{BACKEND}", "api_key": GEMINI_API_KEY}},
        {
            "model_name": TPM_MODEL,
            "litellm_params": {"model": f"gemini/{BACKEND}", "api_key": GEMINI_API_KEY, "tpm": TPM_LIMIT},
        },
        {
            "model_name": MODELS["vertex_ai"],
            "litellm_params": {
                "model": f"vertex_ai/{BACKEND}",
                "vertex_project": VERTEX_PROJECT,
                "vertex_location": VERTEX_LOCATION,
                "vertex_credentials": service_account_json(VERTEX_PROJECT, token_url.rstrip("/")),
            },
        },
    ]


def _config(directory: Path, token_url: str, *, cached: bool) -> Path:
    stock: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    settings: Final = object_value(stock["litellm_settings"])
    kept: Final = {name: value for name, value in settings.items() if name not in {"cache", "cache_params"}}
    cache: Final[dict[str, JsonValue]] = (
        {"cache": True, "cache_params": {**object_value(settings["cache_params"]), "namespace": marker()}}
        if cached
        else {"cache": False}
    )
    routing: Final[dict[str, JsonValue]] = (
        {}
        if cached
        else {
            "routing_strategy": "usage-based-routing-v2",
            "redis_host": os.environ["REDIS_HOST"],
            "redis_port": int(os.environ["REDIS_PORT"]),
        }
    )
    (directory / "tts_billing_callbacks.py").write_text(CALLBACK_MODULE)
    path: Final = directory / "tts-billing.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **stock,
                "model_list": _model_list(token_url),
                "litellm_settings": {**kept, **cache, "callbacks": "tts_billing_callbacks.speech_events"},
                "router_settings": {**object_value(stock["router_settings"]), "num_retries": 0, **routing},
            }
        )
    )
    return path


def _billing_proxy(wire: Wire, directory: Path, *, cached: bool) -> Iterator[BillingProxy]:
    callback_log: Final = directory / "callback_events.jsonl"
    callback_log.touch()
    with gateway_from_environment() as gateway:
        config: Final = _config(directory, gateway.upstream_url, cached=cached)
        overrides: Final = {"GEMINI_API_BASE": wire.url, "VERTEXAI_API_BASE": wire.url, CALLBACK_LOG: str(callback_log)}
        with owned_proxy_process(gateway, directory, overrides, config=config, workers=2) as proxy:
            yield BillingProxy(proxy, callback_log)


@pytest.fixture(scope="module")
def tts_wire() -> Iterator[Wire]:
    with wire_server(gemini_peer) as wire:
        yield wire


@pytest.fixture(scope="module")
def uncached(tts_wire: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[BillingProxy]:
    yield from _billing_proxy(tts_wire, tmp_path_factory.mktemp("tts-billing-uncached"), cached=False)


@pytest.fixture(scope="module")
def cached(tts_wire: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[BillingProxy]:
    yield from _billing_proxy(tts_wire, tmp_path_factory.mktemp("tts-billing-cached"), cached=True)


def _speech(billing: BillingProxy, model: str, text: str, key: str, voice: str = "Kore") -> Sent:
    response: Final = billing.proxy.gateway.request(
        "POST", "/v1/audio/speech", {"model": model, "input": text, "voice": voice}, key=key
    )
    return Sent(response.status_code, response.content, response.headers.get("x-litellm-cache-key", ""))


def _upstream_calls(wire: Wire, text: str) -> int:
    return sum(1 for request in wire.drain() if received_text(request) == text)


def _spend_rows(billing: BillingProxy, key: str, count: int) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: billing.proxy.gateway.rows(SPEND_ROWS_SQL, (_hashed(key),)), lambda rows: len(rows) == count, seconds=70
    )


def _key_spend(billing: BillingProxy, key: str, expected: float) -> JsonValue:
    rows: Final = eventually(
        lambda: billing.proxy.gateway.rows(KEY_SPEND_SQL, (_hashed(key),)),
        lambda found: len(found) == 1 and found[0]["spend"] == pytest.approx(expected),
        seconds=70,
        return_last_on_timeout=True,
    )
    return rows[0]["spend"]


def _chat_spend(billing: BillingProxy, model: str, key: str) -> float:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "messages": [{"role": "user", "content": marker()}],
        "modalities": ["audio"],
        "audio": {"voice": "Kore", "format": "pcm16"},
    }
    response: Final = billing.proxy.gateway.request("POST", "/v1/chat/completions", body, key=key)
    assert response.status_code == 200, response.text
    row: Final = _spend_rows(billing, key, 1)[0]
    assert (row["prompt_tokens"], row["completion_tokens"]) == (PROMPT_TOKENS, AUDIO_TOKENS), row
    spend: Final = row["spend"]
    assert isinstance(spend, float) and spend > 0, row
    return spend


def _events(billing: BillingProxy, key: str) -> tuple[Event, ...]:
    complete_lines: Final = billing.callback_log.read_text().split("\n")[:-1]
    events: Final = tuple(JSON_OBJECT.validate_json(line) for line in complete_lines)
    return tuple(event for event in events if event["key"] == _hashed(key))


def _speech_events(billing: BillingProxy, key: str, count: int) -> tuple[Event, ...]:
    return eventually(
        lambda: _events(billing, key),
        lambda found: sum(1 for event in found if event["response_type"] == BINARY_RESPONSE) >= count,
        seconds=70,
    )


def _summary(event: Event) -> tuple[JsonValue, ...]:
    return (event["call_type"], event["response_type"], event["prompt_tokens"], event["completion_tokens"])


@pytest.mark.parametrize("provider", ["gemini", "vertex_ai"])
def test_speech_calls_google_once_and_bills_and_logs_its_token_usage_once(
    uncached: BillingProxy, tts_wire: Wire, provider: Provider
) -> None:
    with uncached.proxy.gateway.scenario() as scenario:
        speech_key: Final = scenario.key()
        text: Final = marker()
        sent: Final = _speech(uncached, MODELS[provider], text, speech_key)
        assert sent.status == 200, sent.body[:300]
        assert wav_payload(sent.body) == PCM
        assert _upstream_calls(tts_wire, text) == 1
        chat_spend: Final = _chat_spend(uncached, MODELS[provider], scenario.key())
        row: Final = _spend_rows(uncached, speech_key, 1)[0]
        assert row["call_type"] == "aspeech", row
        assert (row["prompt_tokens"], row["completion_tokens"]) == (PROMPT_TOKENS, AUDIO_TOKENS), row
        assert row["spend"] == pytest.approx(chat_spend), row
        assert _key_spend(uncached, speech_key, chat_spend) == pytest.approx(chat_spend)
        events: Final = _speech_events(uncached, speech_key, 1)
        assert [_summary(event) for event in events] == [("aspeech", BINARY_RESPONSE, PROMPT_TOKENS, AUDIO_TOKENS)]
        assert events[0]["response_cost"] == pytest.approx(chat_spend), events
        assert events[0]["logged_usage_total"] == PROMPT_TOKENS + AUDIO_TOKENS, events


def test_cached_speech_repeat_is_a_free_cache_hit_after_a_billed_first_call(
    cached: BillingProxy, tts_wire: Wire
) -> None:
    model: Final = MODELS["gemini"]
    with cached.proxy.gateway.scenario() as scenario:
        speech_key: Final = scenario.key()
        text: Final = marker()
        first: Final = _speech(cached, model, text, speech_key)
        repeat: Final = _speech(cached, model, text, speech_key)
        assert (first.status, repeat.status) == (200, 200), (first.body[:300], repeat.body[:300])
        assert wav_payload(first.body) == PCM and repeat.body == first.body
        assert _upstream_calls(tts_wire, text) == 1
        assert repeat.cache_key, "the cached repeat carried no x-litellm-cache-key header"
        chat_spend: Final = _chat_spend(cached, model, scenario.key())
        billed, hit = _spend_rows(cached, speech_key, 2)
        assert billed["cache_hit"] != "True" and billed["spend"] == pytest.approx(chat_spend), billed
        assert (billed["prompt_tokens"], billed["completion_tokens"]) == (PROMPT_TOKENS, AUDIO_TOKENS), billed
        assert (hit["cache_hit"], hit["spend"]) == ("True", 0.0), hit
        assert _key_spend(cached, speech_key, chat_spend) == pytest.approx(chat_spend)
        events: Final = _speech_events(cached, speech_key, 2)
        assert len(events) == 2, events
        hits: Final = tuple(event for event in events if event["cache_hit"] is True)
        assert len(hits) == 1, events
        assert (hits[0]["response_cost"], hits[0]["cache_key"]) == (0.0, repeat.cache_key), hits
        assert hits[0]["saved_cache_cost"] == pytest.approx(chat_spend), hits
        paid: Final = tuple(event for event in events if event["cache_hit"] is not True)
        assert len(paid) == 1 and paid[0]["response_cost"] == pytest.approx(chat_spend), events


def test_speech_rejected_by_google_reaches_the_caller_and_logs_one_failure_after_one_call(
    uncached: BillingProxy, tts_wire: Wire
) -> None:
    with uncached.proxy.gateway.scenario() as scenario:
        key: Final = scenario.key()
        text: Final = marker()
        sent: Final = _speech(uncached, MODELS["gemini"], text, key, voice="Custom-Voice")
        assert sent.status == 400, sent.body[:300]
        assert f"{REJECTED_PREFIX}Custom-Voice" in sent.body.decode(), sent.body[:300]
        assert _upstream_calls(tts_wire, text) == 1
        events: Final = eventually(lambda: _events(uncached, key), lambda found: len(found) >= 2, seconds=70)
        logged: Final = sorted((str(event["kind"]), event["call_type"]) for event in events)
        assert logged == [("async_failure", "aspeech"), ("sync_failure", "aspeech")], events


def test_a_speech_reply_fills_the_deployment_tpm_limit_so_the_next_request_gets_429(
    uncached: BillingProxy, tts_wire: Wire
) -> None:
    eventually(lambda: datetime.now(UTC), lambda now: now.second < 10, seconds=65)
    with uncached.proxy.gateway.scenario() as scenario:
        key: Final = scenario.key()
        first_text: Final = marker()
        first: Final = _speech(uncached, TPM_MODEL, first_text, key)
        assert first.status == 200, first.body[:300]
        _spend_rows(uncached, key, 1)
        second_text: Final = marker()
        second: Final = _speech(uncached, TPM_MODEL, second_text, key)
        assert second.status == 429, second.body[:300]
        assert b"No deployments available" in second.body, second.body[:300]
        received: Final = tuple(received_text(request) for request in tts_wire.drain())
        assert (received.count(first_text), received.count(second_text)) == (1, 0)
