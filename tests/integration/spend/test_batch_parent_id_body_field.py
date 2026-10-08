from __future__ import annotations

import io
import os
import uuid
import wave
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.upstream import delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import BinaryResponse, JsonResponse
from pydantic import JsonValue

FORGED_PARENT: Final = "batch_forged_by_caller"

_Reply = Callable[[], JsonResponse | BinaryResponse]
_Send = Callable[[Gateway, str, str, dict[str, JsonValue]], httpx.Response]


def _unique_wav() -> bytes:
    output: Final = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(os.urandom(2 * 16000))
    return output.getvalue()


def _moderation_reply() -> JsonResponse:
    return JsonResponse(
        content_type="application/json",
        body={
            "id": f"modr-{uuid.uuid4().hex}",
            "model": "omni-moderation-latest",
            "results": [{"flagged": False, "categories": {}, "category_scores": {}}],
        },
    )


def _speech_reply() -> BinaryResponse:
    return BinaryResponse(content_type="audio/mpeg", length=2048)


def _transcription_reply() -> JsonResponse:
    return JsonResponse(content_type="application/json", body={"text": "hello"})


def _chat_reply() -> JsonResponse:
    return JsonResponse(
        content_type="application/json",
        body={
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
        },
    )


def _embedding_reply() -> JsonResponse:
    return JsonResponse(
        content_type="application/json",
        body={
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 7, "total_tokens": 7},
        },
    )


def _responses_reply() -> JsonResponse:
    return JsonResponse(
        content_type="application/json",
        body={
            "id": f"resp_{uuid.uuid4().hex}",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-4o-mini",
            "output": [],
            "usage": {"input_tokens": 9, "output_tokens": 3, "total_tokens": 12},
        },
    )


def _messages_reply() -> JsonResponse:
    return JsonResponse(
        content_type="application/json",
        body={
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 9, "output_tokens": 3},
        },
    )


def _chat(gateway: Gateway, model: str, key: str, extra: dict[str, JsonValue]) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": "hi"}], **extra},
        key=key,
    )


def _embedding(gateway: Gateway, model: str, key: str, extra: dict[str, JsonValue]) -> httpx.Response:
    return gateway.request("POST", "/v1/embeddings", {"model": model, "input": "hi", **extra}, key=key)


def _responses(gateway: Gateway, model: str, key: str, extra: dict[str, JsonValue]) -> httpx.Response:
    return gateway.request("POST", "/v1/responses", {"model": model, "input": "hi", **extra}, key=key)


def _messages(gateway: Gateway, model: str, key: str, extra: dict[str, JsonValue]) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/messages",
        {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}], **extra},
        key=key,
    )


def _moderation(gateway: Gateway, model: str, key: str, extra: dict[str, JsonValue]) -> httpx.Response:
    return gateway.request("POST", "/v1/moderations", {"model": model, "input": "safe text", **extra}, key=key)


def _speech(gateway: Gateway, model: str, key: str, extra: dict[str, JsonValue]) -> httpx.Response:
    return gateway.request(
        "POST", "/v1/audio/speech", {"model": model, "input": "hello world", "voice": "alloy", **extra}, key=key
    )


def _transcription(gateway: Gateway, model: str, key: str, extra: dict[str, JsonValue]) -> httpx.Response:
    fields: Final = {"model": model, **{name: str(value) for name, value in extra.items()}}
    return gateway.request_multipart(
        "/v1/audio/transcriptions", fields, {"file": ("audio.wav", _unique_wav(), "audio/wav")}, key=key
    )


ROUTES: Final = (
    pytest.param("openai/omni-moderation-latest", _moderation_reply, _moderation, "amoderation", id="moderations"),
    pytest.param("openai/tts-1", _speech_reply, _speech, "aspeech", id="audio-speech"),
    pytest.param("openai/whisper-1", _transcription_reply, _transcription, "atranscription", id="audio-transcriptions"),
    pytest.param("openai/gpt-4o-mini", _chat_reply, _chat, "acompletion", id="chat-completions"),
    pytest.param("openai/text-embedding-3-small", _embedding_reply, _embedding, "aembedding", id="embeddings"),
    pytest.param("openai/gpt-4o-mini", _responses_reply, _responses, "aresponses", id="responses"),
    pytest.param("anthropic/claude-haiku-4-5", _messages_reply, _messages, "anthropic_messages", id="messages"),
)
FORGED_VALUES: Final = (
    pytest.param("", id="empty-string"),
    pytest.param(5, id="int"),
    pytest.param(["batch_forged_by_caller"], id="list"),
    pytest.param({"id": "batch_forged_by_caller"}, id="object"),
    pytest.param("b" * 5000, id="5kb-string"),
)


def _scripted_model(gateway: Gateway, scenario: Scenario, provider_model: str, reply: _Reply) -> str:
    marker: Final = "batch-parent-body-" + uuid.uuid4().hex[:12]
    handle: Final = register_scenario(marker, reply())
    scenario.cleanups.callback(delete_scenario, handle)
    return scenario.model(model=provider_model, api_base=f"{gateway.upstream_url}/{marker}")


def _spend_rows(key: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        read_rows(
            'SELECT call_type, spend FROM "LiteLLM_SpendLogs" WHERE api_key=%s ORDER BY "startTime"',
            (sha256(key.encode()).hexdigest(),),
        )
    )


def _key_spend(key: str) -> float:
    rows: Final = read_rows(
        'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (sha256(key.encode()).hexdigest(),)
    )
    assert len(rows) == 1, rows
    return float(str(rows[0]["spend"]))


def _assert_forged_request_bills_like_control(
    gateway: Gateway, provider_model: str, reply: _Reply, send: _Send, call_type: str, forged_value: JsonValue
) -> None:
    with gateway.scenario() as scenario:
        control_model: Final = _scripted_model(gateway, scenario, provider_model, reply)
        forged_model: Final = _scripted_model(gateway, scenario, provider_model, reply)
        key: Final = scenario.key(models=[control_model, forged_model])

        control: Final = send(gateway, control_model, key, {})
        assert control.status_code == 200, control.text
        control_rows: Final = eventually(lambda: _spend_rows(key), lambda values: len(values) == 1, seconds=70)
        assert control_rows[0]["call_type"] == call_type, control_rows
        control_spend: Final = float(str(control_rows[0]["spend"]))
        eventually(lambda: _key_spend(key), lambda value: value == pytest.approx(control_spend), seconds=70)

        forged: Final = send(gateway, forged_model, key, {"batch_parent_id": forged_value})
        assert forged.status_code == 200, forged.text
        rows: Final = eventually(lambda: _spend_rows(key), lambda values: len(values) == 2, seconds=70)
        assert [row["call_type"] for row in rows] == [call_type, call_type], rows
        assert float(str(rows[1]["spend"])) == pytest.approx(control_spend), rows
        eventually(lambda: _key_spend(key), lambda value: value == pytest.approx(2 * control_spend), seconds=70)


@pytest.mark.parametrize(("provider_model", "reply", "send", "call_type"), ROUTES)
def test_a_caller_supplied_batch_parent_id_is_still_tracked_like_any_request(
    gateway: Gateway,
    provider_model: str,
    reply: _Reply,
    send: _Send,
    call_type: str,
) -> None:
    _assert_forged_request_bills_like_control(gateway, provider_model, reply, send, call_type, FORGED_PARENT)


@pytest.mark.parametrize("forged_value", FORGED_VALUES)
def test_any_shape_of_batch_parent_id_on_a_speech_request_is_still_tracked(
    gateway: Gateway, forged_value: JsonValue
) -> None:
    _assert_forged_request_bills_like_control(gateway, "openai/tts-1", _speech_reply, _speech, "aspeech", forged_value)


def test_a_concurrent_burst_mixing_forged_and_plain_requests_bills_every_request(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = _scripted_model(gateway, scenario, "openai/tts-1", _speech_reply)
        key: Final = scenario.key(models=[model])
        warm_up: Final = _speech(gateway, model, key, {})
        assert warm_up.status_code == 200, warm_up.text
        eventually(lambda: _spend_rows(key), lambda values: len(values) == 1, seconds=70)
        extras: Final = tuple({"batch_parent_id": FORGED_PARENT} if index % 2 else {} for index in range(20))

        with ThreadPoolExecutor(max_workers=10) as pool:
            responses: Final = tuple(pool.map(lambda extra: _speech(gateway, model, key, extra), extras))

        assert [response.status_code for response in responses] == [200] * 20, [r.text for r in responses]
        rows: Final = eventually(lambda: _spend_rows(key), lambda values: len(values) == 21, seconds=90)
        assert {row["call_type"] for row in rows} == {"aspeech"}, rows
        expected_spend: Final = 21 * float(str(rows[0]["spend"]))
        eventually(lambda: _key_spend(key), lambda value: value == pytest.approx(expected_spend), seconds=90)
