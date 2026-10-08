from __future__ import annotations

import io
import os
import uuid
import wave
from collections.abc import Callable
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


@pytest.mark.parametrize(("provider_model", "reply", "send", "call_type"), ROUTES)
def test_a_caller_supplied_batch_parent_id_is_still_tracked_like_any_request(
    gateway: Gateway,
    provider_model: str,
    reply: _Reply,
    send: _Send,
    call_type: str,
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

        forged: Final = send(gateway, forged_model, key, {"batch_parent_id": FORGED_PARENT})
        assert forged.status_code == 200, forged.text
        rows: Final = eventually(lambda: _spend_rows(key), lambda values: len(values) == 2, seconds=70)
        assert [row["call_type"] for row in rows] == [call_type, call_type], rows
        assert float(str(rows[1]["spend"])) == pytest.approx(control_spend), rows
        eventually(lambda: _key_spend(key), lambda value: value == pytest.approx(2 * control_spend), seconds=70)
