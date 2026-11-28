import base64
import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "gpt-6.2-sol"
_API_KEY: Final = "synthetic-openai-key"
_PREFIX: Final = "litellm_encrypted_reasoning:"
_MARKER: Final = "litellm_turn_settings:"
_SEGMENT: Final = "eyJyZWFzb25pbmdfZWZmb3J0IjoiaGlnaCJ9"
_BLOB: Final = "gAAAAABsynthetic-encrypted-reasoning-blob"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_JSON_OBJECTS: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])


def _enveloped(blob: str) -> str:
    return f"{_PREFIX}{_MARKER}{_SEGMENT};{blob}"


def _responses_reply(identity: str) -> bytes:
    return json.dumps(
        {
            "id": f"resp_{identity}",
            "object": "response",
            "created_at": 1789788253,
            "status": "completed",
            "model": _BACKEND,
            "output": [
                {
                    "type": "message",
                    "id": f"msg_{identity}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done.", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 41, "output_tokens": 5, "total_tokens": 46},
        }
    ).encode()


def _replayed_encrypted_contents(request: Request) -> tuple[JsonValue, ...]:
    items: Final = _JSON_OBJECTS.validate_python(_JSON_OBJECT.validate_json(request.body)["input"])
    return tuple(item.get("encrypted_content") for item in items if item.get("type") == "reasoning")


def _send_history(gateway: Gateway, reasoning_block: dict[str, JsonValue]) -> tuple[Request, ...]:
    identity: Final = f"openai-reasoning-envelope-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/responses"
        return Reply(body=_responses_reply(identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 2048,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "messages": [
                    {"role": "user", "content": f"Fix the failing test {identity}."},
                    {"role": "assistant", "content": [reasoning_block, {"type": "text", "text": "Reading the test."}]},
                    {"role": "user", "content": "Continue."},
                ],
            },
        )
        requests: Final = wire.drain()
        assert response.status_code == 200, response.text
        assert len(requests) == 1, response.text
        return requests


_WRAPPED_BLOB: Final = f"litellm_enc:{base64.b64encode(b'model_id:openai-deployment-a').decode()};{_BLOB}"


def _thinking(signature: str) -> dict[str, JsonValue]:
    return {"type": "thinking", "thinking": "Plan.", "signature": signature}


@pytest.mark.parametrize("blob", [_BLOB, _WRAPPED_BLOB], ids=["plain-blob", "deployment-tagged-blob"])
def test_enveloped_thinking_signature_reaches_openai_like_the_same_blob_without_settings(
    gateway: Gateway, blob: str
) -> None:
    """regression: a thinking signature carrying turn settings replays upstream exactly like the bare signature."""
    enveloped: Final = _send_history(gateway, _thinking(_enveloped(blob)))
    bare: Final = _send_history(gateway, _thinking(f"{_PREFIX}{blob}"))
    assert _replayed_encrypted_contents(enveloped[0]) == _replayed_encrypted_contents(bare[0]), enveloped[0].body


def test_enveloped_redacted_thinking_data_replays_only_the_encrypted_content(gateway: Gateway) -> None:
    """regression: the turn settings segment in redacted thinking data never reaches OpenAI."""
    requests: Final = _send_history(gateway, {"type": "redacted_thinking", "data": _enveloped(_BLOB)})
    assert _replayed_encrypted_contents(requests[0]) == (_BLOB,), requests[0].body


def test_legacy_thinking_signature_replays_the_whole_encrypted_content(gateway: Gateway) -> None:
    """guard: a signature minted before the envelope existed replays its full blob unchanged."""
    requests: Final = _send_history(gateway, _thinking(f"{_PREFIX}{_BLOB}"))
    assert _replayed_encrypted_contents(requests[0]) == (_BLOB,), requests[0].body


def test_malformed_envelope_never_sends_the_settings_marker_as_encrypted_content(gateway: Gateway) -> None:
    """regression: an envelope missing its ';' separator never sends the marker or segment upstream."""
    requests: Final = _send_history(gateway, _thinking(f"{_PREFIX}{_MARKER}{_SEGMENT}"))
    replayed: Final = _replayed_encrypted_contents(requests[0])
    assert not any(isinstance(content, str) and _MARKER in content for content in replayed), requests[0].body
