import json
import uuid
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

MODEL: Final = "gemini-2.5-flash"
GENERATE: Final = f"/v1beta/models/{MODEL}:generateContent"
STREAM: Final = f"/v1beta/models/{MODEL}:streamGenerateContent"
CARD: Final = "4111 1111 1111 1111"
PII_PROMPT: Final = f"Please update the billing on my account to card {CARD} and confirm when done."
ON_FLAGGED: Final = ("block", "mask", "monitor")
_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_ANSWER: Final = {"content": {"role": "model", "parts": [{"text": "Hi"}]}, "finishReason": "STOP"}
_USAGE: Final = {"promptTokenCount": 4, "candidatesTokenCount": 1, "totalTokenCount": 5}


class _LakeraMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str
    content: str


class _LakeraRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: tuple[_LakeraMessage, ...]


def _lakera_verdict(request: Request) -> Reply:
    texts: Final = tuple(enumerate(m.content for m in _LakeraRequest.model_validate_json(request.body).messages))
    flagged: Final = tuple(index for index, text in texts if CARD in text)
    return Reply(
        body=json.dumps(
            {
                "flagged": bool(flagged),
                "breakdown": [
                    {"detector_type": "pii/credit_card", "detected": True, "message_id": index} for index in flagged
                ],
                "payload": [
                    {
                        "detector_type": "pii/credit_card",
                        "start": texts[index][1].index(CARD),
                        "end": texts[index][1].index(CARD) + len(CARD),
                        "message_id": index,
                    }
                    for index in flagged
                ],
            }
        ).encode()
    )


def _gemini_reply(request: Request) -> Reply:
    if "streamGenerateContent" in request.target:
        return Reply(
            content_type="text/event-stream",
            chunks=(f"data: {json.dumps({'candidates': [_ANSWER], 'usageMetadata': _USAGE})}\r\n\r\n".encode(),),
        )
    return Reply(body=json.dumps({"candidates": [_ANSWER], "usageMetadata": _USAGE}).encode())


@dataclass(frozen=True, slots=True)
class NativeGeminiRig:
    owned: OwnedProxy
    lakera: Wire
    upstream: Wire


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[NativeGeminiRig]:
    directory: Final = tmp_path_factory.mktemp("lakera-gemini-native")
    with ExitStack() as stack:
        lakera: Final = stack.enter_context(wire_server(_lakera_verdict))
        upstream: Final = stack.enter_context(wire_server(_gemini_reply))
        config: Final = _JSON.validate_python(yaml.safe_load(_CONFIG.read_text()))
        models: Final = TypeAdapter(list[JsonValue]).validate_python(config.get("model_list") or [])
        path: Final = directory / "lakera-gemini-native.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    **config,
                    "model_list": [
                        *models,
                        {
                            "model_name": MODEL,
                            "litellm_params": {
                                "model": f"gemini/{MODEL}",
                                "api_base": f"{upstream.url}/v1beta",
                                "api_key": "synthetic-gemini-key",
                            },
                        },
                    ],
                    "guardrails": [
                        {
                            "guardrail_name": f"lakera-native-{on_flagged}",
                            "litellm_params": {
                                "guardrail": "lakera_v2",
                                "mode": "pre_call",
                                "default_on": False,
                                "on_flagged": on_flagged,
                                "api_base": lakera.url,
                                "api_key": "synthetic-lakera-key",
                            },
                        }
                        for on_flagged in ON_FLAGGED
                    ],
                }
            )
        )
        owned: Final = stack.enter_context(owned_proxy_process(session_gateway, directory, {}, config=path))
        yield NativeGeminiRig(owned, lakera, upstream)


def _body(prompt: str) -> dict[str, JsonValue]:
    return {
        "systemInstruction": {"parts": [{"text": "You are a billing assistant."}]},
        "contents": [
            {"role": "user", "parts": [{"text": "Hi"}]},
            {"role": "model", "parts": [{"text": "Hello, how can I help?"}]},
            {"role": "user", "parts": [{"text": prompt}]},
        ],
    }


def _send(rig: NativeGeminiRig, on_flagged: str, route: str, body: dict[str, JsonValue]) -> httpx.Response:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key(metadata={"guardrails": [f"lakera-native-{on_flagged}"]})
        rig.lakera.drain()
        rig.upstream.drain()
        return gateway.request("POST", route, body, key=key)


def _lakera_messages(rig: NativeGeminiRig) -> tuple[tuple[tuple[str, str], ...], ...]:
    return tuple(
        tuple((m.role, m.content) for m in _LakeraRequest.model_validate_json(request.body).messages)
        for request in rig.lakera.drain()
    )


def test_native_generate_content_sends_gemini_turns_to_lakera(rig: NativeGeminiRig) -> None:
    marker: Final = f"clean-{uuid.uuid4().hex}"

    response: Final = _send(rig, "block", GENERATE, _body(marker))

    assert response.status_code == 200, response.text
    assert _lakera_messages(rig) == (
        (
            ("system", "You are a billing assistant."),
            ("user", "Hi"),
            ("assistant", "Hello, how can I help?"),
            ("user", marker),
        ),
    )
    assert [_JSON.validate_json(request.body)["contents"] for request in rig.upstream.drain()] == [
        _body(marker)["contents"]
    ]


@pytest.mark.parametrize("route", [GENERATE, f"{STREAM}?alt=sse"])
@pytest.mark.parametrize("on_flagged", ["block", "mask"])
def test_card_on_native_route_is_blocked_before_the_provider(rig: NativeGeminiRig, on_flagged: str, route: str) -> None:
    response: Final = _send(rig, on_flagged, route, _body(PII_PROMPT))

    assert response.status_code == 400, response.text
    assert "Violated guardrail policy" in response.text
    assert len(_lakera_messages(rig)) == 1
    assert rig.upstream.drain() == ()


def test_native_stream_with_clean_prompt_is_inspected_once_and_relayed(rig: NativeGeminiRig) -> None:
    marker: Final = f"clean-{uuid.uuid4().hex}"

    response: Final = _send(rig, "block", f"{STREAM}?alt=sse", _body(marker))

    assert response.status_code == 200, response.text
    assert '"Hi"' in response.text
    assert [messages[-1] for messages in _lakera_messages(rig)] == [("user", marker)]
    assert len(rig.upstream.drain()) == 1


def test_monitor_on_native_route_forwards_the_card_unchanged(rig: NativeGeminiRig) -> None:
    body: Final = _body(PII_PROMPT)

    response: Final = _send(rig, "monitor", GENERATE, body)

    assert response.status_code == 200, response.text
    assert len(_lakera_messages(rig)) == 1
    assert [_JSON.validate_json(request.body)["contents"] for request in rig.upstream.drain()] == [body["contents"]]
