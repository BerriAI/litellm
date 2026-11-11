"""Provider pass-through routes relay ``safeguards`` and the streamed bytes verbatim: the classifier only serves
/v1/messages."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Final

import pytest
from integration._support.process import graceful_stop_seconds
from integration._support.safeguards_rig import DANGEROUS_TOOL_USE_BETA, SAFEGUARDS, Rig, nonce, safeguards_rig, sse
from integration._support.wire import Reply, Request
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(2 * graceful_stop_seconds() + 240)

RELAYED: Final = (
    sse({"type": "message_start", "message": {"id": "msg_relayed", "content": []}}),
    b"event: ping\ndata: not json\n\n",
    sse({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 3}}),
    sse({"type": "message_stop"}),
)


def relay(request: Request) -> Reply:
    return Reply(content_type="text/event-stream", chunks=RELAYED)


def provider_environment(wire_url: str) -> Mapping[str, str]:
    return {
        "ANTHROPIC_API_BASE": f"{wire_url}/anthropic-pt",
        "ANTHROPIC_API_KEY": "synthetic-anthropic-pt-key",
        "GEMINI_API_BASE": f"{wire_url}/gemini-pt",
        "GEMINI_API_KEY": "synthetic-gemini-pt-key",
    }


BEDROCK_DEPLOYMENT: Final = "safeguards-bedrock-pt"
BEDROCK_MODEL_ID: Final = "deepseek.v3-v1:0"


def bedrock_deployment(wire_url: str) -> Sequence[Mapping[str, JsonValue]]:
    return (
        {
            "model_name": BEDROCK_DEPLOYMENT,
            "litellm_params": {
                "model": f"bedrock/{BEDROCK_MODEL_ID}",
                "api_base": f"{wire_url}/bedrock-pt",
                "aws_access_key_id": "AKIASYNTHETIC0000000",
                "aws_secret_access_key": "synthetic-secret",
                "aws_region_name": "us-east-1",
            },
        },
    )


@pytest.fixture(scope="module")
def passthrough(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with safeguards_rig(
        tmp_path_factory.mktemp("passthrough"), environment=provider_environment, extra_models=bedrock_deployment
    ) as rig:
        yield rig


ROUTES: Final[Mapping[str, tuple[str, str, Mapping[str, JsonValue]]]] = {
    "anthropic": (
        "/anthropic/v1/messages",
        "/anthropic-pt/v1/messages",
        {"model": "claude-sonnet-4-6", "max_tokens": 64, "stream": True},
    ),
    "gemini": (
        "/gemini/v1beta/models/gemini-3.5-flash:streamGenerateContent?alt=sse",
        "/gemini-pt/v1beta/models/gemini-3.5-flash:streamGenerateContent",
        {},
    ),
    "bedrock": (
        f"/bedrock/model/{BEDROCK_DEPLOYMENT}/invoke-with-response-stream",
        f"/bedrock-pt/model/{BEDROCK_MODEL_ID}/invoke-with-response-stream",
        {"max_tokens": 64},
    ),
}


def request_body(route: str, request_nonce: str) -> Mapping[str, JsonValue]:
    content: Final[Mapping[str, JsonValue]] = (
        {"contents": [{"role": "user", "parts": [{"text": request_nonce}]}]}
        if route == "gemini"
        else {"messages": [{"role": "user", "content": request_nonce}]}
    )
    return {**ROUTES[route][2], **content, "safeguards": list(SAFEGUARDS)}


def route_headers(rig: Rig, route: str) -> Mapping[str, str]:
    headers: Final = rig.headers(None, DANGEROUS_TOOL_USE_BETA)
    return {**headers, "x-goog-api-key": rig.gateway.key} if route == "gemini" else headers


@pytest.mark.parametrize("route", tuple(ROUTES))
def test_passthrough_stream_relays_safeguards_and_bytes_verbatim(passthrough: Rig, route: str) -> None:
    request_nonce: Final = nonce()
    passthrough.register(request_nonce, relay)
    body: Final = request_body(route, request_nonce)
    with passthrough.gateway.client.stream(
        "POST", ROUTES[route][0], json=body, headers=route_headers(passthrough, route), timeout=60
    ) as response:
        status: Final = response.status_code
        raw: Final = response.read()
    assert status == 200, raw
    assert raw == b"".join(RELAYED), raw
    received: Final = passthrough.received(request_nonce)
    assert [request.target.split("?")[0] for request in received] == [ROUTES[route][1]], received
    assert passthrough.classifier_calls(received) == ()
    assert json.loads(received[0].body)["safeguards"] == body["safeguards"], received[0].body
