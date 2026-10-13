import base64
import json
from collections.abc import Mapping
from typing import Final

from pydantic import JsonValue

from integration._support.client import JSON_OBJECT, Gateway
from integration._support.provider import SharedProvider
from integration._support.wire import Reply, Request
from integration.translation.case import SSEFrame, StreamingTranslationTestCase, TranslationTestCase
from litellm.responses.utils import ResponsesAPIRequestUtils

TRANSPORT_HEADERS: Final = frozenset(
    {"host", "accept", "accept-encoding", "connection", "content-length", "user-agent"}
)


def expected_response_for_deployment(
    expected: Mapping[str, JsonValue], model_id: str | None
) -> Mapping[str, JsonValue]:
    identifier: Final = expected.get("id")
    if not isinstance(identifier, str) or not ResponsesAPIRequestUtils._is_litellm_encoded_response_id(identifier):
        return expected
    assert model_id, "An encoded response ID must identify the selected deployment"
    provider, _, response = base64.b64decode(identifier.removeprefix("resp_")).decode().split(";", 2)
    updated: Final = f"{provider};model_id:{model_id};{response}"
    return {
        **expected,
        "id": "resp_" + base64.b64encode(updated.encode()).decode(),
    }


def assert_translation(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    provider.expect(Reply(status=case.mock_provider_status_code, body=json.dumps(case.mock_provider_response).encode()))
    response: Final = gateway.request("POST", case.litellm_endpoint, case.litellm_request)
    received: Final = provider.received()
    _assert_provider_request(
        received,
        case.expected_provider_endpoint,
        case.expected_provider_headers,
        case.expected_provider_request,
    )
    assert response.status_code == case.expected_litellm_status_code, response.text
    model_id: Final = response.headers.get("x-litellm-model-id")
    expected: Final = expected_response_for_deployment(case.expected_litellm_response, model_id)
    if expected is not case.expected_litellm_response:
        deployments: Final = gateway.get("/model/info")["data"]
        assert isinstance(deployments, list), deployments
        assert any(
            isinstance(deployment, dict)
            and deployment.get("model_name") == case.litellm_request["model"]
            and isinstance(deployment.get("model_info"), dict)
            and deployment["model_info"].get("id") == model_id
            for deployment in deployments
        ), "The encoded response ID must match the configured deployment"
    assert response.json() == expected


def _assert_provider_request(
    received: tuple[Request, ...],
    endpoint: str,
    headers: Mapping[str, str],
    request: Mapping[str, JsonValue],
) -> None:
    assert [(sent_request.method, sent_request.target) for sent_request in received] == [("POST", endpoint)]
    sent: Final = received[0]
    assert {
        name: value
        for name, value in sent.headers.items()
        if name not in TRANSPORT_HEADERS and not name.startswith("x-stainless-")
    } == dict(headers)
    assert json.loads(sent.body) == request


def encode_sse_frame(frame: SSEFrame) -> bytes:
    serialized: Final = frame.data if isinstance(frame.data, str) else json.dumps(frame.data)
    event_line: Final = f"event: {frame.event}\n" if frame.event is not None else ""
    return f"{event_line}data: {serialized}\n\n".encode()


def _parse_sse_frame(block: str) -> SSEFrame | None:
    lines: Final = tuple(line for line in block.splitlines() if line and not line.startswith(":"))
    if not lines:
        return None
    event_line: Final = next((line for line in lines if line.startswith("event:")), None)
    event: Final = event_line.removeprefix("event:").removeprefix(" ") if event_line is not None else None
    data_lines: Final = tuple(line for line in lines if line.startswith("data:"))
    if not data_lines:
        return None
    data: Final = "\n".join(line.removeprefix("data:").removeprefix(" ") for line in data_lines)
    if data == "[DONE]":
        return SSEFrame(event=event, data="[DONE]")
    return SSEFrame(event=event, data=JSON_OBJECT.validate_python(json.loads(data)))


def parse_sse_frames(body: str) -> tuple[SSEFrame, ...]:
    normalized: Final = body.replace("\r\n", "\n")
    blocks: Final = tuple(_parse_sse_frame(block) for block in normalized.split("\n\n"))
    return tuple(frame for frame in blocks if frame is not None)


def assert_streaming_translation(
    case: StreamingTranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    provider.expect(
        Reply(
            content_type="text/event-stream",
            chunks=tuple(encode_sse_frame(frame) for frame in case.mock_provider_frames),
        )
    )
    response: Final = gateway.request("POST", case.litellm_endpoint, case.litellm_request)
    _assert_provider_request(
        provider.received(),
        case.expected_provider_endpoint,
        case.expected_provider_headers,
        case.expected_provider_request,
    )
    assert response.status_code == 200, response.text
    assert parse_sse_frames(response.text) == case.expected_litellm_frames
