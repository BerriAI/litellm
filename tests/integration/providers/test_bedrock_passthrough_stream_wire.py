import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, wire_server

_MODEL_ID: Final = "anthropic.claude-sonnet-5-v1:0"
_EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
_REQUEST_BODY: Final = {"messages": [{"role": "user", "content": [{"text": "synthetic passthrough stream"}]}]}
_EVENTS: Final = (
    ("messageStart", {"role": "assistant"}),
    ("contentBlockDelta", {"delta": {"text": "bedrock stream control"}, "contentBlockIndex": 0}),
    ("messageStop", {"stopReason": "end_turn"}),
    ("metadata", {"usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}}),
)
_STREAM_BYTES: Final = b"".join(_aws_event_frame(kind, payload, "sc", "u") for kind, payload in _EVENTS)


def event_stream_peer(request: Request) -> Reply:
    assert request.method == "POST"
    assert request.target == f"/model/{_MODEL_ID}/converse-stream"
    assert json.loads(request.body)["messages"] == _REQUEST_BODY["messages"]
    return Reply(body=_STREAM_BYTES, content_type=_EVENT_STREAM)


@pytest.mark.covers("other.provider_wire.bedrock.passthrough_stream_keeps_event_stream_content_type")
def test_bedrock_passthrough_converse_stream_response_carries_event_stream_content_type(gateway: Gateway) -> None:
    with wire_server(event_stream_peer) as wire, gateway.scenario() as scenario:
        deployment: Final = scenario.model(
            model=f"bedrock/{_MODEL_ID}",
            api_base=wire.url,
            aws_access_key_id="AKIASCRIPTEDPROVIDER",
            aws_secret_access_key="scripted-secret",
            aws_region_name="us-east-1",
        )
        response: Final = gateway.request("POST", f"/bedrock/model/{deployment}/converse-stream", _REQUEST_BODY)
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1, response.text
        assert response.headers.get("content-type") == _EVENT_STREAM, dict(response.headers)
        assert response.content == _STREAM_BYTES, response.text
