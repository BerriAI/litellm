import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, wire_server

_MODEL_ID: Final = "anthropic.claude-sonnet-5-v1:0"
_ACTIONS: Final = ("converse", "invoke", "converse-stream", "invoke-with-response-stream")
_REQUEST_BODY: Final = {"messages": [{"role": "user", "content": [{"text": "synthetic passthrough allowlist"}]}]}
_CONVERSE_RESPONSE: Final = json.dumps(
    {
        "output": {"message": {"role": "assistant", "content": [{"text": "bedrock allowlist control"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15},
        "metrics": {"latencyMs": 1},
    }
).encode()
_STREAM_BYTES: Final = b"".join(
    _aws_event_frame(kind, payload, "sc", "u")
    for kind, payload in (
        ("messageStart", {"role": "assistant"}),
        ("contentBlockDelta", {"delta": {"text": "bedrock allowlist control"}, "contentBlockIndex": 0}),
        ("messageStop", {"stopReason": "end_turn"}),
        ("metadata", {"usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}}),
    )
)


def bedrock_peer(request: Request) -> Reply:
    assert request.method == "POST", request.target
    assert json.loads(request.body)["messages"] == _REQUEST_BODY["messages"], request.body
    if request.target.endswith("-stream"):
        return Reply(body=_STREAM_BYTES, content_type="application/vnd.amazon.eventstream")
    return Reply(body=_CONVERSE_RESPONSE)


@pytest.mark.covers("authz.key_models.bedrock_passthrough_route_model_is_enforced")
def test_key_scoped_to_one_model_cannot_call_another_through_bedrock_passthrough_routes(gateway: Gateway) -> None:
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        allowed: Final = scenario.model(
            model=f"bedrock/{_MODEL_ID}",
            api_base=wire.url,
            aws_access_key_id="AKIASCRIPTEDPROVIDER",
            aws_secret_access_key="scripted-secret",
            aws_region_name="us-east-1",
        )
        denied: Final = scenario.model(
            model=f"bedrock/{_MODEL_ID}",
            api_base=wire.url,
            aws_access_key_id="AKIASCRIPTEDPROVIDER",
            aws_secret_access_key="scripted-secret",
            aws_region_name="us-east-1",
        )
        key: Final = scenario.key(models=[allowed])
        for action in _ACTIONS:
            response: Final = gateway.request("POST", f"/bedrock/model/{denied}/{action}", _REQUEST_BODY, key=key)
            assert response.status_code == 403, f"{action}: {response.status_code} {response.text}"
            assert response.json()["error"]["type"] == "key_model_access_denied", f"{action}: {response.text}"
            assert wire.drain() == (), f"{action} reached the provider: {response.text}"
        for action in _ACTIONS:
            served: Final = gateway.request("POST", f"/bedrock/model/{allowed}/{action}", _REQUEST_BODY, key=key)
            assert served.status_code == 200, f"{action}: {served.status_code} {served.text}"
            assert tuple(request.target for request in wire.drain()) == (f"/model/{_MODEL_ID}/{action}",), served.text
