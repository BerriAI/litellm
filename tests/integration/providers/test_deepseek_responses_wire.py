import json
from typing import Final

from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway
from tests.integration._support.wire import Reply, Request, wire_server


def test_responses_proxy_preserves_native_deepseek_custom_tool(gateway: Gateway) -> None:
    model: Final = "deepseek-flash"
    tool: Final[JsonValue] = {"type": "custom", "name": "apply_patch", "format": {"type": "text"}}
    output: Final = [
        {
            "type": "custom_tool_call",
            "id": "ct_native",
            "call_id": "call_native",
            "name": "apply_patch",
            "input": "*** Begin Patch\n*** End Patch",
            "status": "completed",
        }
    ]
    usage: Final = {
        "input_tokens": 12,
        "output_tokens": 7,
        "total_tokens": 19,
        "input_tokens_details": {"cached_tokens": 5},
        "output_tokens_details": {"reasoning_tokens": 3},
    }

    def peer(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/responses", request.target
        assert request.headers["authorization"] == "Bearer synthetic-deepseek-key"
        assert JSON_OBJECT.validate_json(request.body) == {
            "model": model,
            "input": "Apply the patch",
            "tools": [tool],
        }
        return Reply(
            body=json.dumps(
                {
                    "id": "resp_native",
                    "object": "response",
                    "created_at": 1,
                    "model": model,
                    "status": "completed",
                    "output": output,
                    "usage": usage,
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [tool],
                    "store": False,
                }
            ).encode()
        )

    with wire_server(peer) as wire, gateway.scenario() as scenario:
        alias: Final = scenario.model(
            model=f"deepseek/{model}",
            api_base=wire.url,
            api_key="synthetic-deepseek-key",
        )
        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {"model": alias, "input": "Apply the patch", "tools": [tool]},
        )
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert body["output"] == output, response.text
        actual_usage: Final = JSON_OBJECT.validate_python(body["usage"])
        assert {key: value for key, value in actual_usage.items() if key != "cost"} == usage, response.text
        assert len(wire.drain()) == 1
