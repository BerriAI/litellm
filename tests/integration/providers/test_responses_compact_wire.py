from __future__ import annotations

import json
import uuid
from typing import Final

from integration._support.client import Gateway, Scenario
from integration._support.responses_vendor import same_response
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.responses_test_support import drain_contract_requests, model_discovery_reply
from openai.types.responses import CompactedResponse
from pydantic import JsonValue, TypeAdapter

_MODEL: Final = "gpt-5"
_API_KEY: Final = "synthetic-responses-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_ALIASES: Final = ("/v1/responses/compact", "/responses/compact", "/openai/v1/responses/compact")
_FUNCTION_TOOL: Final = {
    "type": "function",
    "name": "lookup",
    "description": "Look up a record",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    "strict": True,
}


def _deployment(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(
        model=f"openai/{_MODEL}",
        api_key=_API_KEY,
        api_base=f"{wire.url}/v1",
    )


def test_responses_compact_routes_forward_complete_request_and_typed_items(gateway: Gateway) -> None:
    reasoning_item: Final = {
        "id": f"rs_{uuid.uuid4().hex}",
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "Prior reasoning"}],
        "encrypted_content": "synthetic-encrypted-content",
    }
    function_output_item: Final = {
        "type": "function_call_output",
        "call_id": "call_previous",
        "output": "The record is ready.",
    }
    requests: Final = (
        {
            "input": "Compact this conversation.",
        },
        {
            "input": [reasoning_item, function_output_item],
            "instructions": "Preserve the latest decision.",
            "tools": [_FUNCTION_TOOL],
        },
    )
    output_items: Final = [
        {
            "id": f"rs_compact_{uuid.uuid4().hex}",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Compacted state"}],
            "encrypted_content": "synthetic-compact-state",
        },
        {
            "id": "fc_output_compacted",
            "type": "function_call_output",
            "call_id": "call_previous",
            "output": "The record is ready.",
            "status": "completed",
        },
    ]
    completed_id: Final = f"resp_compact_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return model_discovery_reply(_MODEL)
        assert request.method == "POST"
        assert request.headers["authorization"] == f"Bearer {_API_KEY}", dict(request.headers)
        body: Final = _JSON_OBJECT.validate_json(request.body)
        compact_params: Final = (
            {
                "model": _MODEL,
                **requests[0],
            }
            if body.get("input") == requests[0]["input"]
            else {
                "model": _MODEL,
                **requests[1],
            }
        )
        assert body == compact_params, request.body
        return Reply(
            body=json.dumps(
                {
                    "id": completed_id,
                    "created_at": 1,
                    "object": "response.compaction",
                    "output": output_items,
                    "usage": {
                        "input_tokens": 7,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 3,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 10,
                    },
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        for alias in _ALIASES:
            for request_body in requests:
                response: Final = gateway.request(
                    "POST",
                    alias,
                    {"model": model, **request_body},
                )
                assert response.status_code == 200, response.text
                compacted: Final = CompactedResponse.model_validate_json(response.content)
                assert compacted.id != completed_id
                assert same_response(compacted.id, completed_id)
                assert compacted.model_dump(mode="json") == {
                    "id": compacted.id,
                    "created_at": 1,
                    "error": None,
                    "incomplete_details": None,
                    "instructions": None,
                    "metadata": None,
                    "model": model,
                    "object": "response.compaction",
                    "output": [
                        {
                            "id": output_items[0]["id"],
                            "summary": [{"text": "Compacted state", "type": "summary_text"}],
                            "type": "reasoning",
                            "content": None,
                            "encrypted_content": "synthetic-compact-state",
                            "status": None,
                        },
                        {
                            "id": "fc_output_compacted",
                            "call_id": "call_previous",
                            "output": "The record is ready.",
                            "status": "completed",
                            "type": "function_call_output",
                            "created_by": None,
                        },
                    ],
                    "parallel_tool_calls": None,
                    "temperature": None,
                    "tool_choice": None,
                    "tools": None,
                    "top_p": None,
                    "max_output_tokens": None,
                    "previous_response_id": None,
                    "reasoning": None,
                    "status": None,
                    "text": None,
                    "truncation": None,
                    "usage": {
                        "input_tokens": 7,
                        "input_tokens_details": {
                            "audio_tokens": None,
                            "cached_tokens": 0,
                            "cached_tokens_details": None,
                            "image_tokens": None,
                            "text_tokens": None,
                            "video_tokens": None,
                        },
                        "output_tokens": 3,
                        "output_tokens_details": {
                            "audio_tokens": None,
                            "reasoning_tokens": 0,
                            "text_tokens": None,
                        },
                        "total_tokens": 10,
                        "cost": None,
                    },
                    "user": None,
                    "store": None,
                }, response.text
        assert [request.target for request in drain_contract_requests(wire)] == ["/v1/responses/compact"] * 6
