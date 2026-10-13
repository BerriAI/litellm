import base64
import json
import uuid
from itertools import chain, count
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "gpt-5.4-mini"
_API_KEY: Final = "synthetic-openai-key"
_PROMPT: Final = "Summarize this conversation in one sentence."
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _completion(identity: str, content: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": _BACKEND,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 19, "completion_tokens": 7, "total_tokens": 26},
        }
    ).encode()


@pytest.mark.covers("providers.openai_chat_wire.tool_choice_without_tools_is_dropped_before_the_wire")
def test_openai_chat_tool_choice_without_tools_is_not_forwarded(gateway: Gateway) -> None:
    identity: Final = f"openai-toolless-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/chat/completions"
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _BACKEND
        assert body["messages"] == [{"role": "user", "content": _PROMPT}]
        assert "tool_choice" not in body, body
        assert "tools" not in body, body
        return Reply(body=_completion(identity, "One sentence."))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": _PROMPT}], "tool_choice": "none"},
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["id"] == identity
        assert payload["choices"] == [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "One sentence.",
                    "provider_specific_fields": {"refusal": None},
                },
                "provider_specific_fields": {},
            }
        ]
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]


def test_azure_gpt_6_bridged_stream_returns_text_and_tool_call_on_one_choice(gateway: Gateway) -> None:
    identity: Final = f"azure-gpt-6-sol-stream-{uuid.uuid4().hex}"
    expected_text: Final = "Let me check the weather."
    events: Final = (
        {
            "type": "response.created",
            "response": {
                "id": "resp_weather",
                "object": "response",
                "created_at": 1,
                "status": "in_progress",
                "model": "gpt-6-sol",
            },
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "msg_weather",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_weather",
            "output_index": 0,
            "content_index": 0,
            "delta": "Let me check ",
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_weather",
            "output_index": 0,
            "content_index": 0,
            "delta": "the weather.",
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "msg_weather",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": expected_text, "annotations": []}],
            },
        },
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "id": "fc_1",
                "type": "function_call",
                "status": "in_progress",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "output_index": 1,
            "delta": '{"city":',
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "output_index": 1,
            "delta": '"Paris"}',
        },
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {
                "id": "fc_1",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city":"Paris"}',
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp_weather",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-6-sol",
                "output": [
                    {
                        "id": "msg_weather",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": expected_text, "annotations": []}],
                    },
                    {
                        "id": "fc_1",
                        "type": "function_call",
                        "status": "completed",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        },
    )
    stream_chunks: Final = tuple(f"data: {json.dumps(event)}\n\n".encode() for event in events)

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/openai/responses?api-version=2025-04-01-preview"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        return Reply(content_type="text/event-stream", chunks=stream_chunks)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-6-sol",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
        )
        with gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {gateway.key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get the weather for a city.",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
                "stream": True,
                "cache": {"no-cache": True},
            },
        ) as response:
            response_body: Final = response.read()
            assert response.status_code == 200, response.text
            chunks: Final = tuple(
                _JSON_OBJECT.validate_json(line.removeprefix("data: "))
                for line in response_body.decode().splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            )
            choices: Final = tuple(chain.from_iterable(chunk["choices"] for chunk in chunks))
            assert choices, response.text
            assert all(choice["index"] == 0 for choice in choices), response.text
            assert "".join(str(choice["delta"].get("content") or "") for choice in choices) == expected_text, (
                response.text
            )
            tool_call_chunks: Final = tuple(
                chain.from_iterable(choice["delta"].get("tool_calls", []) for choice in choices)
            )
            assert (
                "".join(str(tool_call["function"].get("name") or "") for tool_call in tool_call_chunks) == "get_weather"
            ), response.text
            assert (
                "".join(str(tool_call["function"].get("arguments") or "") for tool_call in tool_call_chunks)
                == '{"city":"Paris"}'
            ), response.text
            assert tuple(
                choice.get("finish_reason") for choice in choices if choice.get("finish_reason") is not None
            ) == ("tool_calls",), response.text
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", "/openai/responses?api-version=2025-04-01-preview")
        ]


_RESPONSES_TARGET: Final = "/openai/responses?api-version=2025-04-01-preview"


def _responses_json(identity: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-6-sol",
            "output": [
                {
                    "id": f"msg_{identity}",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
    ).encode()


def _gpt_6_function_request(model: str, identity: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": f"What is the weather in Paris? {identity}"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        **extra,
    }


def test_azure_gpt_6_bridged_no_cache_function_requests_each_reach_provider_and_log_spend(
    gateway: Gateway,
) -> None:
    identity: Final = f"azure-gpt-6-sol-nocache-{uuid.uuid4().hex}"
    response_ids: Final = ("resp_first", "resp_second")
    calls: Final = count()

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == _RESPONSES_TARGET
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        assert body["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"What is the weather in Paris? {identity}"}],
            }
        ]
        return Reply(body=_responses_json(response_ids[next(calls)]))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-6-sol",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )
        request: Final = _gpt_6_function_request(model, identity, cache={"no-cache": True})
        first: Final = gateway.request("POST", "/v1/chat/completions", request)
        second: Final = gateway.request("POST", "/v1/chat/completions", request)
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert string_value(_JSON_OBJECT.validate_json(first.content)["id"]) == "resp_first", first.text
        assert string_value(_JSON_OBJECT.validate_json(second.content)["id"]) == "resp_second", second.text
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", _RESPONSES_TARGET),
            ("POST", _RESPONSES_TARGET),
        ]
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT request_id, status, cache_hit, spend FROM "LiteLLM_SpendLogs" WHERE model_group=%s',
                (model,),
            ),
            lambda found: len(found) == 2,
            seconds=70,
        )
        by_response_id: Final = {
            (decoded := base64.b64decode(string_value(row["request_id"]).removeprefix("resp_")).decode())
            .rsplit("response_id:", 1)[1]: (decoded, row)
            for row in rows
        }
        for response_id in response_ids:
            decoded, row = by_response_id[response_id]
            assert decoded.startswith("litellm:custom_llm_provider:azure;model_id:"), rows
            assert (
                string_value(row["status"]),
                string_value(row["cache_hit"]),
                float(row["spend"]),
            ) == ("success", "None", pytest.approx(10 * 0.001 + 5 * 0.002)), rows


def test_azure_gpt_6_bridged_function_requests_without_cache_field_still_hit_cache(
    gateway: Gateway,
) -> None:
    identity: Final = f"azure-gpt-6-sol-cached-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == _RESPONSES_TARGET
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "gpt-6-sol"
        return Reply(body=_responses_json("resp_cached"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure/gpt-6-sol",
            api_base=wire.url,
            api_key=_API_KEY,
            api_version="2025-04-01-preview",
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )
        request: Final = _gpt_6_function_request(model, identity)
        first: Final = gateway.request("POST", "/v1/chat/completions", request)
        second: Final = gateway.request("POST", "/v1/chat/completions", request)
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert string_value(_JSON_OBJECT.validate_json(first.content)["id"]) == "resp_cached", first.text
        assert string_value(_JSON_OBJECT.validate_json(second.content)["id"]) == "resp_cached", second.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _RESPONSES_TARGET)]
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT request_id, status, cache_hit, spend FROM "LiteLLM_SpendLogs" WHERE model_group=%s'
                " ORDER BY request_id",
                (model,),
            ),
            lambda found: len(found) == 2,
            seconds=70,
        )
        priced, cached = rows
        priced_request: Final = base64.b64decode(
            string_value(priced["request_id"]).removeprefix("resp_")
        ).decode()
        assert priced_request.startswith("litellm:custom_llm_provider:azure;model_id:"), rows
        assert priced_request.endswith(";response_id:resp_cached"), rows
        assert (
            priced["status"],
            priced["cache_hit"],
            float(priced["spend"]),
        ) == ("success", "None", pytest.approx(10 * 0.001 + 5 * 0.002)), rows
        assert string_value(cached["request_id"]).startswith("resp_cached_cache_hit"), rows
        assert (cached["status"], cached["cache_hit"], float(cached["spend"])) == ("success", "True", 0), rows
