import asyncio
import json
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final
from uuid import uuid4

import anthropic
import httpx
import openai
import pytest
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

from litellm.constants import LITELLM_TRUNCATED_PAYLOAD_FIELD, LITELLM_TRUNCATION_DB_SAFEGUARD_NOTE
from litellm.responses.utils import ResponsesAPIRequestUtils

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
REDACTED: Final = "REDACTED_BY_LITELM"
TOOL_INPUT: Final = {"key": "order-123", "sort_key": "created_at"}
ANTHROPIC_MODEL: Final = "anthropic/claude-sonnet-4-5-20250929"


def _prompt_storage_config(
    tmp_path: Path,
    *,
    store_prompts: bool = True,
    local_cache: bool = False,
    model_list: tuple[dict[str, JsonValue], ...] = (),
) -> Path:
    config: Final = tmp_path / f"spend-log-content-{uuid4()}.json"
    settings: Final = {"cache": True, "cache_params": {"type": "local"}} if local_cache else {}
    config.write_text(
        json.dumps(
            {
                "model_list": list(model_list),
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "disable_responses_id_security": True,
                    "store_model_in_db": True,
                    "store_prompts_in_spend_logs": store_prompts,
                    "proxy_batch_write_at": 1,
                    "proxy_batch_polling_interval": 1,
                },
                "litellm_settings": settings,
            }
        )
    )
    return config


def _json_object(body: bytes) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(body)


def _answering_model_listing(respond: Callable[[Request], Reply]) -> Callable[[Request], Reply]:
    def answer(request: Request) -> Reply:
        if request.method == "GET":
            assert request.target == "/v1/models", request.target
            return Reply(body=b'{"object":"list","data":[]}')
        return respond(request)

    return answer


def _provider_calls(requests: tuple[Request, ...]) -> tuple[Request, ...]:
    return tuple(request for request in requests if request.method != "GET" or request.target != "/v1/models")


def _objects(value: JsonValue) -> tuple[dict[str, JsonValue], ...]:
    assert isinstance(value, list)
    return tuple(object_value(item) for item in value)


def _sse_events(body: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        _json_object(line.removeprefix("data:").strip().encode())
        for line in body.splitlines()
        if line.startswith("data:") and line.removeprefix("data:").strip() != "[DONE]"
    )


def _spend_request_id(response_id: str, *, responses_api: bool = False) -> str:
    if not responses_api:
        return response_id
    decoded: Final = ResponsesAPIRequestUtils._decode_responses_api_response_id(response_id)
    request_id: Final = decoded.get("response_id")
    return string_value(request_id) if isinstance(request_id, str) else response_id


def _stored_row(response_id: str, *, responses_api: bool = False) -> dict[str, JsonValue]:
    request_id: Final = _spend_request_id(response_id, responses_api=responses_api)
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, proxy_server_request, response, status FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
            (request_id,),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    assert rows[0]["request_id"] == request_id
    return rows[0]


def _stored_cache_hit_row(response_id: str) -> dict[str, JsonValue]:
    request_id_prefix: Final = f"{response_id}_cache_hit"
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, proxy_server_request, response, status, cache_hit FROM "LiteLLM_SpendLogs" '
            "WHERE LEFT(request_id, LENGTH(%s)) = %s",
            (request_id_prefix, request_id_prefix),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    row: Final = rows[0]
    assert string_value(row["request_id"]).startswith(request_id_prefix), row
    assert row["cache_hit"] == "True", row
    assert row["proxy_server_request"] is not None, row
    assert row["response"] is not None, row
    return row


def _stored_rows(response_ids: tuple[str, ...], *, responses_api: bool = False) -> tuple[dict[str, JsonValue], ...]:
    request_ids: Final = tuple(
        _spend_request_id(response_id, responses_api=responses_api) for response_id in response_ids
    )
    placeholders: Final = ", ".join("%s" for _ in request_ids)
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, proxy_server_request, response, status FROM "LiteLLM_SpendLogs" '
            f"WHERE request_id IN ({placeholders})",
            request_ids,
        ),
        lambda values: len(values) == len(request_ids),
        seconds=70,
    )
    observed_ids: Final = tuple(string_value(row["request_id"]) for row in rows)
    assert Counter(observed_ids) == Counter(request_ids), rows
    rows_by_id: Final = {string_value(row["request_id"]): row for row in rows}
    return tuple(rows_by_id[request_id] for request_id in request_ids)


def _chat_completion(response_id: str, text: str, *, logprobs: bool = False) -> dict[str, JsonValue]:
    logprob_content: Final = [
        {
            "token": token,
            "logprob": -0.1,
            "bytes": [115],
            "top_logprobs": [{"token": token, "logprob": -0.1}],
        }
        for token in ("sort", "_key")
    ]
    choice: Final = {
        "index": 0,
        "message": {"role": "assistant", "content": text},
        "finish_reason": "stop",
        **({"logprobs": {"content": logprob_content}} if logprobs else {}),
    }
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [choice],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        "system_fingerprint": "fp_scripted",
    }


def _chat_stream(response_id: str, text: str, *, include_usage: bool = False) -> Reply:
    base: Final = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
    }
    frames: Final = (
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": text},
                    "finish_reason": None,
                    "logprobs": {
                        "content": [
                            {"token": "sort", "logprob": -0.1, "top_logprobs": [{"token": "sort", "logprob": -0.1}]}
                        ],
                    },
                }
            ],
        },
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        *(
            (
                {
                    **base,
                    "choices": [],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
            if include_usage
            else ()
        ),
    )
    chunks: Final = tuple(f"data: {json.dumps(frame)}\n\n".encode() for frame in frames) + (b"data: [DONE]\n\n",)
    return Reply(content_type="text/event-stream", chunks=chunks)


def _anthropic_message(
    response_id: str,
    text: str,
    *,
    tool_input: dict[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    content: Final = (
        [{"type": "tool_use", "id": "toolu_scripted", "name": "lookup", "input": tool_input}]
        if tool_input is not None
        else [{"type": "text", "text": text}]
    )
    return {
        "id": response_id,
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": content,
        "stop_reason": "tool_use" if tool_input is not None else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _anthropic_sse(response_id: str, text: str) -> tuple[bytes, ...]:
    events: Final = (
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": response_id,
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )
    return tuple(f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode() for event, payload in events)


def _responses_text(response_body: dict[str, JsonValue]) -> str:
    output: Final = _objects(response_body["output"])
    content: Final = _objects(output[0]["content"])
    return string_value(content[0]["text"])


def test_stored_chat_response_keeps_logprob_tokens(gateway: Gateway, tmp_path: Path) -> None:
    response_body: Final = {
        "id": f"chatcmpl-logprobs-{uuid4()}",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "sort_key"},
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {
                            "token": "sort",
                            "logprob": -0.1,
                            "bytes": [115],
                            "top_logprobs": [{"token": "sort", "logprob": -0.1}],
                        },
                        {
                            "token": "_key",
                            "logprob": -0.2,
                            "bytes": [95],
                            "top_logprobs": [{"token": "_key", "logprob": -0.2}],
                        },
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        "system_fingerprint": "fp_scripted",
    }

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        return Reply(body=json.dumps(response_body).encode())

    with (
        wire_server(_answering_model_listing(respond)) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_prompt_storage_config(tmp_path), workers=2) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="openai/gpt-4o-mini",
            api_base=wire.url,
            api_key="synthetic-openai-key",
        )
        api_key: Final = scenario.key(key_alias=f"spend-log-h1-{uuid4()}", models=[model])
        response: Final = isolated.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "logprobs": True,
                "top_logprobs": 1,
                "prompt_cache_key": "tenant-42-cache",
                "aws_secret_access_key": "AKIAEXAMPLESECRET",
                "secret_fields": {"raw_headers": {"authorization": "Bearer secret-h1"}},
                "metadata": {"user_api_key_alias": "alias-h1", "user_api_key_hash": "hash-h1"},
            },
            key=api_key,
        )
        assert response.status_code == 200, response.text
        caller_response: Final = _json_object(response.content)
        response_id: Final = string_value(caller_response["id"])
        caller_choice: Final = object_value(_objects(caller_response["choices"])[0])
        caller_message: Final = object_value(caller_choice["message"])
        assert caller_message["role"] == "assistant"
        assert caller_message["content"] == "sort_key"
        assert caller_response["system_fingerprint"] == "fp_scripted"
        upstream: Final = _provider_calls(wire.drain())
        post_requests: Final = tuple(request for request in upstream if request.method == "POST")
        assert len(post_requests) == 1
        upstream_body: Final = _json_object(post_requests[0].body)
        assert upstream_body["logprobs"] is True
        assert upstream_body["top_logprobs"] == 1
        assert upstream_body["messages"] == [{"role": "user", "content": "hi"}]
        row: Final = _stored_row(response_id)
        stored_request: Final = object_value(row["proxy_server_request"])
        stored_response: Final = object_value(row["response"])
        logprob_content: Final = stored_response["choices"][0]["logprobs"]["content"]
        response_tokens: Final = [item["token"] for item in logprob_content]
        top_logprob_tokens: Final = [item["top_logprobs"][0]["token"] for item in logprob_content]
        assert response_tokens == ["sort", "_key"]
        assert top_logprob_tokens == ["sort", "_key"]
        assert stored_response["system_fingerprint"] == REDACTED
        assert stored_request["prompt_cache_key"] == REDACTED
        assert stored_request["aws_secret_access_key"] == REDACTED
        assert "secret_fields" not in stored_request
        stored_metadata: Final = object_value(stored_request["metadata"])
        assert stored_metadata["user_api_key_alias"] == REDACTED
        assert stored_metadata["user_api_key_hash"] == REDACTED


def test_stored_messages_keep_tool_use_input(gateway: Gateway, tmp_path: Path) -> None:
    tool_input: Final = {"key": "order-123", "sort_key": "created_at"}
    tool_result: Final = [
        {"type": "tool_result", "tool_use_id": "toolu_01", "content": [{"type": "text", "text": "shipped"}]},
        {"type": "text", "text": "Now order-456"},
    ]
    response_body: Final = {
        "id": f"msg-tool-use-{uuid4()}",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_02",
                "name": "get_order",
                "input": {
                    "key": "order-456",
                    "partition_key": "tenant_42",
                    "access_level": "admin",
                    "token_type": "bearer",
                },
            }
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 8, "output_tokens": 4},
    }

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        received: Final = _json_object(request.body)
        assert received["messages"][1]["content"][0]["input"] == tool_input
        return Reply(body=json.dumps(response_body).encode())

    with (
        wire_server(_answering_model_listing(respond)) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_prompt_storage_config(tmp_path), workers=2) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-4-5-20250929",
            api_base=wire.url,
            api_key="synthetic-anthropic-key",
        )
        response: Final = isolated.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "aws_secret_access_key": "AKIAEXAMPLESECRET",
                "messages": [
                    {"role": "user", "content": "Look up order order-123."},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_01",
                                "name": "get_order",
                                "input": tool_input,
                            }
                        ],
                    },
                    {"role": "user", "content": tool_result},
                ],
            },
        )
        assert response.status_code == 200, response.text
        caller_response: Final = _json_object(response.content)
        response_id: Final = string_value(caller_response["id"])
        caller_tool_input: Final = _objects(caller_response["content"])[0]["input"]
        assert caller_tool_input == {
            "key": "order-456",
            "partition_key": "tenant_42",
            "access_level": "admin",
            "token_type": "bearer",
        }
        row: Final = _stored_row(response_id)
        stored_request: Final = object_value(row["proxy_server_request"])
        stored_response: Final = object_value(row["response"])
        assert stored_request["messages"][1]["content"][0]["input"] == tool_input
        stored_response_tool_arguments: Final = stored_response["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"
        ]
        assert json.loads(stored_response_tool_arguments) == {
            "key": "order-456",
            "partition_key": "tenant_42",
            "access_level": "admin",
            "token_type": "bearer",
        }
        assert stored_request["aws_secret_access_key"] == REDACTED
        observed: Final = _provider_calls(wire.drain())
        assert len(observed) == 1
        assert _json_object(observed[0].body)["messages"][1]["content"][0]["input"] == tool_input


def test_previous_response_id_replay_sends_real_tool_payloads(gateway: Gateway, tmp_path: Path) -> None:
    function_arguments: Final = {"sort_key": "created_at", "access_level": "admin"}
    function_output: Final = {
        "status": "active",
        "token_type": "bearer",
        "partition_key": "tenant_42",
    }

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        return Reply(body=json.dumps(_anthropic_message(f"msg-responses-replay-{uuid4()}", "OK")).encode())

    with (
        wire_server(_answering_model_listing(respond)) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_prompt_storage_config(tmp_path), workers=2) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-4-5-20250929",
            api_base=wire.url,
            api_key="synthetic-anthropic-key",
        )
        first_response: Final = isolated.request(
            "POST",
            "/v1/responses",
            {
                "model": model,
                "input": [
                    {"role": "user", "content": "Fetch my account settings."},
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_settings",
                        "arguments": function_arguments,
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": function_output,
                    },
                    {"role": "user", "content": "Acknowledge with OK"},
                ],
                "aws_secret_access_key": "AKIAEXAMPLESECRET",
            },
        )
        assert first_response.status_code == 200, first_response.text
        first_body: Final = object_value(first_response.json())
        response_id: Final = string_value(first_body["id"])
        assert _responses_text(first_body) == "OK"
        first_row: Final = _stored_row(response_id, responses_api=True)
        stored_request: Final = object_value(first_row["proxy_server_request"])
        assert stored_request["input"][1]["arguments"] == function_arguments
        assert stored_request["input"][2]["output"] == function_output
        assert stored_request["aws_secret_access_key"] == REDACTED
        second_response: Final = isolated.request(
            "POST",
            "/v1/responses",
            {"model": model, "previous_response_id": response_id, "input": "List the values"},
        )
        assert second_response.status_code == 200, second_response.text
        second_body: Final = object_value(second_response.json())
        second_response_id: Final = string_value(second_body["id"])
        assert second_response_id != response_id
        assert _responses_text(second_body) == "OK"
        second_row: Final = _stored_row(second_response_id, responses_api=True)
        second_stored_request: Final = object_value(second_row["proxy_server_request"])
        assert second_stored_request["input"] == "List the values"
        observed: Final = _provider_calls(wire.drain())
        assert len(observed) == 2
        second_request: Final = _json_object(observed[1].body)
        assert second_request["messages"][1]["role"] == "assistant"
        assert second_request["messages"][2]["role"] == "user"
        assistant_content: Final = _objects(object_value(second_request["messages"][1])["content"])
        user_content: Final = _objects(object_value(second_request["messages"][2])["content"])
        tool_use: Final = tuple(block for block in assistant_content if block.get("type") == "tool_use")
        tool_result_blocks: Final = tuple(block for block in user_content if block.get("type") == "tool_result")
        assert len(tool_use) == 1
        assert len(tool_result_blocks) == 1
        assert tool_use[0]["input"] == function_arguments
        replayed_output: Final = tool_result_blocks[0]["content"]
        assert isinstance(replayed_output, str)
        assert JSON_OBJECT.validate_json(replayed_output) == function_output
        assert REDACTED not in replayed_output


def _openai_sdk_chat_response_id(
    isolated: Gateway,
    model: str,
    *,
    client_kind: str,
    messages: list[dict[str, JsonValue]],
) -> str:
    if client_kind == "sync":
        with openai.OpenAI(
            base_url=f"{isolated.client.base_url}/v1",
            api_key=isolated.key,
            max_retries=0,
            http_client=httpx.Client(trust_env=False, timeout=30),
        ) as client:
            response: Final = client.chat.completions.create(
                model=model,
                messages=messages,
                logprobs=True,
                top_logprobs=1,
                extra_body={"prompt_cache_key": "tenant-42-cache", "aws_secret_access_key": "AKIAEXAMPLESECRET"},
            )
            assert response.choices[0].message.content == "sort_key"
            assert response.choices[0].logprobs is not None
            assert response.choices[0].logprobs.content[0].token == "sort"
            return response.id

    async def call() -> str:
        async with openai.AsyncOpenAI(
            base_url=f"{isolated.client.base_url}/v1",
            api_key=isolated.key,
            max_retries=0,
            http_client=httpx.AsyncClient(trust_env=False, timeout=30),
        ) as client:
            response: Final = await client.chat.completions.create(
                model=model,
                messages=messages,
                logprobs=True,
                top_logprobs=1,
                extra_body={"prompt_cache_key": "tenant-42-cache", "aws_secret_access_key": "AKIAEXAMPLESECRET"},
            )
            assert response.choices[0].message.content == "sort_key"
            assert response.choices[0].logprobs is not None
            assert response.choices[0].logprobs.content[0].token == "sort"
            return response.id

    return asyncio.run(call())


def _openai_sdk_stream_response_id(isolated: Gateway, model: str, messages: list[dict[str, JsonValue]]) -> str:
    async def call() -> str:
        async with openai.AsyncOpenAI(
            base_url=f"{isolated.client.base_url}/v1",
            api_key=isolated.key,
            max_retries=0,
            http_client=httpx.AsyncClient(trust_env=False, timeout=30),
        ) as client:
            stream: Final = await client.chat.completions.create(
                model=model,
                messages=messages,
                logprobs=True,
                top_logprobs=1,
                stream=True,
                stream_options={"include_usage": True},
                extra_body={"prompt_cache_key": "tenant-42-cache", "aws_secret_access_key": "AKIAEXAMPLESECRET"},
            )
            chunks: Final = [chunk async for chunk in stream]
            assert chunks[0].choices[0].delta.content == ""
            assert chunks[-1].usage is not None
            assert chunks[-1].usage.total_tokens == 2
            return chunks[0].id

    return asyncio.run(call())


@pytest.mark.parametrize("client_kind", ("sync", "async"), ids=("sync", "async"))
def test_chat_sdk_keeps_logprob_tokens(gateway: Gateway, tmp_path: Path, client_kind: str) -> None:
    response_id_from_wire: Final = f"chatcmpl-sdk-logprobs-{uuid4()}"

    def respond(request: Request) -> Reply:
        received: Final = _json_object(request.body)
        assert received["messages"] == [{"role": "user", "content": "hi"}]
        assert received["logprobs"] is True
        return Reply(body=json.dumps(_chat_completion(response_id_from_wire, "sort_key", logprobs=True)).encode())

    with (
        wire_server(_answering_model_listing(respond)) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_prompt_storage_config(tmp_path), workers=2) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="openai/gpt-4o-mini",
            api_base=wire.url,
            api_key="synthetic-openai-key",
        )
        response_id: Final = _openai_sdk_chat_response_id(
            isolated,
            model,
            client_kind=client_kind,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response_id == response_id_from_wire
        row: Final = _stored_row(response_id)
        stored_request: Final = object_value(row["proxy_server_request"])
        stored_response: Final = object_value(row["response"])
        assert [entry["token"] for entry in stored_response["choices"][0]["logprobs"]["content"]] == ["sort", "_key"]
        assert stored_request["prompt_cache_key"] == REDACTED
        assert stored_request["aws_secret_access_key"] == REDACTED
        observed: Final = _provider_calls(wire.drain())
        assert len(observed) == 1
        assert _json_object(observed[0].body)["messages"] == [{"role": "user", "content": "hi"}]


def test_chat_sdk_stream_include_usage_masks_request_fields(gateway: Gateway, tmp_path: Path) -> None:
    response_id_from_wire: Final = f"chatcmpl-sdk-stream-{uuid4()}"
    messages: Final = [
        {"role": "user", "content": "stream control"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_stream",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"sort_key":"created_at"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_stream", "content": "done"},
    ]

    def respond(request: Request) -> Reply:
        received: Final = _json_object(request.body)
        assert received["stream"] is True
        assert received["stream_options"] == {"include_usage": True}
        assert received["messages"] == messages
        return _chat_stream(response_id_from_wire, "sort_key", include_usage=True)

    with (
        wire_server(_answering_model_listing(respond)) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_prompt_storage_config(tmp_path), workers=2) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="openai/gpt-4o-mini",
            api_base=wire.url,
            api_key="synthetic-openai-key",
        )
        response_id: Final = _openai_sdk_stream_response_id(isolated, model, messages)
        row: Final = _stored_row(response_id)
        stored_request: Final = object_value(row["proxy_server_request"])
        assert stored_request["prompt_cache_key"] == REDACTED
        assert stored_request["aws_secret_access_key"] == REDACTED
        assert len(_stored_rows((response_id,))) == 1
        observed: Final = _provider_calls(wire.drain())
        post_requests: Final = tuple(request for request in observed if request.method == "POST")
        assert len(post_requests) == 1


def test_chat_history_keeps_string_tool_arguments_and_tool_content(gateway: Gateway, tmp_path: Path) -> None:
    arguments: Final = '{"sort_key":"created_at"}'
    tool_content: Final = "tool-result-created_at"
    response_id_from_wire: Final = f"chatcmpl-history-{uuid4()}"
    messages: Final = [
        {"role": "user", "content": "history control"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_history",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_history", "content": tool_content},
    ]

    def respond(request: Request) -> Reply:
        received: Final = _json_object(request.body)
        assert received["messages"] == messages
        return Reply(body=json.dumps(_chat_completion(response_id_from_wire, "done")).encode())

    with (
        wire_server(_answering_model_listing(respond)) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_prompt_storage_config(tmp_path), workers=2) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url, api_key="synthetic-openai-key")
        response: Final = isolated.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": messages},
        )
        assert response.status_code == 200, response.text
        response_id: Final = string_value(_json_object(response.content)["id"])
        row: Final = _stored_row(response_id)
        stored_request: Final = object_value(row["proxy_server_request"])
        assert stored_request["messages"][1]["tool_calls"][0]["function"]["arguments"] == arguments
        assert stored_request["messages"][2]["content"] == tool_content
        observed: Final = _provider_calls(wire.drain())
        assert len(observed) == 1
        assert _json_object(observed[0].body)["messages"] == messages


def _anthropic_sdk_message_response_id(
    isolated: Gateway,
    model: str,
    *,
    client_kind: str,
    messages: list[dict[str, JsonValue]],
    expected_input: dict[str, JsonValue],
) -> str:
    if client_kind == "sync":
        with anthropic.Anthropic(
            base_url=str(isolated.client.base_url),
            api_key=isolated.key,
            max_retries=0,
            http_client=httpx.Client(trust_env=False, timeout=30),
        ) as client:
            response: Final = client.messages.create(model=model, max_tokens=64, messages=messages)
            block: Final = response.content[0]
            assert block.type == "tool_use"
            assert block.input == expected_input
            return response.id

    async def call() -> str:
        async with anthropic.AsyncAnthropic(
            base_url=str(isolated.client.base_url),
            api_key=isolated.key,
            max_retries=0,
            http_client=httpx.AsyncClient(trust_env=False, timeout=30),
        ) as client:
            response: Final = await client.messages.create(model=model, max_tokens=64, messages=messages)
            block: Final = response.content[0]
            assert block.type == "tool_use"
            assert block.input == expected_input
            return response.id

    return asyncio.run(call())


@pytest.mark.parametrize("client_kind", ("sync", "async"), ids=("sync", "async"))
def test_messages_sdk_keeps_tool_use_input(gateway: Gateway, tmp_path: Path, client_kind: str) -> None:
    request_tool_input: Final = {"key": "order-123", "sort_key": "created_at"}
    response_tool_input: Final = {
        "key": "order-456",
        "partition_key": "tenant_42",
        "access_level": "admin",
        "token_type": "bearer",
    }
    response_id_from_wire: Final = f"msg-sdk-tool-{uuid4()}"
    messages: Final = [
        {"role": "user", "content": "SDK tool control"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_sdk", "name": "lookup", "input": request_tool_input}],
        },
    ]

    def respond(request: Request) -> Reply:
        received: Final = _json_object(request.body)
        assert received["messages"][1]["content"][0]["input"] == request_tool_input
        return Reply(
            body=json.dumps(
                _anthropic_message(response_id_from_wire, "unused", tool_input=response_tool_input)
            ).encode()
        )

    with (
        wire_server(_answering_model_listing(respond)) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_prompt_storage_config(tmp_path), workers=2) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(model=ANTHROPIC_MODEL, api_base=wire.url, api_key="synthetic-anthropic-key")
        response_id: Final = _anthropic_sdk_message_response_id(
            isolated,
            model,
            client_kind=client_kind,
            messages=messages,
            expected_input=response_tool_input,
        )
        assert response_id == response_id_from_wire
        row: Final = _stored_row(response_id)
        stored_request: Final = object_value(row["proxy_server_request"])
        stored_response: Final = object_value(row["response"])
        assert stored_request["messages"][1]["content"][0]["input"] == request_tool_input
        assert (
            JSON_OBJECT.validate_json(
                stored_response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            )
            == response_tool_input
        )
        observed: Final = _provider_calls(wire.drain())
        assert len(observed) == 1
        assert _json_object(observed[0].body)["messages"][1]["content"][0]["input"] == request_tool_input


def test_messages_stream_keeps_tool_use_input(gateway: Gateway, tmp_path: Path) -> None:
    request_tool_input: Final = {"key": "order-123", "sort_key": "created_at"}
    response_id_from_wire: Final = f"msg-stream-tool-{uuid4()}"
    messages: Final = [
        {"role": "user", "content": "stream tool control"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_stream", "name": "lookup", "input": request_tool_input}],
        },
    ]

    def respond(request: Request) -> Reply:
        received: Final = _json_object(request.body)
        assert received["stream"] is True
        assert received["messages"][1]["content"][0]["input"] == request_tool_input
        return Reply(content_type="text/event-stream", chunks=_anthropic_sse(response_id_from_wire, "streamed"))

    with (
        wire_server(_answering_model_listing(respond)) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_prompt_storage_config(tmp_path), workers=2) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(model=ANTHROPIC_MODEL, api_base=wire.url, api_key="synthetic-anthropic-key")
        async_client: Final = anthropic.AsyncAnthropic(
            base_url=str(isolated.client.base_url),
            api_key=isolated.key,
            max_retries=0,
            http_client=httpx.AsyncClient(trust_env=False, timeout=30),
        )

        async def call() -> str:
            async with async_client:
                stream: Final = await async_client.messages.create(
                    model=model,
                    max_tokens=64,
                    messages=messages,
                    stream=True,
                )
                events: Final = [event async for event in stream]
                assert events[0].type == "message_start"
                assert events[-1].type == "message_stop"
                return events[0].message.id

        response_id: Final = asyncio.run(call())
        assert response_id == response_id_from_wire
        row: Final = _stored_row(response_id)
        stored_request: Final = object_value(row["proxy_server_request"])
        assert stored_request["messages"][1]["content"][0]["input"] == request_tool_input
        observed: Final = _provider_calls(wire.drain())
        assert len(observed) == 1


def test_responses_stream_keeps_function_call_arguments(gateway: Gateway, tmp_path: Path) -> None:
    function_arguments: Final = {"sort_key": "created_at", "access_level": "admin"}
    response_id_from_wire: Final = f"msg-responses-stream-{uuid4()}"

    def respond(request: Request) -> Reply:
        assert request.target == "/v1/messages"
        received: Final = _json_object(request.body)
        assert received["stream"] is True
        assert "created_at" in request.body.decode()
        return Reply(content_type="text/event-stream", chunks=_anthropic_sse(response_id_from_wire, "streamed"))

    with (
        wire_server(_answering_model_listing(respond)) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_prompt_storage_config(tmp_path), workers=2) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(model=ANTHROPIC_MODEL, api_base=wire.url, api_key="synthetic-anthropic-key")
        response: Final = isolated.request(
            "POST",
            "/v1/responses",
            {
                "model": model,
                "stream": True,
                "input": [
                    {"role": "user", "content": "stream response control"},
                    {
                        "type": "function_call",
                        "call_id": "call_stream",
                        "name": "lookup",
                        "arguments": function_arguments,
                    },
                ],
            },
        )
        assert response.status_code == 200, response.text
        events: Final = _sse_events(response.text)
        completed_event: Final = next(event for event in events if event["type"] == "response.completed")
        completed_response: Final = object_value(completed_event["response"])
        response_id: Final = string_value(completed_response["id"])
        assert _responses_text(completed_response) == "streamed"
        row: Final = _stored_row(response_id, responses_api=True)
        stored_request: Final = object_value(row["proxy_server_request"])
        assert stored_request["input"][1]["arguments"] == function_arguments
        observed: Final = _provider_calls(wire.drain())
        assert len(observed) == 1
        assert "created_at" in observed[0].body.decode()


def test_native_responses_keeps_logprob_tokens(gateway: Gateway, tmp_path: Path) -> None:
    response_id_from_wire: Final = f"resp-native-logprobs-{uuid4()}"
    response_body: Final = {
        "id": response_id_from_wire,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "prompt_cache_key": "tenant-42",
        "output": [
            {
                "type": "message",
                "id": f"msg-native-{uuid4()}",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "sort",
                        "annotations": [],
                        "logprobs": [
                            {
                                "token": "sort",
                                "logprob": -0.1,
                                "bytes": [115],
                                "top_logprobs": [{"token": "sort", "logprob": -0.1}],
                            }
                        ],
                    }
                ],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }

    def respond(request: Request) -> Reply:
        assert request.target.endswith("/responses"), request.target
        assert not request.target.endswith("/chat/completions"), request.target
        received: Final = _json_object(request.body)
        assert received["include"] == ["message.output_text.logprobs"]
        assert received["top_logprobs"] == 1
        assert received["prompt_cache_key"] == "tenant-42"
        return Reply(body=json.dumps(response_body).encode())

    with (
        wire_server(_answering_model_listing(respond)) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_prompt_storage_config(tmp_path), workers=2) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url, api_key="synthetic-openai-key")
        response: Final = isolated.request(
            "POST",
            "/v1/responses",
            {
                "model": model,
                "input": "native response logprob control",
                "include": ["message.output_text.logprobs"],
                "top_logprobs": 1,
                "prompt_cache_key": "tenant-42",
            },
        )
        assert response.status_code == 200, response.text
        caller_body: Final = object_value(response.json())
        response_id: Final = string_value(caller_body["id"])
        assert _responses_text(caller_body) == "sort"
        row: Final = _stored_row(response_id)
        stored_request: Final = object_value(row["proxy_server_request"])
        stored_response: Final = object_value(row["response"])
        stored_output: Final = object_value(_objects(stored_response["output"])[0])
        stored_content: Final = object_value(_objects(stored_output["content"])[0])
        stored_logprobs: Final = _objects(stored_content["logprobs"])
        assert stored_logprobs[0]["token"] == "sort"
        assert stored_request["prompt_cache_key"] == REDACTED
        assert stored_response["prompt_cache_key"] == REDACTED
        observed: Final = _provider_calls(wire.drain())
        assert len(observed) == 1
        assert observed[0].target.endswith("/responses"), observed[0].target


def test_malformed_tool_blocks_keep_only_recognized_content(gateway: Gateway, tmp_path: Path) -> None:
    extra_blocks: Final = [
        {"type": "tool_use", "api_key": "sk-sibling-secret", "input": {"sort_key": "created_at"}},
        {"type": {"bad": 1}, "input": {"api_key": "sk-malformed-secret"}},
    ]
    response_id_from_wire: Final = f"chat-extra-blocks-{uuid4()}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        return Reply(body=json.dumps(_chat_completion(response_id_from_wire, "done")).encode())

    with (
        wire_server(_answering_model_listing(respond)) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_prompt_storage_config(tmp_path), workers=2) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url, api_key="synthetic-openai-key")
        response: Final = isolated.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "extra blocks control"}],
                "extra_blocks": extra_blocks,
            },
        )
        assert response.status_code == 200, response.text
        response_id: Final = string_value(_json_object(response.content)["id"])
        row: Final = _stored_row(response_id)
        stored_request: Final = object_value(row["proxy_server_request"])
        stored_blocks: Final = _objects(stored_request["extra_blocks"])
        assert stored_blocks[0]["api_key"] == REDACTED
        assert object_value(stored_blocks[1]["input"])["api_key"] == REDACTED
        assert object_value(stored_blocks[0]["input"])["sort_key"] == "created_at"
        observed: Final = _provider_calls(wire.drain())
        assert len(observed) == 1
        assert _json_object(observed[0].body)["messages"] == [{"role": "user", "content": "extra blocks control"}]


def test_messages_tool_input_handles_mixed_values_and_truncation(gateway: Gateway, tmp_path: Path) -> None:
    long_partition_key: Final = "x" * 5000
    tool_input: Final = {
        "sort_key": 7,
        "access_level": ["admin"],
        "token_type": "",
        "partition_key": long_partition_key,
        "key": "dup",
        "sort_key_copy": "dup",
    }
    response_id_from_wire: Final = f"msg-mixed-tool-input-{uuid4()}"

    def respond(request: Request) -> Reply:
        received: Final = _json_object(request.body)
        assert received["messages"][1]["content"][0]["input"] == tool_input
        return Reply(body=json.dumps(_anthropic_message(response_id_from_wire, "done")).encode())

    with (
        wire_server(_answering_model_listing(respond)) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_prompt_storage_config(tmp_path), workers=2) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(model=ANTHROPIC_MODEL, api_base=wire.url, api_key="synthetic-anthropic-key")
        response: Final = isolated.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": "mixed tool control"},
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "toolu_mixed", "name": "lookup", "input": tool_input}],
                    },
                ],
            },
        )
        assert response.status_code == 200, response.text
        response_id: Final = string_value(_json_object(response.content)["id"])
        row: Final = _stored_row(response_id)
        stored_request: Final = object_value(row["proxy_server_request"])
        stored_tool_input: Final = _objects(object_value(stored_request["messages"][1])["content"])[0]["input"]
        stored_tool_input_object: Final = object_value(stored_tool_input)
        assert stored_tool_input_object["sort_key"] == 7
        assert stored_tool_input_object["access_level"] == ["admin"]
        assert stored_tool_input_object["token_type"] == ""
        assert stored_tool_input_object["key"] == "dup"
        assert stored_tool_input_object["sort_key_copy"] == "dup"
        partition_key: Final = string_value(stored_tool_input_object["partition_key"])
        assert REDACTED not in partition_key
        assert LITELLM_TRUNCATED_PAYLOAD_FIELD in partition_key
        assert LITELLM_TRUNCATION_DB_SAFEGUARD_NOTE in partition_key
        observed: Final = _provider_calls(wire.drain())
        assert len(observed) == 1


def test_messages_without_auth_create_no_spend_row(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = f"unauthenticated-spend-marker-{uuid4()}"

    def respond(_: Request) -> Reply:
        raise AssertionError("Unauthenticated requests must not reach the upstream")

    with (
        wire_server(respond) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_prompt_storage_config(tmp_path), workers=2) as isolated,
    ):
        response: Final = isolated.client.post(
            "/v1/messages",
            json={
                "model": "missing-model",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": marker}],
            },
        )
        assert response.status_code == 401, response.text
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE proxy_server_request::text LIKE %s',
                (f"%{marker}%",),
            ),
            lambda values: bool(values),
            seconds=1,
            return_last_on_timeout=True,
        )
        assert rows == []
        assert wire.drain() == ()


def test_messages_upstream_error_keeps_tool_input(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = f"upstream-error-{uuid4()}"
    tool_input: Final = {"key": marker, "sort_key": "created_at"}
    error_body: Final = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "synthetic upstream error"},
    }

    def respond(request: Request) -> Reply:
        received: Final = _json_object(request.body)
        assert received["messages"][1]["content"][0]["input"] == tool_input
        return Reply(status=400, content_type="application/json", body=json.dumps(error_body).encode())

    with (
        wire_server(_answering_model_listing(respond)) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_prompt_storage_config(tmp_path), workers=2) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(model=ANTHROPIC_MODEL, api_base=wire.url, api_key="synthetic-anthropic-key")
        response: Final = isolated.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": marker},
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "toolu_error", "name": "lookup", "input": tool_input}],
                    },
                ],
            },
        )
        assert 400 <= response.status_code < 500, response.text
        assert response.status_code != 500
        assert "synthetic upstream error" in response.text
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT proxy_server_request, status FROM "LiteLLM_SpendLogs" WHERE proxy_server_request::text LIKE %s',
                (f"%{marker}%",),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        stored_request: Final = object_value(rows[0]["proxy_server_request"])
        assert object_value(stored_request["messages"][1])["content"][0]["input"] == tool_input
        assert rows[0]["status"] == "failure"
        observed: Final = _provider_calls(wire.drain())
        assert len(observed) == 1


def test_store_prompts_off_keeps_chat_and_messages_representation_equal(gateway: Gateway, tmp_path: Path) -> None:
    chat_response_id: Final = f"chat-store-off-{uuid4()}"
    messages_response_id: Final = f"msg-store-off-{uuid4()}"

    def respond(request: Request) -> Reply:
        if request.target.endswith("/chat/completions"):
            return Reply(body=json.dumps(_chat_completion(chat_response_id, "chat")).encode())
        return Reply(body=json.dumps(_anthropic_message(messages_response_id, "messages")).encode())

    with (
        wire_server(_answering_model_listing(respond)) as wire,
        owned_proxy(
            gateway,
            tmp_path,
            {},
            config=_prompt_storage_config(tmp_path, store_prompts=False),
            workers=2,
        ) as isolated,
        isolated.scenario() as scenario,
    ):
        chat_model: Final = scenario.model(
            model="openai/gpt-4o-mini", api_base=wire.url, api_key="synthetic-openai-key"
        )
        messages_model: Final = scenario.model(
            model=ANTHROPIC_MODEL, api_base=wire.url, api_key="synthetic-anthropic-key"
        )
        chat_response: Final = isolated.request(
            "POST",
            "/v1/chat/completions",
            {"model": chat_model, "messages": [{"role": "user", "content": "store off chat"}]},
        )
        messages_response: Final = isolated.request(
            "POST",
            "/v1/messages",
            {
                "model": messages_model,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "store off messages"}],
            },
        )
        assert chat_response.status_code == 200, chat_response.text
        assert messages_response.status_code == 200, messages_response.text
        chat_id: Final = string_value(_json_object(chat_response.content)["id"])
        messages_id: Final = string_value(_json_object(messages_response.content)["id"])
        chat_row: Final = _stored_row(chat_id)
        messages_row: Final = _stored_row(messages_id)
        assert object_value(chat_row["proxy_server_request"]) == {}
        assert object_value(messages_row["proxy_server_request"]) == {}
        assert len(_provider_calls(wire.drain())) == 2


def test_identical_messages_requests_have_distinct_spend_rows(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = f"identical-messages-{uuid4()}"
    request_body: Final = {
        "model": "",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": marker}],
    }

    def respond(_: Request) -> Reply:
        return Reply(body=json.dumps(_anthropic_message(f"msg-identical-{uuid4()}", "same")).encode())

    with (
        wire_server(_answering_model_listing(respond)) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_prompt_storage_config(tmp_path), workers=2) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(model=ANTHROPIC_MODEL, api_base=wire.url, api_key="synthetic-anthropic-key")
        body: Final = {**request_body, "model": model}
        response_ids: Final = tuple(
            string_value(_json_object(isolated.request("POST", "/v1/messages", body).content)["id"]) for _ in range(3)
        )
        assert len(set(response_ids)) == 3
        assert len(_stored_rows(response_ids)) == 3
        assert len(_provider_calls(wire.drain())) == 3


def test_chat_cache_hit_keeps_logprob_tokens(gateway: Gateway, tmp_path: Path) -> None:
    def respond(_: Request) -> Reply:
        return Reply(body=json.dumps(_chat_completion(f"chat-cache-{uuid4()}", "sort_key", logprobs=True)).encode())

    with (
        wire_server(_answering_model_listing(respond)) as wire,
        owned_proxy(
            gateway,
            tmp_path,
            {},
            config=_prompt_storage_config(tmp_path, local_cache=True),
            workers=2,
        ) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url, api_key="synthetic-openai-key")
        body: Final = {
            "model": model,
            "messages": [{"role": "user", "content": "cache logprob control"}],
            "logprobs": True,
            "top_logprobs": 1,
            "prompt_cache_key": "tenant-42-cache",
            "aws_secret_access_key": "AKIAEXAMPLESECRET",
            "secret_fields": {"raw_headers": {"authorization": "Bearer secret-cache"}},
        }
        first: Final = isolated.request("POST", "/v1/chat/completions", body)
        second: Final = isolated.request("POST", "/v1/chat/completions", body)
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        first_id: Final = string_value(_json_object(first.content)["id"])
        second_id: Final = string_value(_json_object(second.content)["id"])
        second_row: Final = _stored_cache_hit_row(second_id)
        stored_request: Final = object_value(second_row["proxy_server_request"])
        stored_response: Final = object_value(second_row["response"])
        assert [entry["token"] for entry in stored_response["choices"][0]["logprobs"]["content"]] == ["sort", "_key"]
        assert stored_request["prompt_cache_key"] == REDACTED
        assert stored_request["aws_secret_access_key"] == REDACTED
        assert "secret_fields" not in stored_request
        assert stored_response["system_fingerprint"] == REDACTED
        assert len(_provider_calls(wire.drain())) == 1
        assert len(_stored_rows((first_id,))) == 1


def test_messages_cache_hit_keeps_tool_input(gateway: Gateway, tmp_path: Path) -> None:
    tool_input: Final = {"key": "cache-order", "sort_key": "created_at"}

    def respond(_: Request) -> Reply:
        return Reply(
            body=json.dumps(_anthropic_message(f"msg-cache-{uuid4()}", "done", tool_input=tool_input)).encode()
        )

    with (
        wire_server(_answering_model_listing(respond)) as wire,
        owned_proxy(
            gateway,
            tmp_path,
            {},
            config=_prompt_storage_config(tmp_path, local_cache=True),
            workers=2,
        ) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(model=ANTHROPIC_MODEL, api_base=wire.url, api_key="synthetic-anthropic-key")
        body: Final = {
            "model": model,
            "max_tokens": 64,
            "aws_secret_access_key": "AKIAEXAMPLESECRET",
            "secret_fields": {"raw_headers": {"authorization": "Bearer secret-cache"}},
            "messages": [
                {"role": "user", "content": "cache tool control"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_cache", "name": "lookup", "input": tool_input}],
                },
            ],
        }
        first: Final = isolated.request("POST", "/v1/messages", body)
        second: Final = isolated.request("POST", "/v1/messages", body)
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        first_id: Final = string_value(_json_object(first.content)["id"])
        second_id: Final = string_value(_json_object(second.content)["id"])
        second_row: Final = _stored_cache_hit_row(second_id)
        stored_request: Final = object_value(second_row["proxy_server_request"])
        assert stored_request["messages"][1]["content"][0]["input"] == tool_input
        assert stored_request["aws_secret_access_key"] == REDACTED
        assert "secret_fields" not in stored_request
        assert len(_provider_calls(wire.drain())) == 1
        assert len(_stored_rows((first_id,))) == 1


def _burst_case(
    index: int,
    chat_model: str,
    messages_model: str,
    *,
    prefix: str,
) -> tuple[str, str, dict[str, JsonValue]]:
    marker: Final = f"{prefix}-{uuid4()}"
    match index % 5:
        case 0:
            return (
                "chat_nonstream",
                marker,
                {
                    "model": chat_model,
                    "messages": [{"role": "user", "content": marker}],
                    "logprobs": True,
                    "top_logprobs": 1,
                },
            )
        case 1:
            return (
                "chat_stream",
                marker,
                {
                    "model": chat_model,
                    "messages": [{"role": "user", "content": marker}],
                    "logprobs": True,
                    "top_logprobs": 1,
                    "stream": True,
                },
            )
        case 2:
            return (
                "messages_nonstream",
                marker,
                {
                    "model": messages_model,
                    "max_tokens": 16,
                    "messages": [
                        {"role": "user", "content": marker},
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "toolu_burst",
                                    "name": "lookup",
                                    "input": {"sort_key": marker},
                                }
                            ],
                        },
                    ],
                },
            )
        case 3:
            return (
                "messages_stream",
                marker,
                {
                    "model": messages_model,
                    "max_tokens": 16,
                    "messages": [
                        {"role": "user", "content": marker},
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "toolu_burst_stream",
                                    "name": "lookup",
                                    "input": {"sort_key": marker},
                                }
                            ],
                        },
                    ],
                    "stream": True,
                },
            )
        case _:
            return (
                "responses_stream" if index % 2 else "responses_nonstream",
                marker,
                {
                    "model": messages_model,
                    "input": [
                        {"role": "user", "content": marker},
                        {
                            "type": "function_call",
                            "call_id": "call_burst",
                            "name": "lookup",
                            "arguments": {"sort_key": marker},
                        },
                    ],
                    **({"stream": True} if index % 2 else {}),
                },
            )


def _burst_marker(request: Request, prefix: str) -> str:
    tokens: Final = request.body.decode().replace('"', " ").replace(",", " ").split()
    marker: Final = next(
        (token.strip("[]{}:,") for token in tokens if token.startswith(prefix)),
        None,
    )
    assert marker is not None, f"No {prefix} marker in {request.target}: {request.body.decode()}"
    return marker


def _burst_model_list(chat_model: str, messages_model: str, api_base: str) -> tuple[dict[str, JsonValue], ...]:
    return (
        {
            "model_name": chat_model,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_base": api_base,
                "api_key": "synthetic-openai-key",
            },
        },
        {
            "model_name": messages_model,
            "litellm_params": {
                "model": ANTHROPIC_MODEL,
                "api_base": api_base,
                "api_key": "synthetic-anthropic-key",
            },
        },
    )


def _assert_burst_upstream(
    requests: tuple[Request, ...],
    prefix: str,
    markers: tuple[str, ...],
    expected_posts: int,
) -> None:
    assert len(requests) == expected_posts
    assert all(request.method == "POST" for request in requests)
    assert Counter(_burst_marker(request, prefix) for request in requests) == Counter(markers)


def _burst_response_id(response: httpx.Response, kind: str, marker: str) -> str:
    assert response.status_code == 200, f"{kind} {marker}: {response.text}"
    if kind.endswith("_nonstream"):
        body: Final = _json_object(response.content)
        if kind == "chat_nonstream":
            assert object_value(_objects(body["choices"])[0])["message"]["content"] == marker
        elif kind == "messages_nonstream":
            assert _objects(body["content"])[0]["text"] == marker
        else:
            assert _responses_text(body) == marker
        return string_value(body["id"])
    events: Final = _sse_events(response.text)
    if kind == "chat_stream":
        assert marker in response.text
        return string_value(events[0]["id"])
    if kind == "messages_stream":
        assert marker in response.text
        return string_value(object_value(events[0]["message"])["id"])
    completed: Final = next(event for event in events if event["type"] == "response.completed")
    completed_response: Final = object_value(completed["response"])
    assert _responses_text(completed_response) == marker
    return string_value(completed_response["id"])


def _burst_endpoint(kind: str) -> str:
    match kind:
        case "chat_nonstream" | "chat_stream":
            return "/v1/chat/completions"
        case "messages_nonstream" | "messages_stream":
            return "/v1/messages"
        case "responses_nonstream" | "responses_stream":
            return "/v1/responses"
        case _:
            raise AssertionError(f"Unknown burst request kind: {kind}")


async def _send_burst(
    isolated: Gateway,
    cases: tuple[tuple[str, str, dict[str, JsonValue]], ...],
) -> tuple[httpx.Response, ...]:
    async with httpx.AsyncClient(
        base_url=str(isolated.client.base_url),
        headers={"Authorization": f"Bearer {isolated.key}"},
        timeout=180,
        trust_env=False,
    ) as client:
        return tuple(await asyncio.gather(*(client.post(_burst_endpoint(kind), json=body) for kind, _, body in cases)))


def _assert_burst_row(
    row: dict[str, JsonValue],
    kind: str,
    marker: str,
    *,
    require_chat_logprobs: bool = True,
) -> None:
    stored_request: Final = object_value(row["proxy_server_request"])
    stored_response: Final = object_value(row["response"])
    assert marker in json.dumps(stored_request)
    assert marker in json.dumps(stored_response)
    if kind == "chat_nonstream" and require_chat_logprobs:
        choice: Final = _objects(stored_response["choices"])[0]
        logprobs: Final = _objects(object_value(choice["logprobs"])["content"])[0]
        assert logprobs["token"] == "sort"
    if kind == "chat_stream":
        choice: Final = _objects(stored_response["choices"])[0]
        assert object_value(choice["message"])["content"] == marker


@pytest.mark.timeout(240)
def test_concurrent_mixed_requests_land_once(gateway: Gateway, tmp_path: Path) -> None:
    def respond(request: Request) -> Reply:
        marker: Final = _burst_marker(request, "audit-x1")
        response_id: Final = f"{'chatcmpl' if request.target.endswith('/chat/completions') else 'msg'}-{uuid4()}"
        body: Final = _json_object(request.body)
        if request.target.endswith("/chat/completions"):
            if body.get("stream") is True:
                return _chat_stream(response_id, marker)
            return Reply(body=json.dumps(_chat_completion(response_id, marker, logprobs=True)).encode())
        if body.get("stream") is True:
            return Reply(content_type="text/event-stream", chunks=_anthropic_sse(response_id, marker))
        return Reply(body=json.dumps(_anthropic_message(response_id, marker)).encode())

    chat_model: Final = f"integration-x1-chat-{uuid4().hex}"
    messages_model: Final = f"integration-x1-messages-{uuid4().hex}"
    with (
        wire_server(_answering_model_listing(respond)) as wire,
        owned_proxy(
            gateway,
            tmp_path,
            {},
            config=_prompt_storage_config(
                tmp_path,
                model_list=_burst_model_list(chat_model, messages_model, wire.url),
            ),
            workers=2,
        ) as isolated,
    ):
        cases: Final = tuple(_burst_case(index, chat_model, messages_model, prefix="audit-x1") for index in range(30))
        responses: Final = asyncio.run(_send_burst(isolated, cases))
        response_ids: Final = tuple(
            _burst_response_id(response, kind, marker) for response, (kind, marker, _) in zip(responses, cases)
        )
        assert len(set(response_ids)) == len(response_ids)
        rows: Final = tuple(
            _stored_row(response_id, responses_api=kind.startswith("responses"))
            for response_id, (kind, _, _) in zip(response_ids, cases)
        )
        for row, (kind, marker, _) in zip(rows, cases):
            _assert_burst_row(row, kind, marker)
        _assert_burst_upstream(
            _provider_calls(wire.drain()),
            "audit-x1",
            tuple(marker for _, marker, _ in cases),
            30,
        )


@pytest.mark.timeout(240)
def test_slow_upstream_burst_lands_once(gateway: Gateway, tmp_path: Path) -> None:
    def respond(request: Request) -> Reply:
        time.sleep(1)
        marker: Final = _burst_marker(request, "audit-x2")
        response_id: Final = f"{'chatcmpl' if request.target.endswith('/chat/completions') else 'msg'}-{uuid4()}"
        body: Final = _json_object(request.body)
        if request.target.endswith("/chat/completions"):
            if body.get("stream") is True:
                return _chat_stream(response_id, marker)
            return Reply(body=json.dumps(_chat_completion(response_id, marker, logprobs=True)).encode())
        if body.get("stream") is True:
            return Reply(content_type="text/event-stream", chunks=_anthropic_sse(response_id, marker))
        return Reply(body=json.dumps(_anthropic_message(response_id, marker)).encode())

    chat_model: Final = f"integration-x2-chat-{uuid4().hex}"
    messages_model: Final = f"integration-x2-messages-{uuid4().hex}"
    with (
        wire_server(_answering_model_listing(respond)) as wire,
        owned_proxy(
            gateway,
            tmp_path,
            {},
            config=_prompt_storage_config(
                tmp_path,
                model_list=_burst_model_list(chat_model, messages_model, wire.url),
            ),
            workers=2,
        ) as isolated,
    ):
        cases: Final = tuple(_burst_case(index, chat_model, messages_model, prefix="audit-x2") for index in range(15))
        responses: Final = asyncio.run(_send_burst(isolated, cases))
        response_ids: Final = tuple(
            _burst_response_id(response, kind, marker) for response, (kind, marker, _) in zip(responses, cases)
        )
        assert len(set(response_ids)) == len(response_ids)
        rows: Final = tuple(
            _stored_row(response_id, responses_api=kind.startswith("responses"))
            for response_id, (kind, _, _) in zip(response_ids, cases)
        )
        for row, (kind, marker, _) in zip(rows, cases):
            _assert_burst_row(row, kind, marker)
        _assert_burst_upstream(
            _provider_calls(wire.drain()),
            "audit-x2",
            tuple(marker for _, marker, _ in cases),
            15,
        )


@pytest.mark.timeout(240)
def test_upstream_stop_returns_errors_and_recovers(gateway: Gateway, tmp_path: Path) -> None:
    release: Final = threading.Event()

    def stopped_respond(_: Request) -> Reply:
        release.wait(timeout=5)
        return Reply(status=503, content_type="application/json", body=b'{"error":"synthetic upstream stopped"}')

    with (
        owned_proxy(gateway, tmp_path, {}, config=_prompt_storage_config(tmp_path), workers=2) as isolated,
        isolated.scenario() as scenario,
    ):
        with ThreadPoolExecutor(max_workers=1) as executor:
            with wire_server(_answering_model_listing(stopped_respond)) as wire:
                failed_model: Final = scenario.model(
                    model="openai/gpt-4o-mini",
                    api_base=wire.url,
                    api_key="synthetic-openai-key",
                )
                failed_cases: Final = tuple(
                    (
                        "chat_nonstream",
                        marker,
                        {
                            "model": failed_model,
                            "messages": [{"role": "user", "content": marker}],
                        },
                    )
                    for marker in (f"audit-x3-{uuid4()}" for _ in range(10))
                )
                future: Final = executor.submit(asyncio.run, _send_burst(isolated, failed_cases))
                arrived_provider_calls: Final = eventually(
                    lambda: _provider_calls(wire.drain()),
                    lambda requests: len(requests) >= 1,
                    seconds=20,
                )
                release.set()
            failed_upstream: Final = (*arrived_provider_calls, *_provider_calls(wire.drain()))
            assert failed_upstream
            failed_responses: Final = future.result(timeout=60)
            assert all(response.status_code >= 400 and response.text for response in failed_responses)
            health: Final = isolated.request("GET", "/health/liveliness")
            assert health.status_code == 200, health.text

            def recovered_respond(request: Request) -> Reply:
                marker: Final = _burst_marker(request, "audit-x3-recovery")
                return Reply(body=json.dumps(_chat_completion(f"chatcmpl-{uuid4()}", marker)).encode())

            with wire_server(_answering_model_listing(recovered_respond)) as recovered_wire:
                recovered_model: Final = scenario.model(
                    model="openai/gpt-4o-mini",
                    api_base=recovered_wire.url,
                    api_key="synthetic-openai-key",
                )
                recovery_markers: Final = tuple(f"audit-x3-recovery-{uuid4()}" for _ in range(5))
                recovered_cases: Final = tuple(
                    (
                        "chat_nonstream",
                        marker,
                        {
                            "model": recovered_model,
                            "messages": [{"role": "user", "content": marker}],
                        },
                    )
                    for marker in recovery_markers
                )
                recovered_responses: Final = asyncio.run(_send_burst(isolated, recovered_cases))
                recovered_ids: Final = tuple(
                    _burst_response_id(response, kind, marker)
                    for response, (kind, marker, _) in zip(recovered_responses, recovered_cases)
                )
                assert len(set(recovered_ids)) == 5
                recovered_rows: Final = _stored_rows(recovered_ids)
                for row, (_, marker, _) in zip(recovered_rows, recovered_cases):
                    _assert_burst_row(row, "chat_nonstream", marker, require_chat_logprobs=False)
                assert len(_provider_calls(recovered_wire.drain())) == 5
