import json
from pathlib import Path
from typing import Final
from uuid import uuid4

from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from litellm.responses.utils import ResponsesAPIRequestUtils


def _prompt_storage_config(tmp_path: Path) -> Path:
    config: Final = tmp_path / "spend-log-content.json"
    config.write_text(
        json.dumps(
            {
                "model_list": [],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "disable_responses_id_security": True,
                    "store_model_in_db": True,
                    "store_prompts_in_spend_logs": True,
                    "proxy_batch_write_at": 1,
                    "proxy_batch_polling_interval": 1,
                },
            }
        )
    )
    return config


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
        wire_server(respond) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_prompt_storage_config(tmp_path), workers=2) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="openai/gpt-4o-mini",
            api_base=wire.url,
            api_key="synthetic-openai-key",
        )
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
            },
        )
        assert response.status_code == 200, response.text
        response_id: Final = string_value(response.json()["id"])
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT proxy_server_request, response FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (response_id,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        stored_request: Final = object_value(rows[0]["proxy_server_request"])
        stored_response: Final = object_value(rows[0]["response"])
        logprob_content: Final = stored_response["choices"][0]["logprobs"]["content"]
        response_tokens: Final = [item["token"] for item in logprob_content]
        top_logprob_tokens: Final = [item["top_logprobs"][0]["token"] for item in logprob_content]
        assert response_tokens == ["sort", "_key"]
        assert top_logprob_tokens == ["sort", "_key"]
        assert stored_response["system_fingerprint"] == "REDACTED_BY_LITELM"
        assert stored_request["prompt_cache_key"] == "REDACTED_BY_LITELM"
        assert stored_request["aws_secret_access_key"] == "REDACTED_BY_LITELM"


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
        received: Final = json.loads(request.body)
        assert received["messages"][1]["content"][0]["input"] == tool_input
        return Reply(body=json.dumps(response_body).encode())

    with (
        wire_server(respond) as wire,
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
        response_id: Final = string_value(response.json()["id"])
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT proxy_server_request, response FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (response_id,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        stored_request: Final = object_value(rows[0]["proxy_server_request"])
        stored_response: Final = object_value(rows[0]["response"])
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
        assert stored_request["aws_secret_access_key"] == "REDACTED_BY_LITELM"
        observed: Final = wire.drain()
        assert len(observed) == 1
        assert json.loads(observed[0].body)["messages"][1]["content"][0]["input"] == tool_input


def test_previous_response_id_replay_sends_real_tool_payloads(gateway: Gateway, tmp_path: Path) -> None:
    function_arguments: Final = {"sort_key": "created_at", "access_level": "admin"}
    function_output: Final = {
        "status": "active",
        "token_type": "bearer",
        "partition_key": "tenant_42",
    }
    response_body: Final = {
        "id": f"msg-responses-replay-{uuid4()}",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [{"type": "text", "text": "OK"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 8, "output_tokens": 1},
    }

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        return Reply(body=json.dumps(response_body).encode())

    with (
        wire_server(respond) as wire,
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
        decoded_response_id: Final = ResponsesAPIRequestUtils._decode_responses_api_response_id(response_id)
        request_id: Final = string_value(decoded_response_id.get("response_id", response_id))
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT request_id, proxy_server_request FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (request_id,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert rows[0]["request_id"] == request_id
        stored_request: Final = object_value(rows[0]["proxy_server_request"])
        assert stored_request["input"][1]["arguments"] == function_arguments
        assert stored_request["input"][2]["output"] == function_output
        assert stored_request["aws_secret_access_key"] == "REDACTED_BY_LITELM"
        second_response: Final = isolated.request(
            "POST",
            "/v1/responses",
            {"model": model, "previous_response_id": response_id, "input": "List the values"},
        )
        assert second_response.status_code == 200, second_response.text
        observed: Final = wire.drain()
        assert len(observed) == 2
        second_request: Final = json.loads(observed[1].body)
        assert second_request["messages"][1]["role"] == "assistant"
        assert second_request["messages"][2]["role"] == "user"
        tool_use: Final = [
            block for block in second_request["messages"][1]["content"] if block.get("type") == "tool_use"
        ]
        tool_result_blocks: Final = [
            block for block in second_request["messages"][2]["content"] if block.get("type") == "tool_result"
        ]
        assert len(tool_use) == 1
        assert len(tool_result_blocks) == 1
        assert tool_use[0]["input"] == function_arguments
        replayed_output: Final = tool_result_blocks[0]["content"]
        assert isinstance(replayed_output, str)
        assert json.loads(replayed_output) == function_output
        assert "REDACTED_BY_LITELM" not in replayed_output
