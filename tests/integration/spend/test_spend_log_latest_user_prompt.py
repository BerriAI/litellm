import json
from collections.abc import Callable
from pathlib import Path
from typing import Final
from uuid import uuid4

import pytest
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

from litellm.responses.utils import ResponsesAPIRequestUtils

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
OPENAI_MODEL: Final = "openai/gpt-4o-mini"
ANTHROPIC_MODEL: Final = "anthropic/claude-sonnet-4-5-20250929"
PIXEL: Final = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
ROW_COLUMNS: Final = "request_id, proxy_server_request, response, status, total_tokens, model"


def _latest_only_config(tmp_path: Path) -> Path:
    config: Final = tmp_path / f"latest-user-prompt-{uuid4()}.json"
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
                    "spend_logs_prompt_storage": "latest_user_message",
                    "proxy_batch_write_at": 1,
                    "proxy_batch_polling_interval": 1,
                },
            }
        )
    )
    return config


def _answering_model_listing(respond: Callable[[Request], Reply]) -> Callable[[Request], Reply]:
    def answer(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=b'{"object":"list","data":[]}')
        return respond(request)

    return answer


def _chat_completion(text: str) -> bytes:
    return json.dumps(
        {
            "id": f"chatcmpl-latest-{uuid4()}",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
        }
    ).encode()


def _anthropic_message(text: str) -> bytes:
    return json.dumps(
        {
            "id": f"msg-latest-{uuid4()}",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 11, "output_tokens": 3},
        }
    ).encode()


def _row(where: str, value: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(f'SELECT {ROW_COLUMNS} FROM "LiteLLM_SpendLogs" WHERE {where}=%s', (value,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return rows[0]


def _responses_request_id(response_id: str) -> str:
    decoded: Final = ResponsesAPIRequestUtils.decode_responses_api_response_id(response_id)
    request_id: Final = decoded.get("response_id")
    return string_value(request_id) if isinstance(request_id, str) else response_id


def _assert_usage_row(row: dict[str, JsonValue], expected_request: dict[str, JsonValue]) -> None:
    assert object_value(row["proxy_server_request"]) == expected_request, row
    assert object_value(row["response"]) == {}, row
    assert row["status"] == "success", row
    assert row["total_tokens"] == 14, row


CHAT_HISTORY: Final = [
    {"role": "system", "content": "system prompt must not be stored"},
    {"role": "user", "content": "earlier user turn must not be stored"},
    {"role": "assistant", "content": "earlier assistant turn must not be stored"},
]


@pytest.mark.parametrize(
    ("latest", "expected"),
    [
        pytest.param(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "latest chat question"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PIXEL}"}},
                ],
            },
            {"latest_user_message": "latest chat question\n[image omitted]"},
            id="user-text-and-image",
        ),
        pytest.param(
            {"role": "tool", "tool_call_id": "call_1", "content": "tool output must not be stored"},
            {},
            id="tool-result-turn",
        ),
    ],
)
def test_chat_row_keeps_only_latest_user_prompt(
    gateway: Gateway, tmp_path: Path, latest: dict[str, JsonValue], expected: dict[str, JsonValue]
) -> None:
    with (
        wire_server(_answering_model_listing(lambda _: Reply(body=_chat_completion("chat answer")))) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_latest_only_config(tmp_path)) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(model=OPENAI_MODEL, api_base=wire.url, api_key="synthetic-openai-key")
        response: Final = isolated.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [*CHAT_HISTORY, latest],
                "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
            },
        )
        assert response.status_code == 200, response.text
        _assert_usage_row(_row("request_id", string_value(JSON_OBJECT.validate_json(response.content)["id"])), expected)


@pytest.mark.parametrize(
    ("latest_content", "expected"),
    [
        pytest.param(
            [
                {"type": "text", "text": "latest messages question"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PIXEL}},
            ],
            {"latest_user_message": "latest messages question\n[image omitted]"},
            id="user-text-and-image",
        ),
        pytest.param(
            [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "tool output must not be stored"}],
            {},
            id="tool-result-turn",
        ),
    ],
)
def test_messages_row_keeps_only_latest_user_prompt(
    gateway: Gateway, tmp_path: Path, latest_content: list[JsonValue], expected: dict[str, JsonValue]
) -> None:
    with (
        wire_server(_answering_model_listing(lambda _: Reply(body=_anthropic_message("messages answer")))) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_latest_only_config(tmp_path)) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(model=ANTHROPIC_MODEL, api_base=wire.url, api_key="synthetic-anthropic-key")
        response: Final = isolated.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "system": "system prompt must not be stored",
                "messages": [
                    {"role": "user", "content": "earlier user turn must not be stored"},
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}],
                    },
                    {"role": "user", "content": latest_content},
                ],
            },
        )
        assert response.status_code == 200, response.text
        _assert_usage_row(_row("request_id", string_value(JSON_OBJECT.validate_json(response.content)["id"])), expected)


@pytest.mark.parametrize(
    ("latest", "expected"),
    [
        pytest.param(
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "latest responses question"},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{PIXEL}"},
                ],
            },
            {"latest_user_message": "latest responses question\n[image omitted]"},
            id="user-text-and-image",
        ),
        pytest.param(
            {"type": "function_call_output", "call_id": "call_1", "output": "tool output must not be stored"},
            {},
            id="tool-result-turn",
        ),
    ],
)
def test_responses_row_keeps_only_latest_user_prompt(
    gateway: Gateway, tmp_path: Path, latest: dict[str, JsonValue], expected: dict[str, JsonValue]
) -> None:
    with (
        wire_server(_answering_model_listing(lambda _: Reply(body=_anthropic_message("responses answer")))) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_latest_only_config(tmp_path)) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(model=ANTHROPIC_MODEL, api_base=wire.url, api_key="synthetic-anthropic-key")
        response: Final = isolated.request(
            "POST",
            "/v1/responses",
            {
                "model": model,
                "instructions": "system prompt must not be stored",
                "input": [
                    {"role": "user", "content": "earlier user turn must not be stored"},
                    {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
                    latest,
                ],
            },
        )
        assert response.status_code == 200, response.text
        response_id: Final = string_value(object_value(response.json())["id"])
        _assert_usage_row(_row("request_id", _responses_request_id(response_id)), expected)


def test_failed_chat_row_keeps_only_latest_user_prompt(gateway: Gateway, tmp_path: Path) -> None:
    end_user: Final = f"latest-prompt-failure-{uuid4()}"
    failure: Final = Reply(
        status=400, body=b'{"error":{"message":"scripted rejection","type":"invalid_request_error"}}'
    )
    with (
        wire_server(_answering_model_listing(lambda _: failure)) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_latest_only_config(tmp_path)) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(model=OPENAI_MODEL, api_base=wire.url, api_key="synthetic-openai-key")
        response: Final = isolated.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "user": end_user, "messages": [*CHAT_HISTORY, {"role": "user", "content": "failed ask"}]},
        )
        assert response.status_code == 400, response.text
        row: Final = _row("end_user", end_user)
        assert row["status"] == "failure", row
        assert object_value(row["proxy_server_request"]) == {"latest_user_message": "failed ask"}, row
        assert object_value(row["response"]) == {}, row
