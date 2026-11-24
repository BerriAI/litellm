import json
from collections.abc import Iterator
from typing import Final
from uuid import uuid4

import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment
from integration._support.device_login import write_copilot_access
from integration._support.process import OwnedState, owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter
from typing_extensions import NotRequired, ReadOnly, TypedDict

pytestmark: Final = pytest.mark.timeout(180)

_MODEL: Final = "github_copilot/gpt-4.1"
_RESPONSES_MODEL: Final = "github_copilot/gpt-5.4-nano"
_PATTERN: Final = r'^(?!__.*__$)[^\p{Cc}\p{Cf}\p{Zl}\p{Zp}"\\./[\]]{1,200}$'
_LOOKAHEAD: Final = r"^(?!__)[a-z]+$"


class _Parameters(TypedDict):
    type: ReadOnly[str]
    properties: ReadOnly[dict[str, dict[str, JsonValue]]]
    required: ReadOnly[list[str]]


class _Function(TypedDict):
    name: ReadOnly[str]
    description: ReadOnly[str]
    parameters: ReadOnly[_Parameters]


class _Tool(TypedDict):
    type: ReadOnly[str]
    function: ReadOnly[_Function]


class _ChatRequest(TypedDict):
    tools: ReadOnly[list[_Tool]]
    stream: NotRequired[ReadOnly[bool]]


_CHAT_REQUEST: Final = TypeAdapter(_ChatRequest)
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _artifact(name_schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "type": "function",
        "function": {
            "name": "Artifact",
            "description": "Create an artifact",
            "parameters": _parameters(name_schema),
        },
    }


def _parameters(name_schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "type": "object",
        "properties": {"name": name_schema, "slug": {"type": "string", "pattern": _LOOKAHEAD}},
        "required": ["name"],
    }


def _copilot_chat(request: Request) -> Reply:
    if request.target == "/responses":
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["tools"] == [
            {
                "type": "function",
                "name": "Artifact",
                "description": "Create an artifact",
                "parameters": _parameters({"type": "string", "pattern": _PATTERN}),
            }
        ]
        return Reply(
            body=json.dumps(
                {
                    "id": f"resp_{uuid4().hex}",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "gpt-5.4-nano",
                    "output": [],
                    "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
                }
            ).encode()
        )
    assert request.target == "/chat/completions"
    chat: Final = _CHAT_REQUEST.validate_json(request.body)
    tools: Final = chat["tools"]
    pattern: Final = tools[0]["function"]["parameters"]["properties"]["name"].get("pattern")
    if isinstance(pattern, str) and r"\p{" in pattern:
        return Reply(
            status=400,
            body=json.dumps(
                {
                    "error": {
                        "message": f"Invalid schema for function 'Artifact': {pattern!r} is not a 'regex'.",
                        "code": "invalid_function_parameters",
                    }
                }
            ).encode(),
        )
    assert tools == [_artifact({"type": "string"})]
    completion_id: Final = f"chatcmpl-{uuid4().hex}"
    if chat.get("stream"):
        chunks: Final = tuple(
            b"data: "
            + json.dumps(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-4.1",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                }
            ).encode()
            + b"\n\n"
            for delta, finish_reason in (({"role": "assistant", "content": "Hello"}, None), ({}, "stop"))
        )
        return Reply(content_type="text/event-stream", chunks=(*chunks, b"data: [DONE]\n\n"))
    return Reply(
        body=json.dumps(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4.1",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            }
        ).encode()
    )


@pytest.fixture(scope="module")
def copilot_chat(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("copilot-chat")
    with gateway_from_environment() as gateway, wire_server(_copilot_chat) as upstream:
        write_copilot_access(directory, "scripted-copilot-access")
        config: Final = directory / "config.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [
                        {"model_name": model, "litellm_params": {"model": model, "api_base": upstream.url}}
                        for model in (_MODEL, _RESPONSES_MODEL)
                    ],
                    "router_settings": {"num_retries": 0},
                }
            )
        )
        with owned_proxy(
            gateway,
            directory,
            {"GITHUB_COPILOT_TOKEN_DIR": str(directory)},
            config=config,
            workers=2,
            state=OwnedState.PRIVATE,
        ) as proxy:
            yield proxy


def test_copilot_chat_keeps_artifact_tool_after_relaxing_unicode_pattern(copilot_chat: Gateway) -> None:
    response: Final = copilot_chat.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": _MODEL,
            "messages": [{"role": "user", "content": "Say hello"}],
            "tools": [_artifact({"type": "string", "pattern": _PATTERN})],
            "num_retries": 0,
        },
    )
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    ("route", "stream", "terminator"),
    [
        pytest.param("/v1/chat/completions", True, "data: [DONE]", id="chat-stream"),
        pytest.param("/v1/messages", False, '"type":"message"', id="messages-unary"),
        pytest.param("/v1/messages", True, "event: message_stop", id="messages-stream"),
        pytest.param("/v1/responses", False, '"status":"completed"', id="responses-unary"),
        pytest.param("/v1/responses", True, "event: response.completed", id="responses-stream"),
    ],
)
def test_copilot_chat_routes_keep_artifact_tool_and_complete(
    copilot_chat: Gateway, route: str, stream: bool, terminator: str
) -> None:
    name_schema: Final = {"type": "string", "pattern": _PATTERN}
    messages: Final = [{"role": "user", "content": "Say hello"}]
    body: Final[dict[str, JsonValue]] = (
        {
            "input": messages,
            "tools": [
                {
                    "type": "function",
                    "name": "Artifact",
                    "description": "Create an artifact",
                    "parameters": _parameters(name_schema),
                }
            ],
            "store": False,
        }
        if route == "/v1/responses"
        else {
            "messages": messages,
            "tools": [
                {
                    "name": "Artifact",
                    "description": "Create an artifact",
                    "input_schema": _parameters(name_schema),
                }
                if route == "/v1/messages"
                else _artifact(name_schema)
            ],
            **({"max_tokens": 64} if route == "/v1/messages" else {}),
        }
    )
    response: Final = copilot_chat.request("POST", route, {**body, "model": _MODEL, "stream": stream})
    assert response.status_code == 200, response.text
    assert "Hello" in response.text and terminator in response.text, response.text


def test_copilot_native_responses_preserves_artifact_pattern(copilot_chat: Gateway) -> None:
    response: Final = copilot_chat.request(
        "POST",
        "/v1/responses",
        {
            "model": _RESPONSES_MODEL,
            "input": "Say hello",
            "tools": [
                {
                    "type": "function",
                    "name": "Artifact",
                    "description": "Create an artifact",
                    "parameters": _parameters({"type": "string", "pattern": _PATTERN}),
                }
            ],
            "store": False,
        },
    )
    assert response.status_code == 200, response.text
    assert _JSON_OBJECT.validate_json(response.content)["status"] == "completed"
