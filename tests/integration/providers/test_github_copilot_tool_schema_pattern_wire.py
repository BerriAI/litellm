import json
from collections.abc import Iterator
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment
from integration._support.device_login import write_copilot_access
from integration._support.process import OwnedState, owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter
from typing_extensions import TypedDict

pytestmark: Final = pytest.mark.timeout(180)

_MODEL: Final = "github_copilot/gpt-4.1"
_PATTERN: Final = r'^(?!__.*__$)[^\p{Cc}\p{Cf}\p{Zl}\p{Zp}"\\./[\]]{1,200}$'
_LOOKAHEAD: Final = r"^(?!__)[a-z]+$"


class _Parameters(TypedDict):
    type: str
    properties: dict[str, dict[str, JsonValue]]
    required: list[str]


class _Function(TypedDict):
    name: str
    description: str
    parameters: _Parameters


class _Tool(TypedDict):
    type: str
    function: _Function


class _ChatRequest(TypedDict):
    tools: list[_Tool]


_CHAT_REQUEST: Final = TypeAdapter(_ChatRequest)


def _artifact(name_schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "type": "function",
        "function": {
            "name": "Artifact",
            "description": "Create an artifact",
            "parameters": {
                "type": "object",
                "properties": {"name": name_schema, "slug": {"type": "string", "pattern": _LOOKAHEAD}},
                "required": ["name"],
            },
        },
    }


def _copilot_chat(request: Request) -> Reply:
    assert request.target == "/chat/completions"
    tools: Final = _CHAT_REQUEST.validate_json(request.body)["tools"]
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
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-wire",
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
                {"model_list": [{"model_name": _MODEL, "litellm_params": {"model": _MODEL, "api_base": upstream.url}}]}
            )
        )
        with owned_proxy(
            gateway,
            directory,
            {"GITHUB_COPILOT_TOKEN_DIR": str(directory)},
            config=config,
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
