import base64
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, JsonValue, object_value, string_value
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

_RAISING_HOOK: Final = """
from litellm.integrations.custom_logger import CustomLogger


class RaisingHook(CustomLogger):
    async def async_post_call_success_deployment_hook(self, request_data, response, call_type):
        raise RuntimeError(f"hook rejected {type(response).__name__} for {call_type}")


instance = RaisingHook()
"""

_VIDEO_JOB: Final = {
    "id": "video_hook_isolation",
    "object": "video",
    "status": "queued",
    "model": "sora-2",
    "seconds": "4",
    "size": "720x1280",
}

_UPSTREAM_REPLIES: Final[Mapping[str, Mapping[str, JsonValue]]] = {
    "/v1/chat/completions": {
        "id": "chatcmpl_hook_isolation",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5.6",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
    "/v1/embeddings": {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    },
    "/v1/responses": {
        "id": "resp_hook_isolation",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-5.6",
        "output": [
            {
                "type": "message",
                "id": "msg_hook_isolation",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi", "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    },
    "/v1/videos": _VIDEO_JOB,
}


def _item(value: JsonValue, index: int) -> JsonValue:
    assert isinstance(value, list), f"Expected a list, received {type(value).__name__}"
    return value[index]


def _chat_text(body: dict[str, JsonValue]) -> str:
    return string_value(object_value(object_value(_item(body["choices"], 0))["message"])["content"])


def _embedding_vector(body: dict[str, JsonValue]) -> JsonValue:
    return object_value(_item(body["data"], 0))["embedding"]


def _responses_text(body: dict[str, JsonValue]) -> str:
    return string_value(object_value(_item(object_value(_item(body["output"], 0))["content"], 0))["text"])


def _video_job(body: dict[str, JsonValue]) -> tuple[str, str]:
    encoded_id: Final = string_value(body["id"]).removeprefix("video_")
    decoded: Final = base64.b64decode(encoded_id).decode()
    return decoded.rsplit("video_id:", 1)[-1], string_value(body["status"])


@dataclass(frozen=True, slots=True)
class _Surface:
    route: str
    upstream_model: str
    body: Callable[[str], dict[str, JsonValue]]
    observed: Callable[[dict[str, JsonValue]], JsonValue | tuple[str, str]]
    expected: JsonValue | tuple[str, str]


_SURFACES: Final = (
    pytest.param(
        _Surface(
            "/v1/chat/completions",
            "openai/gpt-5.6",
            lambda model: {"model": model, "messages": [{"role": "user", "content": "hook isolation"}]},
            _chat_text,
            "hi",
        ),
        id="chat",
    ),
    pytest.param(
        _Surface(
            "/v1/embeddings",
            "openai/text-embedding-3-small",
            lambda model: {"model": model, "input": "hook isolation"},
            _embedding_vector,
            [0.1, 0.2],
        ),
        id="embeddings",
    ),
    pytest.param(
        _Surface(
            "/v1/responses",
            "openai/gpt-5.6",
            lambda model: {"model": model, "input": "hook isolation"},
            _responses_text,
            "hi",
        ),
        id="responses",
    ),
    pytest.param(
        _Surface(
            "/v1/videos",
            "openai/sora-2",
            lambda model: {"model": model, "prompt": "a cat"},
            _video_job,
            (_VIDEO_JOB["id"], _VIDEO_JOB["status"]),
        ),
        id="videos",
    ),
)


@pytest.mark.covers("other.observability.callbacks.raising_success_deployment_hook_keeps_response")
@pytest.mark.parametrize("surface", _SURFACES)
def test_response_survives_raising_success_deployment_hook(gateway: Gateway, tmp_path: Path, surface: _Surface) -> None:
    def upstream(request: Request) -> Reply:
        assert request.target == surface.route, request.target
        assert b"hook isolation" in request.body or b"a cat" in request.body, request.body[:300]
        return Reply(body=json.dumps(_UPSTREAM_REPLIES[surface.route]).encode())

    (tmp_path / "raising_hook.py").write_text(_RAISING_HOOK)
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"].update({"callbacks": ["raising_hook.instance"]})
    path: Final = tmp_path / "hook.yaml"
    path.write_text(yaml.safe_dump(config))
    with (
        wire_server(upstream) as provider,
        owned_proxy(gateway, tmp_path, {}, config=path) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model=surface.upstream_model, api_base=provider.url + "/v1")
        response: Final = candidate.request("POST", surface.route, surface.body(model))
        assert response.status_code == 200, response.text
        assert surface.observed(object_value(response.json())) == surface.expected, response.text
