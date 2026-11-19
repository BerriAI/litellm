import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, object_value
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_MODEL: Final = "messages-failure-hook"
_MODULE: Final = "failure_hook_replacement"
_REPLACEMENT: Final = "rewritten by hook"
_UPSTREAM_DENIAL: Final = "The security token included in the request is invalid."
_MODULE_SOURCE: Final = f'''from fastapi import HTTPException

from litellm.integrations.custom_logger import CustomLogger


class RetryableDenials(CustomLogger):
    async def async_post_call_failure_hook(
        self, request_data, original_exception, user_api_key_dict, traceback_str=None
    ):
        if getattr(original_exception, "status_code", None) == 403:
            return HTTPException(status_code=500, detail="{_REPLACEMENT}")
        return None


replacer = RetryableDenials()
'''


def _deny(request: Request) -> Reply:
    return Reply(
        status=403,
        body=json.dumps({"type": "error", "error": {"type": "permission_error", "message": _UPSTREAM_DENIAL}}).encode(),
    )


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    upstream: Wire


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with wire_server(_deny) as upstream:
        directory: Final = tmp_path_factory.mktemp("messages-failure-hook")
        (directory / f"{_MODULE}.py").write_text(_MODULE_SOURCE)
        stock: Final = JSON_OBJECT.validate_python(
            yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        )
        config: Final = directory / "messages-failure-hook.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    **stock,
                    "litellm_settings": {
                        **object_value(stock["litellm_settings"]),
                        "callbacks": [f"{_MODULE}.replacer"],
                    },
                    "model_list": [
                        {
                            "model_name": _MODEL,
                            "litellm_params": {
                                "model": "anthropic/claude-haiku-5-5",
                                "api_base": upstream.url,
                                "api_key": "synthetic-anthropic-key",
                            },
                        }
                    ],
                }
            )
        )
        with owned_proxy(session_gateway, directory, {}, config=config) as candidate:
            yield Rig(candidate, upstream)


def _messages_body(marker: str, *, stream: bool) -> dict[str, JsonValue]:
    return {
        "model": _MODEL,
        "max_tokens": 16,
        "stream": stream,
        "messages": [{"role": "user", "content": f"marker-{marker}"}],
        "num_retries": 0,
    }


def _upstream_calls(rig: Rig, marker: str) -> int:
    return sum(1 for request in rig.upstream.drain() if marker.encode() in request.body)


@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream_before_first_byte"))
def test_messages_answers_the_error_the_failure_hook_returns(rig: Rig, stream: bool) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = rig.gateway.request("POST", "/v1/messages", _messages_body(marker, stream=stream))

    assert response.status_code == 500, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert body["type"] == "error", body
    assert object_value(body["error"])["message"] == _REPLACEMENT, body
    assert _UPSTREAM_DENIAL not in response.text, response.text
    assert _upstream_calls(rig, marker) == 1


def test_chat_completions_answers_the_error_the_failure_hook_returns(rig: Rig) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = rig.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": _MODEL, "messages": [{"role": "user", "content": f"marker-{marker}"}], "num_retries": 0},
    )

    assert response.status_code == 500, response.text
    error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
    assert error["message"] == _REPLACEMENT, error
    assert _UPSTREAM_DENIAL not in response.text, response.text
    assert _upstream_calls(rig, marker) == 1
