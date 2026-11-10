from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import JsonValue

from tests.integration._support.anthropic_sse import message_json
from tests.integration._support.client import Gateway, gateway_from_environment
from tests.integration._support.process import owned_proxy
from tests.integration._support.provider import PROVIDER_URL, SharedProvider
from tests.integration._support.wire import Reply

_ANTHROPIC_GROUP: Final = "web-search-anthropic"
_GEMINI_GROUP: Final = "web-search-gemini"
_REQUESTS: Final = 10
_PROMPT: Final = "Reply with OK."


def _gemini_json() -> bytes:
    return json.dumps(
        {
            "candidates": [
                {"content": {"parts": [{"text": "OK"}], "role": "model"}, "finishReason": "STOP", "index": 0}
            ],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1, "totalTokenCount": 6},
            "modelVersion": "gemini-3.8-flash",
        }
    ).encode()


def _supported(group: str) -> str:
    return f"{group}-supported"


def _unsupported(group: str) -> str:
    return f"{group}-unsupported"


def _group(group: str, model: str, api_base: str) -> tuple[dict[str, JsonValue], ...]:
    # The unsupported deployment is weighted so a router that keeps it in rotation picks it
    # nearly every time, rather than half the time by chance
    return (
        {
            "model_name": group,
            "litellm_params": {"model": model, "api_key": "sk-fixture", "api_base": api_base, "weight": 1},
            "model_info": {"id": _supported(group)},
        },
        {
            "model_name": group,
            "litellm_params": {"model": model, "api_key": "sk-fixture", "api_base": api_base, "weight": 9},
            "model_info": {"id": _unsupported(group), "supports_web_search": False},
        },
    )


def _tools(tool: Mapping[str, object]) -> dict[str, object]:
    return {"tools": [dict(tool)]}


def _config(directory: Path) -> Path:
    shared: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration: Final = {
        **shared,
        "model_list": [
            *shared["model_list"],
            *_group(_ANTHROPIC_GROUP, "anthropic/claude-haiku-5-5", PROVIDER_URL),
            *_group(_GEMINI_GROUP, "gemini/gemini-3.8-flash", PROVIDER_URL),
        ],
        "router_settings": {**shared.get("router_settings", {}), "routing_strategy": "simple-shuffle"},
    }
    path: Final = directory / "config-web-search-filter.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@pytest.fixture(scope="module")
def web_search_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("web-search-filter")
    with (
        gateway_from_environment() as shared,
        owned_proxy(shared, directory, {}, config=_config(directory)) as owned,
    ):
        yield owned


@pytest.mark.parametrize(
    ("path", "group", "body", "reply"),
    [
        pytest.param(
            "/v1/chat/completions",
            _ANTHROPIC_GROUP,
            _tools({"type": "web_search_20250305", "name": "web_search", "max_uses": 1}),
            message_json("msg_web_search", "claude-haiku-5-5", "OK"),
            id="anthropic-web_search_20250305-chat",
        ),
        pytest.param(
            "/v1/messages",
            _ANTHROPIC_GROUP,
            {"max_tokens": 16, **_tools({"type": "web_search_20260301", "name": "web_search"})},
            message_json("msg_web_search", "claude-haiku-5-5", "OK"),
            id="anthropic-web_search_newer-version-messages",
        ),
        pytest.param(
            "/v1/chat/completions",
            _GEMINI_GROUP,
            _tools({"googleSearch": {}}),
            _gemini_json(),
            id="gemini-googleSearch-chat",
        ),
        pytest.param(
            "/v1/chat/completions",
            _GEMINI_GROUP,
            _tools({"google_search": {}}),
            _gemini_json(),
            id="gemini-google_search-chat",
        ),
    ],
)
def test_a_web_search_request_never_reaches_a_deployment_declared_without_web_search(
    web_search_proxy: Gateway,
    provider: SharedProvider,
    path: str,
    group: str,
    body: Mapping[str, JsonValue],
    reply: bytes,
) -> None:
    provider.expect(*(Reply(body=reply) for _ in range(_REQUESTS)))
    responses: Final = tuple(
        web_search_proxy.request(
            "POST",
            path,
            {"model": group, "messages": [{"role": "user", "content": f"{_PROMPT} {uuid.uuid4().hex}"}], **body},
        )
        for _ in range(_REQUESTS)
    )
    received: Final = provider.received()
    assert all(response.status_code == 200 for response in responses), [response.text for response in responses]
    served: Final = tuple(response.headers["x-litellm-model-id"] for response in responses)
    assert served == (_supported(group),) * _REQUESTS, served
    assert len(received) == _REQUESTS
