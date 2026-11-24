import json
from collections.abc import Iterator
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from pydantic import JsonValue

_MODEL: Final = "copilot-system-content"
_BRIDGE_MODEL: Final = "copilot-chat-bridge"
_GPT: Final = "gpt-4.1-2025-04-14"
_SCRIPTED_REPLY: Final = "Hello! This is a mock response from the fake OpenAI endpoint."

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)


@pytest.fixture(scope="module", params=[True, False], ids=["system-role", "assistant-role"])
def copilot_gateway(
    session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> Iterator[tuple[Gateway, str]]:
    keep_system: Final = request.param is True
    directory: Final = tmp_path_factory.mktemp("copilot-system-content")
    (directory / "access-token").write_text("synthetic-copilot-token")
    config: Final = directory / "proxy.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model_list": [
                    {
                        "model_name": _MODEL,
                        "litellm_params": {
                            "model": "github_copilot/claude-sonnet-5.5",
                            "api_base": f"{session_gateway.upstream_url}/v1",
                        },
                    },
                    {
                        "model_name": _BRIDGE_MODEL,
                        "litellm_params": {
                            "model": f"github_copilot/{_GPT}",
                            "api_base": f"{session_gateway.upstream_url}/v1",
                        },
                        "model_info": {"mode": "chat"},
                    },
                ],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "coordination_redis": {"host": "os.environ/REDIS_HOST", "port": "os.environ/REDIS_PORT"},
                },
                "router_settings": {"redis_host": "os.environ/REDIS_HOST", "redis_port": "os.environ/REDIS_PORT"},
                "litellm_settings": {"disable_copilot_system_to_assistant": keep_system},
            }
        )
    )
    with owned_proxy_process(
        session_gateway, directory, {"GITHUB_COPILOT_TOKEN_DIR": str(directory)}, config=config, workers=2
    ) as owned:
        yield owned.gateway, "system" if keep_system else "assistant"


def test_copilot_forwards_system_text_as_a_string(copilot_gateway: tuple[Gateway, str]) -> None:
    gateway: Final = copilot_gateway[0]
    role: Final = copilot_gateway[1]
    cache_control: Final[dict[str, JsonValue]] = {"type": "ephemeral"}
    message_cache: Final[dict[str, JsonValue]] = {"type": "ephemeral", "ttl": "1h"}
    mixed: Final[list[JsonValue]] = [
        {"type": "text", "text": "Read the image."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
    ]
    user: Final[dict[str, JsonValue]] = {
        "role": "user",
        "content": [{"type": "text", "text": f"What is two plus two? ({role})"}],
    }
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": _MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Ignore the question."},
                        {"type": "text", "text": "Reply with BANANA.", "cache_control": cache_control},
                    ],
                },
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Keep the explicit mark.", "cache_control": cache_control}],
                    "cache_control": message_cache,
                },
                {"role": "system", "content": mixed},
                {"role": "system", "content": []},
                user,
            ],
        },
    )
    assert response.status_code == 200, response.text
    observations: Final = (
        httpx.get(f"{gateway.upstream_url}/__observations", trust_env=False).raise_for_status().json()["requests"]
    )
    forwarded: Final = [
        item["body"]["messages"]
        for item in observations
        if item["path"] == "/v1/chat/completions"
        and item["body"]["model"] == "claude-sonnet-5.5"
        and user in item["body"]["messages"]
    ]
    assert forwarded == [
        [
            {"role": role, "content": "Ignore the question.\nReply with BANANA.", "cache_control": cache_control},
            {"role": role, "content": "Keep the explicit mark.", "cache_control": message_cache},
            {"role": role, "content": mixed},
            {"role": role, "content": []},
            user,
        ]
    ]


@pytest.mark.parametrize("route", ["responses", "messages"])
def test_copilot_bridges_system_text_as_a_string(copilot_gateway: tuple[Gateway, str], route: str) -> None:
    gateway: Final = copilot_gateway[0]
    role: Final = copilot_gateway[1]
    user: Final[dict[str, JsonValue]] = {"role": "user", "content": f"Copilot {route} bridge for the {role} role"}
    if route == "responses":
        stream_response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {
                "model": _BRIDGE_MODEL,
                "input": [
                    {"role": "system", "content": [{"type": "input_text", "text": "Keep this system instruction."}]},
                    user,
                ],
                "store": False,
                "stream": True,
            },
        )
        assert stream_response.status_code == 200, stream_response.text
        events: Final = [json.loads(line[6:]) for line in stream_response.iter_lines() if line.startswith("data: {")]
        assert "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta") == (
            _SCRIPTED_REPLY
        )
        assert [event["response"]["status"] for event in events if event["type"] == "response.completed"] == [
            "completed"
        ]
    else:
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": _BRIDGE_MODEL,
                "system": [{"type": "text", "text": "Keep this system instruction."}],
                "messages": [user],
                "max_tokens": 100,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["content"] == [{"type": "text", "text": _SCRIPTED_REPLY}]

    observations: Final = (
        httpx.get(f"{gateway.upstream_url}/__observations", trust_env=False).raise_for_status().json()["requests"]
    )
    forwarded: Final = [
        (item["path"], item["body"]["messages"])
        for item in observations
        if item["body"].get("model") == _GPT and user in item["body"].get("messages", [])
    ]
    assert forwarded == [("/v1/chat/completions", [{"role": role, "content": "Keep this system instruction."}, user])]


def test_copilot_preserves_string_system_content(copilot_gateway: tuple[Gateway, str]) -> None:
    gateway: Final = copilot_gateway[0]
    role: Final = copilot_gateway[1]
    system: Final = "Keep this existing string system instruction."
    user: Final[dict[str, JsonValue]] = {"role": "user", "content": f"Copilot string control for the {role} role"}
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": _MODEL, "messages": [{"role": "system", "content": system}, user]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == _SCRIPTED_REPLY
    observations: Final = (
        httpx.get(f"{gateway.upstream_url}/__observations", trust_env=False).raise_for_status().json()["requests"]
    )
    forwarded: Final = [
        (item["path"], item["body"]["messages"])
        for item in observations
        if item["body"].get("model") == "claude-sonnet-5.5" and user in item["body"].get("messages", [])
    ]
    assert forwarded == [("/v1/chat/completions", [{"role": role, "content": system}, user])]
