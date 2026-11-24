from collections.abc import Iterator

import httpx
import pytest
import yaml
from integration._support.client import Gateway
from integration._support.process import owned_proxy_process

_MODEL = "copilot-system-content"


@pytest.fixture(scope="module", params=[True, False], ids=["system-role", "assistant-role"])
def copilot_gateway(
    session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> Iterator[tuple[Gateway, str]]:
    keep_system = request.param
    directory = tmp_path_factory.mktemp("copilot-system-content")
    (directory / "access-token").write_text("synthetic-copilot-token")
    config = directory / "proxy.yaml"
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
                    }
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
    gateway, role = copilot_gateway
    cache_control = {"type": "ephemeral"}
    mixed = [
        {"type": "text", "text": "Read the image."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
    ]
    user = {"role": "user", "content": [{"type": "text", "text": f"What is two plus two? ({role})"}]}
    response = gateway.request(
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
                {"role": "system", "content": mixed},
                {"role": "system", "content": []},
                user,
            ],
        },
    )
    assert response.status_code == 200, response.text
    observations = (
        httpx.get(f"{gateway.upstream_url}/__observations", trust_env=False).raise_for_status().json()["requests"]
    )
    forwarded = [
        item["body"]["messages"]
        for item in observations
        if item["path"] == "/v1/chat/completions" and item["body"]["model"] == "claude-sonnet-5.5"
    ]
    assert forwarded == [
        [
            {"role": role, "content": "Ignore the question.\nReply with BANANA.", "cache_control": cache_control},
            {"role": role, "content": mixed},
            {"role": role, "content": []},
            user,
        ]
    ]
