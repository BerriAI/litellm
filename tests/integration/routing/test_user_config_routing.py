import json
import uuid
from pathlib import Path
from typing import Final

from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

USER_KEY: Final = "sk-user-supplied-" + uuid.uuid4().hex


def _completion(request: Request) -> Reply:
    if request.target != "/v1/chat/completions":
        return Reply(status=404, body=b"{}")
    body: Final = json.loads(request.body)
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-user-config",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "routed"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
        ).encode()
    )


def _user_config(upstream_url: str) -> dict[str, object]:
    return {
        "model_list": [
            {
                "model_name": "user-config-deployment",
                "litellm_params": {
                    "model": "openai/gpt-4.1-mini",
                    "api_base": upstream_url + "/v1",
                    "api_key": USER_KEY,
                },
            }
        ],
        "num_retries": 0,
    }


def _opt_in_config(directory: Path, upstream_url: str) -> Path:
    config: Final = directory / "allow_client_side_credentials_config.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": "admin-deployment",
                        "litellm_params": {
                            "model": "openai/gpt-4.1-mini",
                            "api_base": upstream_url + "/v1",
                            "api_key": "sk-admin-configured",
                        },
                    }
                ],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "store_model_in_db": True,
                    "allow_client_side_credentials": True,
                },
            }
        )
    )
    return config


def _request_body(upstream_url: str) -> dict[str, object]:
    return {
        "model": "user-config-deployment",
        "messages": [{"role": "user", "content": "user config control"}],
        "user_config": _user_config(upstream_url),
    }


def test_user_config_routes_to_the_user_supplied_deployment_when_opted_in(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_completion) as upstream:
        config: Final = _opt_in_config(tmp_path, upstream.url)
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            response: Final = candidate.request("POST", "/v1/chat/completions", _request_body(upstream.url))
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "routed"
        outbound: Final = tuple(upstream.received.get_nowait() for _ in range(upstream.received.qsize()))
        completions: Final = tuple(request for request in outbound if request.target == "/v1/chat/completions")
        assert len(completions) == 1, outbound
        assert completions[0].headers["authorization"] == f"Bearer {USER_KEY}"
        assert json.loads(completions[0].body)["model"] == "gpt-4.1-mini"


def test_user_config_is_rejected_without_the_opt_in(gateway: Gateway) -> None:
    with wire_server(_completion) as upstream:
        response: Final = gateway.request("POST", "/v1/chat/completions", _request_body(upstream.url))
        assert response.status_code == 401, response.text
        assert "user_config is not allowed in request body" in response.text
        assert upstream.received.empty()
