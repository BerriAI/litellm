import json
import uuid
from pathlib import Path
from typing import Final

import pytest

from integration._support.client import Gateway
from integration._support.process import graceful_stop_seconds, owned_proxy
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


def _deployment_params(upstream_url: str) -> dict[str, object]:
    return {"model": "openai/gpt-4.1-mini", "api_base": upstream_url + "/v1", "api_key": USER_KEY}


def _user_config(upstream_url: str) -> dict[str, object]:
    return {
        "model_list": [{"model_name": "user-config-deployment", "litellm_params": _deployment_params(upstream_url)}],
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


def _identity_in_deployment(upstream_url: str, identity_param: str) -> dict[str, object]:
    return {
        "model_list": [
            {
                "model_name": "user-config-deployment",
                "litellm_params": {**_deployment_params(upstream_url), identity_param: "caller-chosen"},
            }
        ],
        "num_retries": 0,
    }


def _identity_in_defaults(upstream_url: str, identity_param: str) -> dict[str, object]:
    return {**_user_config(upstream_url), "default_litellm_params": {identity_param: "caller-chosen"}}


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_user_config_cannot_carry_a_server_owned_identity_param_even_when_opted_in(
    gateway: Gateway, tmp_path: Path
) -> None:
    cases: Final = (
        (_identity_in_deployment, "openai_identity_provider_id"),
        (_identity_in_defaults, "openai_service_account_id"),
    )
    with wire_server(_completion) as upstream:
        config: Final = _opt_in_config(tmp_path, upstream.url)
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            responses: Final = tuple(
                (
                    identity_param,
                    candidate.request(
                        "POST",
                        "/v1/chat/completions",
                        {
                            "model": "user-config-deployment",
                            "messages": [{"role": "user", "content": "user config identity"}],
                            "user_config": placement(upstream.url, identity_param),
                        },
                    ),
                )
                for placement, identity_param in cases
            )
        for identity_param, response in responses:
            assert response.status_code == 401, response.text
            assert identity_param in response.text, response.text
        assert tuple(request for request in upstream.drain() if request.target == "/v1/chat/completions") == ()


def test_user_config_is_rejected_without_the_opt_in(gateway: Gateway) -> None:
    with wire_server(_completion) as upstream:
        response: Final = gateway.request("POST", "/v1/chat/completions", _request_body(upstream.url))
        assert response.status_code == 401, response.text
        assert "user_config is not allowed in request body" in response.text
        assert upstream.received.empty()
