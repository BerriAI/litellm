from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.forward_proxy import refusing_forward_proxy
from integration._support.process import owned_proxy
from pydantic import JsonValue

_PACKAGE: Final = Path(__file__).resolve().parents[3] / "litellm"
_CONTROL_HOST: Final = "provider.offline-mode.example"
_REMOTE_URLS: Final = {
    "LITELLM_MODEL_COST_MAP_URL": "https://cost-map.offline-mode.example/model_prices.json",
    "LITELLM_ANTHROPIC_BETA_HEADERS_URL": "https://beta-headers.offline-mode.example/config.json",
    "LITELLM_AUTOROUTER_PRESETS_URL": "https://autorouter.offline-mode.example/presets.json",
    "LITELLM_BLOG_POSTS_URL": "https://blog.offline-mode.example/rss.xml",
}
_SELF_INITIATED_HOSTS: Final = frozenset(
    {
        "cost-map.offline-mode.example:443",
        "beta-headers.offline-mode.example:443",
        "autorouter.offline-mode.example:443",
        "blog.offline-mode.example:443",
        "raw.githubusercontent.com:443",
        "docs.litellm.ai:443",
        "api.pwnedpasswords.com:443",
        "huggingface.co:443",
    }
)
_PER_FEATURE_FLAGS: Final = (
    "LITELLM_OFFLINE_MODE",
    "LITELLM_LOCAL_MODEL_COST_MAP",
    "LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS",
    "LITELLM_LOCAL_AUTOROUTER_PRESETS",
    "LITELLM_LOCAL_POLICY_TEMPLATES",
    "LITELLM_LOCAL_BLOG_POSTS",
    "LITELLM_DISABLE_LATEST_RELEASE_CHECK",
    "LITELLM_LICENSE",
)


def _bundled(name: str) -> JsonValue:
    return json.loads((_PACKAGE / name).read_text(encoding="utf-8"))


def _write_config(directory: Path) -> Path:
    config: Final = directory / f"offline_mode_{uuid.uuid4().hex}.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": "offline-mode-control",
                        "litellm_params": {
                            "model": "openai/gpt-4o-mini",
                            "api_base": f"https://{_CONTROL_HOST}/v1",
                            "api_key": "synthetic-provider-key",
                        },
                    },
                    {
                        "model_name": "offline-mode-kimi",
                        "litellm_params": {
                            "model": "bedrock/us.moonshotai.kimi-k3",
                            "aws_region_name": "us-east-1",
                            "aws_access_key_id": "synthetic-access-key",
                            "aws_secret_access_key": "synthetic-secret-key",
                        },
                    },
                ],
                "router_settings": {"num_retries": 0},
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                },
            }
        )
    )
    return config


def _json(proxy: Gateway, method: str, path: str, body: dict[str, JsonValue] | None = None) -> JsonValue:
    response: Final = proxy.request(method, path, body)
    assert response.status_code == 200, f"{method} {path}: {response.status_code} {response.text}"
    return json.loads(response.content)


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    ("offline_mode", "expected_hosts", "reload_status"),
    [
        pytest.param(None, _SELF_INITIATED_HOSTS | {f"{_CONTROL_HOST}:443"}, 502, id="unset"),
        pytest.param("true", frozenset({f"{_CONTROL_HOST}:443"}), 200, id="true"),
    ],
)
def test_offline_mode_stops_every_self_initiated_request(
    gateway: Gateway, tmp_path: Path, offline_mode: str | None, expected_hosts: frozenset[str], reload_status: int
) -> None:
    with refusing_forward_proxy() as egress:
        environment: Final = {
            "HTTPS_PROXY": egress.url,
            "HTTP_PROXY": egress.url,
            "ALL_PROXY": egress.url,
            "NO_PROXY": "127.0.0.1,localhost",
            "AIOHTTP_TRUST_ENV": "True",
            **_REMOTE_URLS,
            **({"LITELLM_OFFLINE_MODE": offline_mode} if offline_mode is not None else {}),
        }
        with owned_proxy(
            gateway,
            tmp_path,
            environment,
            config=_write_config(tmp_path),
            remove_environment=_PER_FEATURE_FLAGS,
            workers=2,
        ) as isolated:
            source: Final = object_value(_json(isolated, "GET", "/model/cost_map/source"))
            reloaded: Final = isolated.request("POST", "/reload/model_cost_map")
            beta_headers: Final = object_value(_json(isolated, "POST", "/reload/anthropic_beta_headers"))
            release: Final = _json(isolated, "GET", "/get/latest_release_info")
            presets: Final = object_value(_json(isolated, "GET", "/public/autorouter_presets"))
            templates: Final = _json(isolated, "GET", "/policy/templates")
            blog: Final = object_value(_json(isolated, "GET", "/public/litellm_blog_posts"))
            user: Final = object_value(
                _json(isolated, "POST", "/user/new", {"user_email": f"offline-{uuid.uuid4().hex}@example.com"})
            )
            password_set: Final = object_value(
                _json(
                    isolated,
                    "POST",
                    "/user/update",
                    {"user_id": user["user_id"], "password": f"Offline-{uuid.uuid4()}"},
                )
            )
            tokens: Final = object_value(
                _json(
                    isolated,
                    "POST",
                    "/utils/token_counter",
                    {"model": "meta-llama/llama-3-8b-instruct", "messages": [{"role": "user", "content": "count me"}]},
                )
            )
            kimi_tokens: Final = object_value(
                _json(
                    isolated,
                    "POST",
                    "/utils/token_counter",
                    {"model": "offline-mode-kimi", "messages": [{"role": "user", "content": "count me"}]},
                )
            )
            control: Final = isolated.request(
                "POST",
                "/v1/chat/completions",
                {"model": "offline-mode-control", "messages": [{"role": "user", "content": "control"}]},
            )
        reached: Final = frozenset(egress.targets())

    assert reached == expected_hosts
    assert control.status_code != 200, control.text
    assert source["source"] == "local", source
    assert source["is_env_forced"] is (offline_mode is not None), source
    assert reloaded.status_code == reload_status, reloaded.text
    assert beta_headers["status"] == "success", beta_headers
    assert release is None
    assert sorted(presets) == sorted(object_value(_bundled("proxy/public_endpoints/autorouter_presets.json")))
    assert templates == _bundled("policy_templates_backup.json")
    assert [object_value(post)["url"] for post in _list(blog["posts"])] == [
        object_value(post)["url"] for post in _list(object_value(_bundled("blog_posts.json"))["posts"])
    ]
    assert password_set["user_id"] == user["user_id"], password_set
    assert tokens["tokenizer_type"] == "openai_tokenizer", tokens
    assert isinstance(kimi_tokens["total_tokens"], int) and kimi_tokens["total_tokens"] > 0, kimi_tokens


def _list(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list), value
    return value
