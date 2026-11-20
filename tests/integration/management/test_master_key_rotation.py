import uuid
from pathlib import Path
from typing import Final

import httpx
import pytest

from litellm.constants import PROXY_CONFIG_RELOAD_INTERVAL_SECONDS
from tests.integration._support.client import Gateway, eventually, list_value, object_value, string_value
from tests.integration._support.database import scratch_database
from tests.integration._support.process import owned_proxy

SALT_KEY: Final = "sk-integration-rotation-salt"


def _register(proxy: Gateway, upstream: str, model_key: str, credential_key: str) -> tuple[str, str]:
    credential: Final = f"rotation-credential-{uuid.uuid4().hex}"
    inline_model: Final = f"rotation-inline-{uuid.uuid4().hex}"
    credential_model: Final = f"rotation-credential-model-{uuid.uuid4().hex}"
    proxy.post(
        "/credentials",
        {
            "credential_name": credential,
            "credential_values": {"api_key": credential_key},
            "credential_info": {"custom_llm_provider": "openai"},
        },
    )
    proxy.post(
        "/model/new",
        {
            "model_name": inline_model,
            "litellm_params": {"model": "openai/gpt-4o-mini", "api_base": upstream, "api_key": model_key},
        },
    )
    proxy.post(
        "/model/new",
        {
            "model_name": credential_model,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_base": upstream,
                "litellm_credential_name": credential,
            },
        },
    )
    return inline_model, credential_model


def _statuses(proxy: Gateway, models: tuple[str, ...], key: str, text: str) -> dict[str, int]:
    return {
        model: proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": text}]},
            key=key,
        ).status_code
        for model in models
    }


def _upstream_authorizations(gateway: Gateway) -> list[str]:
    response: Final = httpx.get(f"{gateway.upstream_url}/__observations", timeout=15, trust_env=False)
    response.raise_for_status()
    return [
        string_value(object_value(entry)["authorization"])
        for entry in list_value(response.json()["requests"])
        if string_value(object_value(entry)["path"]).endswith("/chat/completions")
    ]


def _clear_upstream(gateway: Gateway) -> None:
    httpx.delete(f"{gateway.upstream_url}/__observations", timeout=15, trust_env=False).raise_for_status()


@pytest.mark.timeout(300)
def test_master_key_rotation_with_a_salt_key_keeps_db_models_and_credentials_working(
    gateway: Gateway, tmp_path: Path
) -> None:
    new_master_key: Final = f"sk-integration-rotated-{uuid.uuid4().hex}"
    model_key: Final = f"sk-model-{uuid.uuid4().hex}"
    credential_key: Final = f"sk-credential-{uuid.uuid4().hex}"
    expected: Final = [f"Bearer {model_key}", f"Bearer {credential_key}"]
    (tmp_path / "before").mkdir()
    (tmp_path / "after").mkdir()
    with scratch_database() as database_url:
        environment: Final = {"DATABASE_URL": database_url, "LITELLM_SALT_KEY": SALT_KEY}
        with owned_proxy(gateway, tmp_path / "before", environment) as before:
            models: Final = _register(before, f"{gateway.upstream_url}/v1", model_key, credential_key)
            _clear_upstream(gateway)
            assert _statuses(before, models, gateway.key, "before rotation") == dict.fromkeys(models, 200)
            assert _upstream_authorizations(gateway) == expected
            rotated: Final = before.post("/key/regenerate", {"key": gateway.key, "new_master_key": new_master_key})
            assert rotated["key"] == new_master_key

        with owned_proxy(gateway, tmp_path / "after", {**environment, "LITELLM_MASTER_KEY": new_master_key}) as after:
            _clear_upstream(gateway)
            served: Final = eventually(
                lambda: _statuses(after, models, new_master_key, "after rotation"),
                lambda statuses: statuses == dict.fromkeys(models, 200),
                seconds=2 * PROXY_CONFIG_RELOAD_INTERVAL_SECONDS,
                return_last_on_timeout=True,
            )
            assert served == dict.fromkeys(models, 200)
            assert _upstream_authorizations(gateway)[-2:] == expected
