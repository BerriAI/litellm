from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, object_value
from integration._support.openai_wire import chat_reply
from integration._support.process import owned_proxy
from integration._support.provider import SharedProvider
from integration.caching.response_cache_case import await_entry, redis_store
from pydantic import JsonValue

pytestmark: Final = pytest.mark.shares_worker_state


@pytest.mark.parametrize(
    "ssl_settings",
    [
        pytest.param({"ssl": False, "ssl_check_hostname": False}, id="saved-by-cache-settings-form"),
        pytest.param(
            {"ssl": False, "ssl_cert_reqs": "none", "ssl_ca_certs": "/unused/ca.pem"}, id="leftover-tls-fields"
        ),
    ],
)
def test_redis_cache_stores_responses_when_ssl_is_off_and_ssl_settings_are_present(
    gateway: Gateway, provider: SharedProvider, tmp_path: Path, ssl_settings: dict[str, JsonValue]
) -> None:
    config: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config_path: Final = tmp_path / "non-tls-redis-cache.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                **config,
                "litellm_settings": {
                    **object_value(config["litellm_settings"]),
                    "cache_params": {
                        "type": "redis",
                        "host": os.environ["REDIS_HOST"],
                        "port": os.environ["REDIS_PORT"],
                        **ssl_settings,
                    },
                },
            }
        )
    )
    answer: Final = f"cached answer {uuid.uuid4().hex}"
    provider.expect(chat_reply(f"chatcmpl-{uuid.uuid4().hex}", "gpt-5.4", answer, stream=False))
    body: Final[dict[str, JsonValue]] = {
        "model": "openai/gpt-5.4",
        "messages": [{"role": "user", "content": f"non-tls redis {uuid.uuid4().hex}"}],
    }

    with owned_proxy(gateway, tmp_path, {}, config=config_path) as proxy, redis_store() as store:
        proxy.post("/v1/chat/completions", body)
        await_entry(store, answer)
        proxy.post("/v1/chat/completions", body)
        ping: Final = proxy.request("GET", "/cache/ping")

    assert len(provider.received()) == 1, "the repeated request was not served from the redis cache"
    assert ping.status_code == 200, ping.text
    assert object_value(ping.json())["status"] == "healthy", ping.text
