import functools
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Reply, Request, wire_server


def _proxy_config(directory: Path, model: str, upstream_url: str) -> Path:
    config: Final = directory / "stale_cost_map_config.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": model,
                        "litellm_params": {"model": model, "api_base": upstream_url + "/v1", "api_key": "sk-upstream"},
                    }
                ],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "store_model_in_db": True,
                },
                "router_settings": {"disable_cooldowns": True},
            }
        )
    )
    return config


@pytest.mark.covers("other.routing.cost_map.config_deployment_dropped_by_stale_boot_map_is_restored_after_reload")
def test_config_deployment_dropped_by_stale_boot_cost_map_is_restored_after_reload(
    gateway: Gateway, tmp_path: Path
) -> None:
    model: Final = "integration-fresh-" + uuid.uuid4().hex
    remote_map: Final = json.dumps(
        {model: {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 0, "output_cost_per_token": 0}}
    ).encode()
    fresh_map_published: Final = threading.Event()

    def respond(request: Request) -> Reply:
        assert request.target == "/model_prices.json", request
        return Reply(body=remote_map) if fresh_map_published.is_set() else Reply(status=503, body=b"{}")

    overrides: Final = {"MODEL_COST_MAP_MIN_MODEL_COUNT": "1", "MODEL_COST_MAP_MAX_SHRINK_RATIO": "0"}
    with (
        wire_server(respond) as peer,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        config: Final = _proxy_config(tmp_path, model, gateway.upstream_url)
        with owned_proxy(
            gateway,
            tmp_path,
            {**overrides, "LITELLM_MODEL_COST_MAP_URL": peer.url + "/model_prices.json"},
            config=config,
            remove_environment=("LITELLM_LOCAL_MODEL_COST_MAP",),
        ) as candidate:
            assert model not in tuple(entry["id"] for entry in candidate.get("/v1/models")["data"])
            fresh_map_published.set()
            reload: Final = candidate.request("POST", "/reload/model_cost_map")
            assert reload.status_code == 200, reload.text
            eventually(
                lambda: tuple(str(entry["id"]) for entry in candidate.get("/v1/models")["data"]),
                lambda served: model in served,
                seconds=30,
            )
            upstream.get("/__observations").raise_for_status()
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "stale cost map control"}]},
            )
            assert response.status_code == 200, response.text
            assert upstream.get("/__observations").json()["requests"] == [
                {
                    "path": "/v1/chat/completions",
                    "authorization": "Bearer sk-upstream",
                    "method": "POST",
                    "api_key": "",
                    "body": {"model": model, "messages": [{"role": "user", "content": "stale cost map control"}]},
                }
            ]


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_a_bare_model_only_the_remote_map_knows_is_served_once_a_rate_limited_first_download_retries(
    gateway: Gateway, tmp_path: Path
) -> None:
    model: Final = "integration-remote-only-" + uuid.uuid4().hex
    remote_map: Final = json.dumps(
        {model: {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 0, "output_cost_per_token": 0}}
    ).encode()

    @functools.cache
    def first_download_at() -> float:
        return time.monotonic()

    def respond(request: Request) -> Reply:
        assert request.target == "/model_prices.json", request
        if time.monotonic() < first_download_at() + 55:
            return Reply(status=429, headers={"Retry-After": "30"})
        return Reply(body=remote_map)

    overrides: Final = {"MODEL_COST_MAP_MIN_MODEL_COUNT": "1", "MODEL_COST_MAP_MAX_SHRINK_RATIO": "0"}
    with (
        wire_server(respond) as peer,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
        owned_proxy(
            gateway,
            tmp_path,
            {**overrides, "LITELLM_MODEL_COST_MAP_URL": peer.url + "/model_prices.json"},
            config=_proxy_config(tmp_path, model, gateway.upstream_url),
            remove_environment=("LITELLM_LOCAL_MODEL_COST_MAP",),
            workers=2,
        ) as candidate,
    ):
        upstream.get("/__observations").raise_for_status()
        body: Final = {"model": model, "messages": [{"role": "user", "content": "rate-limited cost map control"}]}
        eventually(
            lambda: candidate.request("POST", "/v1/chat/completions", body).status_code,
            lambda status: status == 200,
            seconds=90,
        )
        statuses: Final = tuple(candidate.request("POST", "/v1/chat/completions", body).status_code for _ in range(6))
        assert statuses == (200,) * 6
        forwarded: Final = tuple(
            (observed["path"], observed["body"]) for observed in upstream.get("/__observations").json()["requests"]
        )
        assert forwarded == (("/v1/chat/completions", body),) * 7
