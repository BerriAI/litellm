import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import TypeAdapter
from redis import Redis

from tests.integration._support.client import Gateway, eventually, list_value, object_value, string_value
from tests.integration._support.database import scratch_database
from tests.integration._support.process import graceful_stop_seconds, owned_proxy
from tests.integration._support.redis_process import owned_redis

pytestmark: Final = [pytest.mark.timeout(int(4 * graceful_stop_seconds() + 120))]

COORDINATION_CONFIG: Final = Path(__file__).resolve().parents[1] / "coordination_redis_proxy_config.yaml"
CONFIG_SYNC_CHANNEL: Final = "litellm_proxy.config_change"
SUBSCRIBER_COUNTS: Final = TypeAdapter(tuple[tuple[str, int], ...])
PERIODIC_RELOAD_SECONDS: Final = 3600
PEER_PICKUP_SECONDS: Final = 4
PROVIDER_KEY: Final = "integration-provider-key"


@contextmanager
def _proxy_pair(gateway: Gateway, directory: Path) -> Iterator[tuple[Gateway, Gateway]]:
    with scratch_database("integration_model_add_sync") as database, owned_redis(directory) as coordination:
        shared_stores: Final = {
            "DATABASE_URL": database,
            "REDIS_HOST": coordination.host,
            "REDIS_PORT": str(coordination.port),
            "PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": str(PERIODIC_RELOAD_SECONDS),
        }
        with (
            owned_proxy(gateway, directory / "writer", shared_stores, config=COORDINATION_CONFIG) as writer,
            owned_proxy(gateway, directory / "peer", shared_stores, config=COORDINATION_CONFIG) as peer,
            Redis(host=coordination.host, port=coordination.port, socket_timeout=2) as redis,
        ):
            eventually(
                lambda: SUBSCRIBER_COUNTS.validate_python(redis.pubsub_numsub(CONFIG_SYNC_CHANNEL))[0][1],
                lambda count: count >= 2,
                seconds=30,
            )
            yield writer, peer


@contextmanager
def _failing_upstream_model(gateway: Gateway, upstream_model: str) -> Iterator[None]:
    script: Final = f"{gateway.upstream_url}/__scripts/{upstream_model}"
    httpx.post(script, json={"statuses": [500]}, timeout=5, trust_env=False).raise_for_status()
    try:
        yield
    finally:
        httpx.delete(script, timeout=5, trust_env=False).raise_for_status()


def _add_deployment(writer: Gateway, group: str, upstream_model: str, order: int) -> None:
    writer.post(
        "/model/new",
        {
            "model_name": group,
            "litellm_params": {
                "model": f"openai/{upstream_model}",
                "api_key": PROVIDER_KEY,
                "api_base": f"{writer.upstream_url}/v1",
                "order": order,
                "num_retries": 0,
            },
        },
    )


def _listed_models(proxy: Gateway) -> frozenset[str]:
    return frozenset(string_value(object_value(entry)["id"]) for entry in list_value(proxy.get("/v1/models")["data"]))


def _chat(proxy: Gateway, group: str) -> httpx.Response:
    return proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": group, "messages": [{"role": "user", "content": f"order fallback {uuid.uuid4().hex}"}]},
    )


def test_a_backup_added_on_one_proxy_catches_the_peers_failing_primary_right_away(
    gateway: Gateway, tmp_path: Path
) -> None:
    group: Final = f"integration-{uuid.uuid4().hex}"
    primary: Final = f"integration-primary-{uuid.uuid4().hex}"
    backup: Final = f"integration-backup-{uuid.uuid4().hex}"
    with _proxy_pair(gateway, tmp_path) as (writer, peer), _failing_upstream_model(gateway, primary):
        _add_deployment(writer, group, primary, order=1)
        eventually(lambda: _listed_models(peer), lambda listed: group in listed, seconds=30)

        _add_deployment(writer, group, backup, order=2)
        on_writer: Final = _chat(writer, group)
        on_peer: Final = eventually(
            lambda: _chat(peer, group),
            lambda response: response.status_code == 200,
            seconds=PEER_PICKUP_SECONDS,
            return_last_on_timeout=True,
        )

    assert on_writer.status_code == 200, on_writer.text
    assert object_value(on_writer.json())["model"] == backup, on_writer.text
    assert on_peer.status_code == 200, on_peer.text
    assert object_value(on_peer.json())["model"] == backup, on_peer.text
