"""The ``batch_redis_requests`` callback prefetches a key's response cache entries
into memory with one Redis scan, so the next identical request is served without a
Redis round trip for the entry."""

import os
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, object_value
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.redis_process import OwnedRedis, owned_redis
from redis import Redis
from redis.exceptions import TimeoutError as RedisTimeoutError

ROOT: Final = Path(os.environ.get("INTEGRATION_PROXY_ROOT") or Path(__file__).resolve().parents[3])
CELL_TIMEOUT: Final = int(2 * graceful_stop_seconds() + 120)


def _config_with_batch_redis_requests(directory: Path) -> Path:
    shared: Final = object_value(yaml.safe_load((ROOT / "tests/integration/proxy_config.yaml").read_text()))
    litellm_settings: Final = {**object_value(shared["litellm_settings"]), "callbacks": ["batch_redis_requests"]}
    path: Final = directory / f"proxy_config_{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump({**shared, "litellm_settings": litellm_settings}, sort_keys=False))
    return path


@contextmanager
def _monitored(redis: OwnedRedis) -> Iterator[list[str]]:
    commands: Final[list[str]] = []
    stopped: Final = threading.Event()
    with Redis(host=redis.host, port=redis.port, socket_timeout=0.3) as client, client.monitor() as monitor:

        def pump() -> None:
            while not stopped.is_set():
                try:
                    commands.append(str(monitor.next_command()["command"]))
                except RedisTimeoutError:
                    continue

        thread: Final = threading.Thread(target=pump, daemon=True)
        thread.start()
        try:
            yield commands
        finally:
            stopped.set()
            thread.join(timeout=5)


@pytest.mark.timeout(CELL_TIMEOUT)
def test_the_prefetched_response_cache_entry_is_served_from_memory_on_the_next_request(
    gateway: Gateway, tmp_path: Path
) -> None:
    with owned_redis(tmp_path) as redis:
        overrides: Final = {"REDIS_HOST": redis.host, "REDIS_PORT": str(redis.port)}
        with (
            owned_proxy_process(
                gateway, tmp_path, overrides, config=_config_with_batch_redis_requests(tmp_path)
            ) as owned,
            owned.gateway.scenario() as scenario,
            Redis(host=redis.host, port=redis.port) as cache,
            httpx.Client(base_url=gateway.upstream_url, timeout=30) as upstream,
        ):
            model: Final = scenario.model()
            key: Final = scenario.key(models=[model])
            body: Final = {"model": model, "messages": [{"role": "user", "content": f"prefetch {uuid.uuid4().hex}"}]}
            namespace: Final = f"litellm:{key}:acompletion"
            first: Final = owned.gateway.request("POST", "/v1/chat/completions", body, key=key)
            assert first.status_code == 200, first.text
            entries: Final = eventually(lambda: cache.keys(f"{namespace}:*"), lambda keys: len(keys) == 1, seconds=10)
            upstream.get("/__observations").raise_for_status()
            with _monitored(redis) as commands:
                second: Final = owned.gateway.request("POST", "/v1/chat/completions", body, key=key)
            assert second.status_code == 200, second.text
            assert second.json()["id"] == first.json()["id"], (first.text, second.text)
            assert upstream.get("/__observations").json()["requests"] == []
            fetched: Final = [command for command in commands if command.startswith(f"GET {namespace}:")]
            assert fetched == [], (entries, commands)
            third: Final = owned.gateway.request("POST", "/v1/chat/completions", body, key=key)
            assert third.status_code == 200 and third.json()["id"] == first.json()["id"], third.text
            assert upstream.get("/__observations").json()["requests"] == []
