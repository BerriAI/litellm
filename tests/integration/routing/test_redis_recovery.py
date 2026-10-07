import os
import uuid
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import httpx
import psycopg
import pytest
from psycopg import sql
from redis import Redis

from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy
from integration._support.redis_process import owned_redis


@pytest.mark.covers("other.routing.redis.owned_outage_recovers_serving_and_response_cache")
def test_owned_redis_outage_recovers_requests_and_real_response_cache(gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original: Final = os.environ["DATABASE_URL"]
    identity: Final = "integration_recovery_" + uuid.uuid4().hex
    parsed: Final = urlsplit(original)
    database_url: Final = urlunsplit((parsed.scheme, parsed.netloc, "/" + identity, "", ""))
    with psycopg.connect(original, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(identity)))
        try:
            with owned_redis(tmp_path) as cache, monkeypatch.context() as environment:
                environment.setenv("DATABASE_URL", database_url)
                with owned_proxy(gateway, tmp_path, {"DATABASE_URL": database_url, "REDIS_HOST": cache.host, "REDIS_PORT": str(cache.port), "REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT": "1"}) as candidate, candidate.scenario() as scenario, httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream:
                    model: Final = scenario.model()
                    key: Final = scenario.key(models=[model])
                    for generation in ("before", "after"):
                        with Redis(host=cache.host, port=cache.port, socket_timeout=1) as client:
                            eventually(client.ping, bool)
                            eventually(lambda: client.pubsub_numsub("litellm_proxy.auth_cache_invalidation")[0][1], lambda count: count >= 1, seconds=8)
                        upstream.get("/__observations").raise_for_status()
                        first: Final = candidate.chat(model, key=key, text=identity + generation)
                        second: Final = candidate.chat(model, key=key, text=identity + generation)
                        assert first["id"] == second["id"]
                        assert first["choices"] == second["choices"] and first["usage"]["total_tokens"] == 40
                        assert len(upstream.get("/__observations").json()["requests"]) == 1
                        with Redis(host=cache.host, port=cache.port, socket_timeout=1) as client:
                            eventually(
                                lambda first=first: tuple(client.get(name) for name in client.scan_iter() if client.type(name) == b"string"),
                                lambda values, first=first: any(str(first["id"]).encode() in value for value in values if value is not None),
                                seconds=10,
                            )
                        if generation == "before":
                            cache.stop()
                            upstream.get("/__observations").raise_for_status()
                            during: Final = candidate.chat(model, key=key, text=identity + "during")
                            assert during["usage"]["total_tokens"] == 40
                            assert len(upstream.get("/__observations").json()["requests"]) == 1
                            cache.start()
                    with psycopg.connect(database_url) as fresh:
                        assert fresh.execute('SELECT count(*) FROM "LiteLLM_VerificationToken"').fetchone()[0] >= 1
        finally:
            admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(identity)))
        assert admin.execute("SELECT datname FROM pg_database WHERE datname=%s", (identity,)).fetchall() == []
