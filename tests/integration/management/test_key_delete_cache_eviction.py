"""
`/key/delete` cache eviction across both proxies: a key the primary deletes must stop authenticating
on the peer at once instead of until the peer's in-memory copy of the key object expires.

Auth caches the key object under its sha256 in memory and, with `enable_redis_auth_cache`, in Redis;
the pubsub channel carries the delete to the peer, whose next request misses locally, re-reads the
db and answers 401 for the missing row.
"""

import os
import uuid
from hashlib import sha256
from typing import Final

import httpx
from redis import Redis

from tests.integration._support.client import Gateway, delete_key_if_present, eventually, object_value, string_value
from tests.integration._support.database import read_rows


def _redis() -> Redis:
    return Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))


def _chat(proxy: Gateway, model: str, key: str, text: str) -> httpx.Response:
    return proxy.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": text}]}, key=key
    )


def test_key_delete_stops_the_key_on_the_peer_before_its_cache_ttl(gateway: Gateway, peer: Gateway) -> None:
    """The peer warmed the key into its own memory; the primary's delete has to reach it over pubsub so
    the peer's next request with that key is a 401, well inside the 60 s in-memory TTL."""
    with gateway.scenario() as scenario, _redis() as cache:
        model: Final = scenario.model()
        key: Final = string_value(gateway.post("/key/generate", {"models": [model]})["key"])
        scenario.cleanups.callback(delete_key_if_present, gateway, key)
        digest: Final = sha256(key.encode()).hexdigest()
        warm: Final = _chat(peer, model, key, "warm key on peer " + uuid.uuid4().hex)
        assert warm.status_code == 200, warm.text
        eventually(lambda: cache.get(digest), lambda cached: cached is not None, seconds=10)

        deleted: Final = gateway.post("/key/delete", {"keys": [key]})
        assert deleted["deleted_keys"] == [key], deleted
        assert read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE token = %s', (digest,)) == []

        denied: Final = eventually(
            lambda: _chat(peer, model, key, "deleted key on peer " + uuid.uuid4().hex),
            lambda response: response.status_code == 401,
            seconds=10,
            return_last_on_timeout=True,
        )
        assert denied.status_code == 401, denied.text
        assert object_value(object_value(denied.json())["error"])["type"] == "token_not_found_in_db", denied.text
        assert cache.get(digest) is None
