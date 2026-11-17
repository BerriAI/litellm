"""The Redis response cache seen from two proxy instances: an answer cached by one pod is a hit on the other.

Two instances run the same `litellm_settings.cache: true` with `cache_params.type: redis` on one Redis. An entry
is stored under the bare sha256 hex digest of the request's cache key, or `<namespace>:<digest>` when
`cache_params.namespace` (or a request's `cache.namespace`) is set, as JSON holding the response and a
timestamp, with `EX` the request's `cache.ttl` or `cache_params.ttl`. A hit never reaches the provider and
carries `x-litellm-cache-key` naming the entry. `cache: {"no-store": true}` leaves no entry, `cache:
{"no-cache": true}` skips the lookup, and `POST /cache/delete {"keys": [...]}` through either instance removes
the entry for both.

The single-instance versions of these contracts live in the `test_dec1_cache_*` files; here the second pod is
the observer, which is what breaks when a rewrite changes the key derivation, the namespace prefix, the value
encoding or the TTL on one side of a rolling deploy. Read from `litellm/caching/caching.py`,
`litellm/caching/redis_cache.py` and `litellm/rust/crates/cache-response/src/caching.rs:92`.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, eventually, object_value
from tests.integration.routing.dec1_multiinst_support import CACHE_KEY_HEADER, Fleet, ask, fleet, served, unique
from tests.integration.routing.dec1_routing_support import deployment

pytestmark: Final = [pytest.mark.timeout(400), pytest.mark.shares_worker_state]

NAMESPACE: Final = "dec1-multiinst-ns"
DEFAULT_TTL_SECONDS: Final = 45
SHORT_TTL_SECONDS: Final = 2
DIGEST: Final = re.compile(r"[0-9a-f]{64}")


def _models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment("cache-chat", "ok-cache-chat", api_base),
        deployment("cache-embed", "ok-cache-embed", api_base),
    )


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Fleet]:
    with fleet(
        tmp_path_factory.mktemp("dec1-multiinst-cache"),
        models=_models,
        router_settings={"num_retries": 0, "disable_cooldowns": True},
        cache_params={"namespace": NAMESPACE, "ttl": DEFAULT_TTL_SECONDS},
    ) as value:
        yield value


def _id(response: httpx.Response) -> str:
    assert response.status_code == 200, response.text
    return str(JSON_OBJECT.validate_json(response.content)["id"])


def _entry_naming(pair: Fleet, needle: str) -> str:
    """The Redis key whose stored value names `needle` (a response id), polled because the write is asynchronous."""

    def found() -> tuple[str, ...]:
        with pair.store() as client:
            return tuple(
                raw.decode()
                for raw in client.scan_iter(count=1000)
                if client.type(raw) == b"string" and needle.encode() in (client.get(raw) or b"")
            )

    keys: Final = eventually(found, lambda matches: len(matches) >= 1, seconds=10)
    return keys[0]


def test_an_answer_cached_by_one_instance_is_a_hit_on_the_other(pair: Fleet) -> None:
    """The first instance answers a chat request for `cache-chat` from the upstream. The same request through the
    second instance must return 200 with the same `id` and content, and the upstream must have received exactly
    one request for `ok-cache-chat` for that text.
    """
    text: Final = unique("shared hit")
    before: Final = pair.upstream.count("ok-cache-chat")
    filled: Final = ask(pair.first, "cache-chat", text)
    _entry_naming(pair, _id(filled))
    hit: Final = ask(pair.second, "cache-chat", text)
    assert _id(hit) == _id(filled)
    assert served(hit) == "ok-cache-chat"
    assert pair.upstream.count("ok-cache-chat") - before == 1


def test_the_peers_hit_names_the_namespaced_key_the_filling_instance_wrote(pair: Fleet) -> None:
    """`cache_params.namespace` is `dec1-multiinst-ns` on both instances. After the first instance fills an entry,
    the Redis key holding it must be `dec1-multiinst-ns:` followed by 64 hex characters, and the second instance's
    hit must carry `x-litellm-cache-key` equal to that full key.
    """
    text: Final = unique("namespaced hit")
    filled: Final = ask(pair.first, "cache-chat", text)
    key: Final = _entry_naming(pair, _id(filled))
    namespace, _, digest = key.partition(":")
    assert namespace == NAMESPACE
    assert DIGEST.fullmatch(digest), key
    assert ask(pair.second, "cache-chat", text).headers.get(CACHE_KEY_HEADER) == key


def test_an_entry_written_by_one_instance_expires_after_the_configured_default_ttl(pair: Fleet) -> None:
    """`cache_params.ttl: 45`. After the first instance fills an entry, its Redis TTL must be between 1 and 45
    seconds. A rewrite that drops the configured TTL leaves entries for the backend default instead.
    """
    filled: Final = ask(pair.first, "cache-chat", unique("default ttl"))
    key: Final = _entry_naming(pair, _id(filled))
    assert 0 < pair.ttl(key) <= DEFAULT_TTL_SECONDS


def test_a_request_ttl_on_one_instance_makes_the_entry_expire_for_the_peer(pair: Fleet) -> None:
    """The first instance fills an entry with `cache: {"ttl": 2}`. After the Redis key is gone, the same request
    through the second instance must reach the upstream again (a second upstream call, and no
    `x-litellm-cache-key` header on the answer).
    """
    text: Final = unique("short ttl")
    before: Final = pair.upstream.count("ok-cache-chat")
    filled: Final = ask(pair.first, "cache-chat", text, cache={"ttl": SHORT_TTL_SECONDS})
    key: Final = _entry_naming(pair, _id(filled))
    eventually(lambda: pair.raw(key), lambda raw: raw is None, seconds=10)
    again: Final = ask(pair.second, "cache-chat", text, cache={"ttl": SHORT_TTL_SECONDS})
    assert again.status_code == 200
    assert CACHE_KEY_HEADER not in again.headers
    assert pair.upstream.count("ok-cache-chat") - before == 2


def test_no_store_on_one_instance_leaves_nothing_for_the_peer(pair: Fleet) -> None:
    """The first instance answers a request sent with `cache: {"no-store": true}`. The same request without controls
    through the second instance must reach the upstream (two upstream calls in total) and carry no
    `x-litellm-cache-key` header.
    """
    text: Final = unique("no store")
    before: Final = pair.upstream.count("ok-cache-chat")
    assert ask(pair.first, "cache-chat", text, cache={"no-store": True}).status_code == 200
    second: Final = ask(pair.second, "cache-chat", text)
    assert second.status_code == 200
    assert CACHE_KEY_HEADER not in second.headers
    assert pair.upstream.count("ok-cache-chat") - before == 2


def test_no_cache_on_the_peer_skips_the_entry_the_other_instance_wrote(pair: Fleet) -> None:
    """The first instance fills an entry. The same request through the second instance with
    `cache: {"no-cache": true}` must reach the upstream and return a different `id` than the cached answer.
    """
    text: Final = unique("no cache")
    filled: Final = ask(pair.first, "cache-chat", text)
    _entry_naming(pair, _id(filled))
    fresh: Final = ask(pair.second, "cache-chat", text, cache={"no-cache": True})
    assert _id(fresh) != _id(filled)


def test_cache_delete_through_one_instance_evicts_the_entry_for_the_other(pair: Fleet) -> None:
    """The first instance fills an entry and the second gets a hit on it. `POST /cache/delete` through the second
    instance with `{"keys": [<entry key>]}` must answer 200, the Redis key must be gone, and the same request
    through the first instance must then reach the upstream again with a new `id`.
    """
    text: Final = unique("delete")
    filled: Final = ask(pair.first, "cache-chat", text)
    key: Final = _entry_naming(pair, _id(filled))
    assert _id(ask(pair.second, "cache-chat", text)) == _id(filled)
    pair.second.post("/cache/delete", {"keys": [key]})
    assert pair.raw(key) is None
    assert _id(ask(pair.first, "cache-chat", text)) != _id(filled)


def test_an_embedding_cached_by_one_instance_is_a_hit_on_the_other(pair: Fleet) -> None:
    """The first instance answers `POST /v1/embeddings` for `cache-embed` with a fixed input from the upstream. The
    same request through the second instance must return 200 with the same vector and carry
    `x-litellm-cache-key`, and the upstream must have received one request for `ok-cache-embed` for that input.
    UNVERIFIED: that an embedding hit carries the cache key header; the single-instance file asserts the vector and
    zero cost.
    """
    text: Final = unique("embed input")
    before: Final = pair.upstream.count("ok-cache-embed")
    body: Final[dict[str, JsonValue]] = {"model": "cache-embed", "input": text}
    first: Final = pair.first.request("POST", "/v1/embeddings", body)
    assert first.status_code == 200, first.text
    vector: Final = object_value(JSON_OBJECT.validate_json(first.content)["data"][0])["embedding"]  # type: ignore[index]
    second: Final = eventually(
        lambda: pair.second.request("POST", "/v1/embeddings", body),
        lambda response: CACHE_KEY_HEADER in response.headers,
        seconds=10,
    )
    assert object_value(JSON_OBJECT.validate_json(second.content)["data"][0])["embedding"] == vector  # type: ignore[index]
    assert pair.upstream.count("ok-cache-embed") - before == 1


def test_a_streamed_answer_cached_by_one_instance_is_replayed_as_a_stream_by_the_other(pair: Fleet) -> None:
    """The first instance streams a chat answer for `cache-chat` (`stream: true`). The same streaming request through
    the second instance must return 200 server-sent events that contain `served ok-cache-chat` and end with
    `data: [DONE]`, and the upstream must have received one request for that text.
    """
    text: Final = unique("stream")
    before: Final = pair.upstream.count("ok-cache-chat")
    body: Final[dict[str, JsonValue]] = {
        "model": "cache-chat",
        "stream": True,
        "messages": [{"role": "user", "content": text}],
    }

    def stream(target: Fleet, which: int) -> tuple[int, str]:
        gateway = target.instances[which]
        with gateway.client.stream(
            "POST", "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {gateway.key}"}
        ) as response:
            return response.status_code, response.read().decode()

    first_status, first_events = stream(pair, 0)
    assert first_status == 200
    stream_id: Final = re.search(r'"id":\s*"(chatcmpl-[0-9a-f]+)"', first_events)
    assert stream_id is not None, first_events
    _entry_naming(pair, stream_id.group(1))
    status, events = stream(pair, 1)
    assert status == 200
    assert "served ok-cache-chat" in events
    assert events.rstrip().endswith("data: [DONE]")
    assert pair.upstream.count("ok-cache-chat") - before == 1


def test_instances_with_different_namespaces_do_not_share_entries(tmp_path: Path) -> None:
    """The first instance runs `cache_params.namespace: dec1-ns-one`, the second `dec1-ns-two`, on one Redis. The
    same request through both must reach the upstream twice, and the Redis keys holding the two answers must start
    with `dec1-ns-one:` and `dec1-ns-two:` respectively. Namespaces are how two fleets share one Redis safely.
    """
    with fleet(
        tmp_path,
        models=_models,
        router_settings={"num_retries": 0, "disable_cooldowns": True},
        cache_params={"namespace": "dec1-ns-one"},
        second_cache_params={"namespace": "dec1-ns-two"},
    ) as split:
        text: Final = unique("split namespaces")
        first: Final = ask(split.first, "cache-chat", text)
        second: Final = ask(split.second, "cache-chat", text)
        assert _id(first) != _id(second)
        assert split.upstream.count("ok-cache-chat") == 2
        assert _entry_naming(split, _id(first)).startswith("dec1-ns-one:")
        assert _entry_naming(split, _id(second)).startswith("dec1-ns-two:")


def test_a_request_namespace_on_one_instance_is_hit_by_the_same_namespace_on_the_peer(pair: Fleet) -> None:
    """The first instance fills an entry with `cache: {"namespace": "dec1-request-ns"}`. The same request with the
    same namespace through the second instance must be a hit (same `id`) whose `x-litellm-cache-key` starts with
    `dec1-request-ns:`, while the same text without the namespace must be a miss.
    """
    text: Final = unique("request namespace")
    filled: Final = ask(pair.first, "cache-chat", text, cache={"namespace": "dec1-request-ns"})
    _entry_naming(pair, _id(filled))
    hit: Final = ask(pair.second, "cache-chat", text, cache={"namespace": "dec1-request-ns"})
    assert _id(hit) == _id(filled)
    assert str(hit.headers.get(CACHE_KEY_HEADER, "")).startswith("dec1-request-ns:")
    assert _id(ask(pair.second, "cache-chat", text)) != _id(filled)
