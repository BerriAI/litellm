"""Rate limits on request tags: tag objects (``/tag/new`` rpm_limit, tpm_limit) and a key's ``tag_rpm_limit``.

The contract, read from ``litellm/proxy/hooks/parallel_request_limiter_v3.py``:

- A tag created with ``rpm_limit`` or ``tpm_limit`` gets one window shared by every caller that sends the tag
  (descriptor ``tag``, value the tag name, Redis ``{tag:<name>}:requests|tokens``). Tags are read from
  ``metadata.tags`` and the top-level ``tags`` field together, de-duplicated in order
  (``get_tags_from_request_body`` and ``dict.fromkeys`` in ``_create_tag_rate_limit_descriptors``). A tag whose
  budget has neither limit creates no descriptor.
- A key's ``metadata.tag_rpm_limit={<tag>: N}`` gives that key its own per-tag request window (descriptor
  ``tag_per_key``, value ``<key hash>:<tag>``), requests only.
- Served responses carry ``x-ratelimit-tag-*`` and ``x-ratelimit-tag_per_key-*`` headers.

The Python limiter resolves tag limits through the database (``resolve_tag_rate_limits_from_db``); the Rust port
(#51537) keeps that lookup in Python and builds the descriptor in Rust, so the names and keys here are the seam.
"""

from __future__ import annotations

from typing import Final

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limits_support import (
    assert_limited,
    chat,
    hashed,
    posts,
    provider,
    scope_headers,
    statuses,
)
from tests.integration.authorization.dec1_ratelim_support import (
    redis_counters,
    requests_counter,
    tag,
    tagged_chat,
    tokens_counter,
)


def test_a_tags_request_window_is_shared_by_two_keys(gateway: Gateway) -> None:
    """A tag has rpm_limit=2 and two unlimited keys each send one tagged request. A third tagged request must be
    refused with ``Rate limit exceeded for tag: <tag>. Limit type: requests. Current limit: 2, Remaining: 0``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        name: Final = tag(gateway, scenario, rpm_limit=2)
        first: Final = scenario.key(models=[model])
        second: Final = scenario.key(models=[model])
        assert tagged_chat(gateway, model, first, metadata_tags=(name,)).status_code == 200
        assert tagged_chat(gateway, model, second, metadata_tags=(name,)).status_code == 200
        assert_limited(tagged_chat(gateway, model, first, metadata_tags=(name,)), "tag", name, "requests", 2)
        assert len(posts(wire)) == 2


def test_the_top_level_tags_field_is_charged_like_metadata_tags(gateway: Gateway) -> None:
    """A tag has rpm_limit=1. One request carries it only in the top-level ``tags`` field; a second carries it
    only in ``metadata.tags``. The second must be refused for ``tag: <tag>``: both fields feed one window."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        name: Final = tag(gateway, scenario, rpm_limit=1)
        key: Final = scenario.key(models=[model])
        assert tagged_chat(gateway, model, key, body_tags=(name,)).status_code == 200
        assert_limited(tagged_chat(gateway, model, key, metadata_tags=(name,)), "tag", name, "requests", 1)


def test_a_tag_named_twice_in_one_request_is_charged_once(gateway: Gateway) -> None:
    """A tag has rpm_limit=2. One request names it in both ``metadata.tags`` and ``tags``; the Redis counter
    ``{tag:<tag>}:requests`` must then read 1 and a second request must still be served."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario, redis_counters() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        name: Final = tag(gateway, scenario, rpm_limit=2)
        key: Final = scenario.key(models=[model])
        assert tagged_chat(gateway, model, key, metadata_tags=(name,), body_tags=(name,)).status_code == 200
        assert requests_counter(redis, "tag", name) == 1
        assert tagged_chat(gateway, model, key, metadata_tags=(name,)).status_code == 200


def test_a_request_with_two_limited_tags_is_charged_to_both(gateway: Gateway) -> None:
    """Tags A and B each have rpm_limit=5. One request carrying both must leave ``{tag:A}:requests`` and
    ``{tag:B}:requests`` at 1 each."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario, redis_counters() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        first: Final = tag(gateway, scenario, rpm_limit=5)
        second: Final = tag(gateway, scenario, rpm_limit=5)
        key: Final = scenario.key(models=[model])
        assert tagged_chat(gateway, model, key, metadata_tags=(first, second)).status_code == 200
        assert (requests_counter(redis, "tag", first), requests_counter(redis, "tag", second)) == (1, 1)


def test_an_exhausted_tag_refuses_a_request_that_also_carries_a_free_tag(gateway: Gateway) -> None:
    """Tag A has rpm_limit=1 and has been used; tag B has rpm_limit=5. A request carrying both A and B must be
    refused for ``tag: <A>``: every tag on the request must have room."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        exhausted: Final = tag(gateway, scenario, rpm_limit=1)
        free: Final = tag(gateway, scenario, rpm_limit=5)
        key: Final = scenario.key(models=[model])
        assert tagged_chat(gateway, model, key, metadata_tags=(exhausted,)).status_code == 200
        assert_limited(
            tagged_chat(gateway, model, key, metadata_tags=(free, exhausted)), "tag", exhausted, "requests", 1
        )


def test_a_served_tagged_response_reports_the_tag_window(gateway: Gateway) -> None:
    """A tag has rpm_limit=3 and tpm_limit=1000; one tagged request asks for max_tokens=20 (estimate 30). The 200
    must carry ``x-ratelimit-tag-limit-requests: 3``, ``-remaining-requests: 2``, ``-limit-tokens: 1000`` and
    ``-remaining-tokens: 970``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        name: Final = tag(gateway, scenario, rpm_limit=3, tpm_limit=1000)
        key: Final = scenario.key(models=[model])
        served: Final = tagged_chat(gateway, model, key, metadata_tags=(name,), max_tokens=20)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "tag") == {
            "limit-requests": "3",
            "remaining-requests": "2",
            "limit-tokens": "1000",
            "remaining-tokens": "970",
        }


def test_a_tags_tpm_limit_refuses_a_request_too_large_for_it_before_the_provider(gateway: Gateway) -> None:
    """A tag has tpm_limit=20 and a tagged request asks for max_tokens=50 (estimate 60). It must be refused with
    ``Rate limit exceeded for tag: <tag>. Limit type: tokens. Current limit: 20, Remaining: 20`` and never reach
    the provider: tag token windows take part in the upfront reservation."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        name: Final = tag(gateway, scenario, tpm_limit=20)
        key: Final = scenario.key(models=[model])
        assert_limited(
            tagged_chat(gateway, model, key, metadata_tags=(name,), max_tokens=50), "tag", name, "tokens", 20, 20
        )
        assert posts(wire) == ()


def test_a_tags_token_window_settles_to_the_reported_usage(gateway: Gateway) -> None:
    """A tag has tpm_limit=10000; a tagged request reserves an estimate and the provider reports 11 prompt plus 7
    completion tokens. ``{tag:<tag>}:tokens`` must settle to exactly 18."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario, redis_counters() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        name: Final = tag(gateway, scenario, tpm_limit=10000)
        key: Final = scenario.key(models=[model])
        assert tagged_chat(gateway, model, key, metadata_tags=(name,), max_tokens=40).status_code == 200
        assert eventually(lambda: tokens_counter(redis, "tag", name), lambda tokens: tokens == 18, seconds=20) == 18


def test_a_tag_with_only_a_budget_is_not_rate_limited(gateway: Gateway) -> None:
    """A tag created with only ``max_budget=100`` is sent on three requests. All must be served and no
    ``x-ratelimit-tag-*`` header may appear, because a tag budget without rpm or tpm creates no descriptor."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        name: Final = tag(gateway, scenario, max_budget=100)
        key: Final = scenario.key(models=[model])
        responses: Final = [tagged_chat(gateway, model, key, metadata_tags=(name,)) for _ in range(3)]
        assert statuses(responses) == [200, 200, 200]
        assert scope_headers(responses[-1], "tag") == {}


def test_a_request_without_the_tag_is_not_charged_to_it(gateway: Gateway) -> None:
    """A tag has rpm_limit=1. An untagged request from the same key must leave ``{tag:<tag>}:requests`` absent and
    the next tagged request must still be served."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario, redis_counters() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        name: Final = tag(gateway, scenario, rpm_limit=1)
        key: Final = scenario.key(models=[model])
        assert chat(gateway, model, key).status_code == 200
        assert requests_counter(redis, "tag", name) is None
        assert tagged_chat(gateway, model, key, metadata_tags=(name,)).status_code == 200


def test_a_keys_per_tag_window_is_not_shared_with_another_key(gateway: Gateway) -> None:
    """Two keys each have ``tag_rpm_limit={<tag>: 1}``. After key one is refused on its second tagged request, key
    two's first tagged request must still be served: the window is ``<key hash>:<tag>``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        name: Final = "ratelim-shared-tag-name"
        first: Final = scenario.key(models=[model], metadata={"tag_rpm_limit": {name: 1}})
        second: Final = scenario.key(models=[model], metadata={"tag_rpm_limit": {name: 1}})
        assert tagged_chat(gateway, model, first, metadata_tags=(name,)).status_code == 200
        assert_limited(
            tagged_chat(gateway, model, first, metadata_tags=(name,)),
            "tag_per_key",
            f"{hashed(first)}:{name}",
            "requests",
            1,
        )
        assert tagged_chat(gateway, model, second, metadata_tags=(name,)).status_code == 200


def test_a_keys_per_tag_limit_reports_only_request_headers(gateway: Gateway) -> None:
    """A key has ``tag_rpm_limit={<tag>: 4}``. A tagged request must carry exactly
    ``x-ratelimit-tag_per_key-limit-requests: 4`` and ``-remaining-requests: 3`` and no token header for that
    scope, because ``tag_per_key`` descriptors have ``tokens_per_unit=None``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        name: Final = "ratelim-header-tag"
        key: Final = scenario.key(models=[model], metadata={"tag_rpm_limit": {name: 4}})
        served: Final = tagged_chat(gateway, model, key, metadata_tags=(name,))
        assert served.status_code == 200, served.text
        assert scope_headers(served, "tag_per_key") == {"limit-requests": "4", "remaining-requests": "3"}


def test_a_keys_per_tag_limit_ignores_untagged_requests(gateway: Gateway) -> None:
    """A key has ``tag_rpm_limit={<tag>: 1}`` and no key-wide limit. Three untagged requests must all be served:
    tags without a matching entry fall through to the key-level descriptor, which here is absent."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], metadata={"tag_rpm_limit": {"ratelim-only-this-tag": 1}})
        assert statuses([chat(gateway, model, key) for _ in range(3)]) == [200, 200, 200]


def test_a_tag_limit_raised_by_tag_update_admits_the_tag_again(gateway: Gateway) -> None:
    """A tag with rpm_limit=1 has been used and refused. After ``POST /tag/update`` sets rpm_limit=100, a tagged
    request must eventually be served again in the same window. The new limit is 100 so refused polls (each of
    which still bumps the counter) cannot use it up. UNVERIFIED how fast: the tag object cache is invalidated on
    update (``spend/test_tag_budget_enforcement.py`` relies on it), so this polls for 10 seconds."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        name: Final = tag(gateway, scenario, rpm_limit=1)
        key: Final = scenario.key(models=[model])
        assert tagged_chat(gateway, model, key, metadata_tags=(name,)).status_code == 200
        assert tagged_chat(gateway, model, key, metadata_tags=(name,)).status_code == 429
        gateway.post("/tag/update", {"name": name, "rpm_limit": 100})
        eventually(
            lambda: tagged_chat(gateway, model, key, metadata_tags=(name,)).status_code,
            lambda status: status == 200,
        )
