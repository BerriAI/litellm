"""The refusal answered from headers alone (#51428) versus the refusal answered after the body is parsed.

The contract, read from ``litellm/proxy/auth/user_api_key_auth.py`` (the pre-body check around line 2280) and
``raise_if_over_limit_before_body`` in ``litellm/proxy/hooks/parallel_request_limiter_v3.py``:

- Once a key is in the auth cache, auth builds the limiter's descriptors from the key alone (``data={}``) and
  calls ``should_rate_limit(read_only=True, skip_tpm_check=<reservation enabled>)``. A window whose counter plus
  one exceeds its limit, or a concurrency gauge that is full, is refused with the normal v3 429 before the body
  is read. Nothing is incremented.
- Limits that need the body are NOT checked there: per-model limits (``model_per_*``), tag limits (tags live in
  the body), and, with token reservation on (the default), token windows. Those requests are read in full and
  refused by ``async_pre_call_hook`` instead.
- Unbilled routes (model lists, ``/utils/token_counter``, any ``count_tokens`` path) and body-less GETs skip the
  check.

The Rust port (#51537) moves this "body-free check" into ``scoped.rs``; a port that checks too much refuses
requests that fit, and one that checks too little buffers every refused payload again.

A request the proxy keeps waiting on is reported as ``NO_ANSWER`` by the raw-socket helper.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Final

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limits_support import (
    anthropic_messages,
    assert_limited,
    chat,
    embed,
    hashed,
    posts,
    provider,
)
from tests.integration.authorization.dec1_ratelim_support import (
    NO_ANSWER,
    lower_headers,
    raw_answer_before_body_completes,
    redis_counters,
    requests_counter,
    status_before_body_completes,
    tag,
    tagged_chat,
    tokens_counter,
)


def test_a_full_parallel_slot_is_refused_before_the_body_arrives(gateway: Gateway) -> None:
    """A key with max_parallel_requests=1 has one request held at the provider. A second request that sends only
    1 KB of a declared 8 MB body must already be answered 429: the gauge check runs read-only in auth."""
    release: Final = threading.Event()
    with (
        wire_server(provider(hold=release)) as wire,
        gateway.scenario() as scenario,
        ThreadPoolExecutor(1) as pool,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], max_parallel_requests=1)
        held: Final = pool.submit(chat, gateway, model, key)
        try:
            eventually(wire.received.qsize, lambda count: count == 1)
            assert status_before_body_completes(gateway, key, model=model) == 429
        finally:
            release.set()
        assert held.result().status_code == 200


def test_an_exhausted_team_window_is_refused_before_the_body_arrives(gateway: Gateway) -> None:
    """A team with rpm_limit=1 has one key that has used its one request. The next request that sends only part of
    its body must be answered 429, because the team descriptor is built from the cached key alone."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], rpm_limit=1)
        key: Final = scenario.key(team_id=team)
        assert chat(gateway, model, key).status_code == 200
        assert status_before_body_completes(gateway, key, model=model) == 429


def test_an_exhausted_user_window_is_refused_before_the_body_arrives(gateway: Gateway) -> None:
    """A user with rpm_limit=1 owns a key that has used the one request. The next partial-body request must be
    answered 429 from headers alone."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        user: Final = scenario.user(rpm_limit=1)
        key: Final = scenario.key(models=[model], user_id=user)
        assert chat(gateway, model, key).status_code == 200
        assert status_before_body_completes(gateway, key, model=model) == 429


def test_the_pre_body_refusal_carries_the_limiter_headers_and_message(gateway: Gateway) -> None:
    """A key with rpm_limit=1 has used its request; a partial-body request is refused. The raw answer must be a 429
    with ``rate_limited_by: api_key``, ``rate_limit_type: requests``, ``retry-after: 60`` and a body naming
    ``Rate limit exceeded for api_key: <hash>``, the same fields the post-body refusal has."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert chat(gateway, model, key).status_code == 200
        status, answer = raw_answer_before_body_completes(gateway, key)
        assert status == 429, answer
        headers: Final = lower_headers(answer)
        assert headers["rate_limited_by"] == "api_key", headers
        assert headers["rate_limit_type"] == "requests", headers
        assert headers["retry-after"] == "60", headers
        assert f"Rate limit exceeded for api_key: {hashed(key)}." in answer, answer


def test_a_per_model_limit_is_not_checked_before_the_body_arrives(gateway: Gateway) -> None:
    """A key with ``model_rpm_limit={<model>: 1}`` has used its one request for that model. A partial-body request
    must get NO answer from headers alone (the model is in the body); the same request sent in full must then be
    refused as ``model_per_key: <hash>:<model>``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], model_rpm_limit={model: 1})
        assert chat(gateway, model, key).status_code == 200
        assert status_before_body_completes(gateway, key, model=model) == NO_ANSWER
        assert_limited(chat(gateway, model, key), "model_per_key", f"{hashed(key)}:{model}", "requests", 1)


def test_a_tag_limit_is_not_checked_before_the_body_arrives(gateway: Gateway) -> None:
    """A tag with rpm_limit=1 has been used once by a key without limits of its own. A partial-body request with
    that key must get NO answer from headers alone, since tags are read from the body; the full tagged request is
    then refused as ``tag: <tag>``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        name: Final = tag(gateway, scenario, rpm_limit=1)
        key: Final = scenario.key(models=[model])
        assert tagged_chat(gateway, model, key, metadata_tags=(name,)).status_code == 200
        assert status_before_body_completes(gateway, key, model=model) == NO_ANSWER
        assert_limited(tagged_chat(gateway, model, key, metadata_tags=(name,)), "tag", name, "requests", 1)


def test_an_exhausted_token_window_is_not_checked_before_the_body_arrives(gateway: Gateway) -> None:
    """A key with tpm_limit=100 and rpm_limit=100 is held at a token window with no room (a request that reserved
    and settled 100 tokens: provider usage 60 prompt plus 40 completion). With reservation on (the default) auth
    passes ``skip_tpm_check=True``, so a partial-body request must get NO answer; the full request is then refused
    for ``tokens`` by the pre-call reservation."""
    from tests.integration.authorization.dec1_limits_support import Usage

    with wire_server(provider(Usage(prompt=60, completion=40))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=100, rpm_limit=100)
        assert chat(gateway, model, key, max_tokens=5).status_code == 200
        with redis_counters() as redis:
            eventually(lambda: tokens_counter(redis, "api_key", hashed(key)), lambda tokens: tokens == 100, seconds=20)
        assert status_before_body_completes(gateway, key, model=model) == NO_ANSWER
        refused: Final = chat(gateway, model, key, max_tokens=5)
        assert refused.status_code == 429, refused.text
        assert refused.headers["rate_limit_type"] == "tokens"


def test_pre_body_refusals_do_not_increment_the_team_window(gateway: Gateway) -> None:
    """A team with rpm_limit=1 has used its request; three more partial-body requests are refused in auth. The
    Redis counter ``{team:<team_id>}:requests`` must still read 1, because the pre-body check is read-only."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario, redis_counters() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], rpm_limit=1)
        key: Final = scenario.key(team_id=team)
        assert chat(gateway, model, key).status_code == 200
        assert [status_before_body_completes(gateway, key, model=model) for _ in range(3)] == [429, 429, 429]
        assert requests_counter(redis, "team", team) == 1


def test_a_pre_body_refusal_takes_no_parallel_slot(gateway: Gateway) -> None:
    """A key with rpm_limit=1 and max_parallel_requests=5 has used its request. A partial-body refusal must leave
    the sorted set ``{api_key:<hash>}:max_parallel_requests`` empty: a read-only gauge check never registers a
    member."""
    from tests.integration.authorization.dec1_ratelim_support import in_flight

    with wire_server(provider()) as wire, gateway.scenario() as scenario, redis_counters() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1, max_parallel_requests=5)
        assert chat(gateway, model, key).status_code == 200
        eventually(lambda: in_flight(redis, "api_key", hashed(key)), lambda members: members == 0)
        assert status_before_body_completes(gateway, key, model=model) == 429
        assert in_flight(redis, "api_key", hashed(key)) == 0


def test_an_embeddings_request_is_refused_before_its_body_arrives(gateway: Gateway) -> None:
    """A key with rpm_limit=1 has used its request on ``/v1/embeddings``. A partial-body POST to
    ``/v1/embeddings`` must be answered 429 from headers alone: the check is route-agnostic for billed routes."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert embed(gateway, model, key).status_code == 200
        assert status_before_body_completes(gateway, key, path="/v1/embeddings", model=model) == 429


def test_a_messages_request_is_refused_before_its_body_arrives(gateway: Gateway) -> None:
    """A key with rpm_limit=1 has used its request on ``/v1/messages``. A partial-body POST to ``/v1/messages``
    must be answered 429 from headers alone. UNVERIFIED for the Rust-served messages route (#51119): the check
    lives in Python auth, which that route is expected to keep calling."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert anthropic_messages(gateway, model, key).status_code == 200
        assert status_before_body_completes(gateway, key, path="/v1/messages", model=model) == 429


def test_a_count_tokens_route_is_never_refused_before_its_body(gateway: Gateway) -> None:
    """A key with rpm_limit=1 is exhausted. A POST to ``/v1/messages/count_tokens`` must not be refused with 429,
    because any path whose last segment is ``count_tokens`` is unbilled (``RouteChecks.is_unbilled_route``)."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert chat(gateway, model, key).status_code == 200
        assert chat(gateway, model, key).status_code == 429
        counted: Final = gateway.request(
            "POST",
            "/v1/messages/count_tokens",
            {"model": model, "messages": [{"role": "user", "content": "count"}]},
            key=key,
            headers={"anthropic-version": "2023-06-01"},
        )
        assert counted.status_code != 429, counted.text
        assert len(posts(wire)) <= 2


def test_the_post_body_refusal_carries_a_call_id_that_the_pre_body_refusal_lacks(gateway: Gateway) -> None:
    """A key has ``model_rpm_limit={<model>: 1}`` and has used it, so the next full request is refused after its
    body by ``async_pre_call_hook``. That 429 must carry a non-empty ``x-litellm-call-id``. Contrast: the
    pre-body 429 has none (the addendum and ``test_dec1_limits_key_rpm.py`` pin that as a bug). UNVERIFIED: read
    from the post-call failure path in ``litellm/proxy/common_request_processing.py`` adding the call id to
    error responses."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], model_rpm_limit={model: 1})
        assert chat(gateway, model, key).status_code == 200
        refused: Final = chat(gateway, model, key)
        assert refused.status_code == 429, refused.text
        assert refused.headers.get("x-litellm-call-id"), dict(refused.headers)
