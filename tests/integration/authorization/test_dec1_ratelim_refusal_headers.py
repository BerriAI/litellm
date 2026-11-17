"""Which limit refused a request: the ``rate_limited_by`` header and its siblings on every scope's 429.

The contract: every refusal raised by the v3 limiter (``_handle_rate_limit_error`` in
``litellm/proxy/hooks/parallel_request_limiter_v3.py``) is a 429 whose body says ``Rate limit exceeded for
<scope>: <value>`` and whose headers carry exactly four limiter fields: ``retry-after`` (the window, 60),
``rate_limit_type`` (``requests``, ``tokens`` or ``max_parallel_requests``), ``reset_at`` (``YYYY-MM-DD
HH:MM:SS UTC``) and ``rate_limited_by`` (the scope word, #51228). Clients and dashboards switch on
``rate_limited_by`` to tell a key limit from a team, tag or model limit, so the Rust port (#51537, ``refusal.rs``)
must emit the same scope word for every descriptor kind. A served response never carries ``rate_limited_by``.

The ``test_dec1_limits_*`` files pin the message text; this file pins the header for each scope.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Final

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limits_support import RESET_AT, chat, hashed, posts, provider
from tests.integration.authorization.dec1_ratelim_support import assert_refused_by, end_user, tag, tagged_chat


def test_a_key_rpm_refusal_is_attributed_to_api_key(gateway: Gateway) -> None:
    """A key with rpm_limit=1 sends two requests. The second must be a 429 for ``api_key: <hash>`` with
    ``rate_limited_by: api_key``, ``rate_limit_type: requests`` and ``retry-after: 60``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert chat(gateway, model, key).status_code == 200
        assert_refused_by(chat(gateway, model, key), "api_key", hashed(key), "requests", 1)


def test_a_key_tpm_refusal_is_attributed_to_api_key_with_type_tokens(gateway: Gateway) -> None:
    """A key with tpm_limit=20 asks for max_tokens=50 (estimate 60). The 429 must carry ``rate_limited_by:
    api_key`` and ``rate_limit_type: tokens`` and the message must report ``Remaining: 20``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=20)
        assert_refused_by(chat(gateway, model, key, max_tokens=50), "api_key", hashed(key), "tokens", 20, 20)
        assert posts(wire) == ()


def test_a_parallel_refusal_is_attributed_to_api_key_with_type_max_parallel_requests(gateway: Gateway) -> None:
    """A key with max_parallel_requests=1 has one request held at the provider. A second request must be a 429
    with ``rate_limited_by: api_key`` and ``rate_limit_type: max_parallel_requests``."""
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
            assert_refused_by(chat(gateway, model, key), "api_key", hashed(key), "max_parallel_requests", 1)
        finally:
            release.set()
        assert held.result().status_code == 200


def test_a_user_refusal_is_attributed_to_user(gateway: Gateway) -> None:
    """A user with rpm_limit=1 owns a key without limits. The second request must be a 429 for ``user:
    <user_id>`` with ``rate_limited_by: user``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        user: Final = scenario.user(rpm_limit=1)
        key: Final = scenario.key(models=[model], user_id=user)
        assert chat(gateway, model, key).status_code == 200
        assert_refused_by(chat(gateway, model, key), "user", user, "requests", 1)


def test_a_team_refusal_is_attributed_to_team(gateway: Gateway) -> None:
    """A team with rpm_limit=1 has one key. The second request must be a 429 for ``team: <team_id>`` with
    ``rate_limited_by: team``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], rpm_limit=1)
        key: Final = scenario.key(team_id=team)
        assert chat(gateway, model, key).status_code == 200
        assert_refused_by(chat(gateway, model, key), "team", team, "requests", 1)


def test_a_team_member_refusal_is_attributed_to_team_member(gateway: Gateway) -> None:
    """A team member given rpm_limit=1 through ``/team/member_update`` sends two requests with their team key. The
    second must be a 429 for ``team_member: <team_id>:<user_id>`` with ``rate_limited_by: team_member``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        user: Final = scenario.member(team)
        gateway.post("/team/member_update", {"team_id": team, "user_id": user, "rpm_limit": 1})
        key: Final = scenario.key(team_id=team, user_id=user)
        assert chat(gateway, model, key).status_code == 200
        assert_refused_by(chat(gateway, model, key), "team_member", f"{team}:{user}", "requests", 1)


def test_an_end_user_refusal_is_attributed_to_end_user(gateway: Gateway) -> None:
    """An end user whose budget has rpm_limit=1 is named in the ``user`` field of two requests. The second must be
    a 429 for ``end_user: <id>`` with ``rate_limited_by: end_user``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        customer, _ = end_user(gateway, scenario, rpm_limit=1)
        key: Final = scenario.key(models=[model])
        assert chat(gateway, model, key, user=customer).status_code == 200
        assert_refused_by(chat(gateway, model, key, user=customer), "end_user", customer, "requests", 1)


def test_an_organization_refusal_is_attributed_to_organization(gateway: Gateway) -> None:
    """An organization with rpm_limit=1 owns a team whose key sends two requests. The second must be a 429 for
    ``organization: <organization_id>`` with ``rate_limited_by: organization``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        organization: Final = scenario.organization(models=[model], rpm_limit=1)
        team: Final = scenario.team(models=[model], organization_id=organization)
        key: Final = scenario.key(team_id=team)
        assert chat(gateway, model, key).status_code == 200
        assert_refused_by(chat(gateway, model, key), "organization", organization, "requests", 1)


def test_a_per_model_key_refusal_is_attributed_to_model_per_key(gateway: Gateway) -> None:
    """A key has ``model_rpm_limit={<model>: 1}``. The second request for that model must be a 429 for
    ``model_per_key: <hash>:<model>`` with ``rate_limited_by: model_per_key``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], model_rpm_limit={model: 1})
        assert chat(gateway, model, key).status_code == 200
        assert_refused_by(chat(gateway, model, key), "model_per_key", f"{hashed(key)}:{model}", "requests", 1)


def test_a_per_model_team_refusal_is_attributed_to_model_per_team(gateway: Gateway) -> None:
    """A team has ``model_rpm_limit={<model>: 1}`` and two keys. A request on each key: the second must be a 429
    for ``model_per_team: <team_id>:<model>`` with ``rate_limited_by: model_per_team``, because the team window is
    shared while each key's inherited per-key window still has room."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], model_rpm_limit={model: 1})
        first: Final = scenario.key(team_id=team)
        second: Final = scenario.key(team_id=team)
        assert chat(gateway, model, first).status_code == 200
        assert_refused_by(chat(gateway, model, second), "model_per_team", f"{team}:{model}", "requests", 1)


def test_a_tag_refusal_is_attributed_to_tag(gateway: Gateway) -> None:
    """A tag created with rpm_limit=1 is carried in ``metadata.tags`` of two requests. The second must be a 429
    for ``tag: <tag>`` with ``rate_limited_by: tag``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        name: Final = tag(gateway, scenario, rpm_limit=1)
        key: Final = scenario.key(models=[model])
        assert tagged_chat(gateway, model, key, metadata_tags=(name,)).status_code == 200
        assert_refused_by(tagged_chat(gateway, model, key, metadata_tags=(name,)), "tag", name, "requests", 1)


def test_a_per_key_tag_refusal_is_attributed_to_tag_per_key(gateway: Gateway) -> None:
    """A key has ``metadata.tag_rpm_limit={<tag>: 1}``. The second request carrying the tag must be a 429 for
    ``tag_per_key: <hash>:<tag>`` with ``rate_limited_by: tag_per_key``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        name: Final = "ratelim-key-tag"
        key: Final = scenario.key(models=[model], metadata={"tag_rpm_limit": {name: 1}})
        assert tagged_chat(gateway, model, key, metadata_tags=(name,)).status_code == 200
        assert_refused_by(
            tagged_chat(gateway, model, key, metadata_tags=(name,)),
            "tag_per_key",
            f"{hashed(key)}:{name}",
            "requests",
            1,
        )


def test_the_reset_at_header_matches_the_reset_time_in_the_message(gateway: Gateway) -> None:
    """A key with rpm_limit=1 is refused. The ``reset_at`` header must have the form ``YYYY-MM-DD HH:MM:SS UTC``
    and equal the text after ``Limit resets at: `` in the error message, since both come from one timestamp."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert chat(gateway, model, key).status_code == 200
        refused: Final = chat(gateway, model, key)
        assert refused.status_code == 429, refused.text
        assert RESET_AT.match(refused.headers["reset_at"]), refused.headers
        assert refused.json()["error"]["message"].endswith(f"Limit resets at: {refused.headers['reset_at']}")


def test_a_served_response_never_carries_rate_limited_by(gateway: Gateway) -> None:
    """A key with rpm_limit=5 sends one request. The 200 must carry no ``rate_limited_by``, ``rate_limit_type``,
    ``reset_at`` or ``retry-after`` header: those belong to refusals only."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=5)
        served: Final = chat(gateway, model, key)
        assert served.status_code == 200, served.text
        assert {"rate_limited_by", "rate_limit_type", "reset_at", "retry-after"}.isdisjoint(served.headers)


def test_a_provider_429_is_not_attributed_to_any_proxy_scope(gateway: Gateway) -> None:
    """A key without limits calls a provider that answers 429. The proxy's 429 must not carry
    ``rate_limited_by`` and its message must not start with ``Rate limit exceeded for``, so a client can tell a
    provider throttle from a proxy limit. UNVERIFIED: the vendor 429 path is ``litellm/exceptions.py``
    ``RateLimitError`` with category ``vendor_rate_limit``; the absence of ``rate_limited_by`` is read from
    ``_handle_rate_limit_error`` being the only writer of that header."""
    with wire_server(provider(status=429)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model])
        throttled: Final = chat(gateway, model, key)
        assert throttled.status_code == 429, throttled.text
        assert "rate_limited_by" not in throttled.headers
        assert not throttled.json()["error"]["message"].startswith("Rate limit exceeded for")
