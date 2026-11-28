"""Per-model limits on a key or team (``model_rpm_limit``, ``model_tpm_limit``).

The contract: ``model_rpm_limit`` and ``model_tpm_limit`` are maps from a model name to a limit. They only
count requests for that model, each model has its own window, and they are charged in addition to the
key-wide limits. The scopes are:

- ``model_per_key``, identity ``<key hash>:<model>``: from the key's own map, else inherited from the team's
  map, else from the deployment's ``litellm_params.default_api_key_rpm_limit`` / ``default_api_key_tpm_limit``
  (the smallest across the model's deployments). An inherited limit still gives each key its own window.
- ``model_per_team``, identity ``<team_id>:<model>``: from the team's map, shared by all keys of the team,
  unless the key has its own limit for that model, in which case the team's limit is not applied to that key.

Refusals read ``Rate limit exceeded for model_per_key: <hash>:<model>. Limit type: requests. ...`` and served
responses carry ``x-ratelimit-model_per_key-*`` and ``x-ratelimit-model_per_team-*`` headers. Redis keys are
``{model_per_key:<hash>:<model>}:requests``.

``routing/test_per_model_rate_limit_fallbacks.py`` already pins that these refusals answer 429 on every
endpoint and how they interact with fallbacks. This file pins scoping, inheritance, headers and storage.

Read from ``litellm/proxy/hooks/parallel_request_limiter_v3.py`` (``_add_model_per_key_rate_limit_descriptor``,
``_add_team_model_rate_limit_descriptor_from_metadata``) and ``litellm/proxy/auth/auth_utils.py``
(``get_key_model_rpm_limit``, ``_get_deployment_default_limit``).
"""

from __future__ import annotations

from typing import Final

from tests.integration._support.client import Gateway
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limits_support import (
    WINDOW_SECONDS,
    assert_limited,
    chat,
    counter_key,
    hashed,
    posts,
    provider,
    ratelimit_header_names,
    redis_integer,
    scope_headers,
    shared_redis,
    statuses,
)


def test_a_per_model_rpm_limit_only_counts_requests_for_that_model(gateway: Gateway) -> None:
    """A key has ``model_rpm_limit`` of 1 for model A and may also call model B. After A is exhausted (200, then
    429 naming ``model_per_key: <hash>:<A>``), three requests for B must all be served and carry no
    ``x-ratelimit-model_per_key-*`` header."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        limited: Final = scenario.model(api_base=f"{wire.url}/v1")
        free: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[limited, free], model_rpm_limit={limited: 1})
        assert chat(gateway, limited, key).status_code == 200
        assert_limited(chat(gateway, limited, key), "model_per_key", f"{hashed(key)}:{limited}", "requests", 1)
        others: Final = [chat(gateway, free, key) for _ in range(3)]
        assert statuses(others) == [200, 200, 200], [response.text for response in others]
        assert all(scope_headers(response, "model_per_key") == {} for response in others)
        assert len(posts(wire)) == 4


def test_each_model_in_the_map_has_its_own_window(gateway: Gateway) -> None:
    """A key has ``model_rpm_limit`` of 1 for model A and 2 for model B. A is served once and refused; B must
    then still be served twice and refused on its third request with limit 2 and B's name in the message."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        first: Final = scenario.model(api_base=f"{wire.url}/v1")
        second: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[first, second], model_rpm_limit={first: 1, second: 2})
        assert statuses([chat(gateway, first, key), chat(gateway, first, key)]) == [200, 429]
        assert statuses([chat(gateway, second, key), chat(gateway, second, key)]) == [200, 200]
        assert_limited(chat(gateway, second, key), "model_per_key", f"{hashed(key)}:{second}", "requests", 2)


def test_a_served_response_reports_the_per_model_request_and_token_limits(gateway: Gateway) -> None:
    """A key has ``model_rpm_limit`` 10 and ``model_tpm_limit`` 1000 for a model and sends a request estimated at
    30 tokens. The 200 must carry ``x-ratelimit-model_per_key-limit-requests: 10``, ``-remaining-requests: 9``,
    ``-limit-tokens: 1000`` and ``-remaining-tokens: 970`` and no other ``x-ratelimit-*`` header."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], model_rpm_limit={model: 10}, model_tpm_limit={model: 1000})
        served: Final = chat(gateway, model, key, max_tokens=20)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "model_per_key") == {
            "limit-requests": "10",
            "remaining-requests": "9",
            "limit-tokens": "1000",
            "remaining-tokens": "970",
        }
        assert ratelimit_header_names(served) == {
            f"x-ratelimit-model_per_key-{field}-{kind}"
            for field in ("limit", "remaining")
            for kind in ("requests", "tokens")
        }


def test_a_per_model_tpm_limit_refuses_a_request_too_large_for_it(gateway: Gateway) -> None:
    """A key has ``model_tpm_limit`` of 20 for a model and asks for max_tokens=50 (estimate 60). It must be
    refused with ``Rate limit exceeded for model_per_key: <hash>:<model>. Limit type: tokens. Current limit: 20,
    Remaining: 20``, ``rate_limit_type: tokens`` and ``retry-after: 60``, before the provider is called."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], model_tpm_limit={model: 20})
        refused: Final = chat(gateway, model, key, max_tokens=50)
        assert_limited(refused, "model_per_key", f"{hashed(key)}:{model}", "tokens", 20, remaining=20)
        assert refused.headers["rate_limit_type"] == "tokens"
        assert refused.headers["retry-after"] == str(WINDOW_SECONDS)
        assert posts(wire) == ()


def test_the_per_model_window_is_charged_alongside_the_key_wide_window(gateway: Gateway) -> None:
    """A key has rpm_limit=10 and ``model_rpm_limit`` of 5 for a model. One request for that model must report
    both windows: ``x-ratelimit-api_key-*`` reading 10 and 9, and ``x-ratelimit-model_per_key-*`` reading 5
    and 4."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=10, model_rpm_limit={model: 5})
        served: Final = chat(gateway, model, key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-requests": "10", "remaining-requests": "9"}
        assert scope_headers(served, "model_per_key") == {"limit-requests": "5", "remaining-requests": "4"}


def test_the_per_model_counter_is_stored_under_the_key_hash_and_model_name(gateway: Gateway) -> None:
    """A key with ``model_rpm_limit`` of 5 for a model sends one request. Redis must hold
    ``{model_per_key:<hash>:<model>}:requests`` equal to 1."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario, shared_redis() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], model_rpm_limit={model: 5})
        assert chat(gateway, model, key).status_code == 200
        assert redis_integer(redis, counter_key("model_per_key", f"{hashed(key)}:{model}", "requests")) == 1
        assert len(posts(wire)) == 1


def test_a_teams_per_model_limit_is_shared_by_the_teams_keys(gateway: Gateway) -> None:
    """A team has ``model_rpm_limit`` of 2 for a model and two keys. Each key is served once. A third request
    from a key that has only used one of its own two must be refused naming the team:
    ``model_per_team: <team_id>:<model>`` with limit 2."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], model_rpm_limit={model: 2})
        first: Final = scenario.key(team_id=team)
        second: Final = scenario.key(team_id=team)
        assert statuses([chat(gateway, model, first), chat(gateway, model, second)]) == [200, 200]
        assert_limited(chat(gateway, model, first), "model_per_team", f"{team}:{model}", "requests", 2)
        assert len(posts(wire)) == 2


def test_a_teams_per_model_limit_is_also_applied_to_each_key_on_its_own(gateway: Gateway) -> None:
    """A team has ``model_rpm_limit`` of 3 for a model and one of its keys sends a request. The 200 must carry
    both ``x-ratelimit-model_per_key-*`` and ``x-ratelimit-model_per_team-*`` headers, each reading 3 and 2: a
    key with no per-model limit of its own inherits the team's number as its personal per-model limit too."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], model_rpm_limit={model: 3})
        served: Final = chat(gateway, model, scenario.key(team_id=team))
        assert served.status_code == 200, served.text
        assert scope_headers(served, "model_per_key") == {"limit-requests": "3", "remaining-requests": "2"}
        assert scope_headers(served, "model_per_team") == {"limit-requests": "3", "remaining-requests": "2"}


def test_one_key_exhausting_an_inherited_team_limit_is_named_as_model_per_key(gateway: Gateway) -> None:
    """A team has ``model_rpm_limit`` of 1 for a model and a single key uses it up. That key's second request is
    over both its inherited per-key window and the team window, and the 429 must name
    ``model_per_key: <hash>:<model>``, which is checked first."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], model_rpm_limit={model: 1})
        key: Final = scenario.key(team_id=team)
        assert chat(gateway, model, key).status_code == 200
        assert_limited(chat(gateway, model, key), "model_per_key", f"{hashed(key)}:{model}", "requests", 1)


def test_a_keys_own_per_model_limit_replaces_the_teams_for_that_key(gateway: Gateway) -> None:
    """A team has ``model_rpm_limit`` of 1 for a model and one of its keys has its own ``model_rpm_limit`` of 3
    for the same model. That key must be served three times, carry no ``x-ratelimit-model_per_team-*`` header,
    and be refused on the fourth request naming ``model_per_key`` with limit 3. This is the one place where a
    narrower scope overrides a wider one instead of both applying."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], model_rpm_limit={model: 1})
        key: Final = scenario.key(team_id=team, model_rpm_limit={model: 3})
        served: Final = [chat(gateway, model, key) for _ in range(3)]
        assert statuses(served) == [200, 200, 200], [response.text for response in served]
        assert all(scope_headers(response, "model_per_team") == {} for response in served)
        assert_limited(chat(gateway, model, key), "model_per_key", f"{hashed(key)}:{model}", "requests", 3)


def test_a_teams_per_model_limit_does_not_count_other_models(gateway: Gateway) -> None:
    """A team has ``model_rpm_limit`` of 1 for model A and may also use model B. After A is exhausted, the team's
    key must still be served on B with no per-model headers."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        limited: Final = scenario.model(api_base=f"{wire.url}/v1")
        free: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[limited, free], model_rpm_limit={limited: 1})
        key: Final = scenario.key(team_id=team)
        assert statuses([chat(gateway, limited, key), chat(gateway, limited, key)]) == [200, 429]
        served: Final = chat(gateway, free, key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "model_per_key") == {}
        assert scope_headers(served, "model_per_team") == {}


def test_a_deployments_default_api_key_rpm_limit_gives_every_key_its_own_window(gateway: Gateway) -> None:
    """A deployment is created with ``litellm_params.default_api_key_rpm_limit: 1``. Two keys with no limits of
    their own each get exactly one request served on that model; a second request from the first key must be
    refused naming ``model_per_key: <its hash>:<model>`` with limit 1."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", default_api_key_rpm_limit=1)
        first: Final = scenario.key(models=[model])
        second: Final = scenario.key(models=[model])
        assert chat(gateway, model, first).status_code == 200
        assert_limited(chat(gateway, model, first), "model_per_key", f"{hashed(first)}:{model}", "requests", 1)
        served: Final = chat(gateway, model, second)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "model_per_key") == {"limit-requests": "1", "remaining-requests": "0"}


def test_a_deployments_default_api_key_tpm_limit_is_reserved_per_key(gateway: Gateway) -> None:
    """A deployment is created with ``litellm_params.default_api_key_tpm_limit: 1000``. A key with no limits sends
    a request estimated at 30 tokens and must get ``x-ratelimit-model_per_key-limit-tokens: 1000`` and
    ``-remaining-tokens: 970``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", default_api_key_tpm_limit=1000)
        served: Final = chat(gateway, model, scenario.key(models=[model]), max_tokens=20)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "model_per_key") == {"limit-tokens": "1000", "remaining-tokens": "970"}


def test_a_keys_own_per_model_limit_wins_over_the_deployment_default(gateway: Gateway) -> None:
    """A deployment has ``default_api_key_rpm_limit: 1`` and a key has ``model_rpm_limit`` of 3 for that model.
    The key must be served three times and refused on the fourth with limit 3: the deployment default is only a
    fallback for keys and teams that set nothing."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", default_api_key_rpm_limit=1)
        key: Final = scenario.key(models=[model], model_rpm_limit={model: 3})
        served: Final = [chat(gateway, model, key) for _ in range(3)]
        assert statuses(served) == [200, 200, 200], [response.text for response in served]
        assert_limited(chat(gateway, model, key), "model_per_key", f"{hashed(key)}:{model}", "requests", 3)
        assert len(posts(wire)) == 3
