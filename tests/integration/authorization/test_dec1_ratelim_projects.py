"""Per-model rate limits on projects, including the separate input and output token windows (ITPM and OTPM).

The contract, read from ``_project_model_rate_limit_descriptor``, ``_project_io_token_rate_limit_descriptors``,
``reserve_io_tokens`` and ``_reserve_project_io_tokens_or_raise`` in
``litellm/proxy/hooks/parallel_request_limiter_v3.py``:

- A project created with ``model_rpm_limit`` / ``model_tpm_limit`` gives each listed model one window shared by
  every key in the project: descriptor ``model_per_project``, value ``<project_id>:<model>``.
- ``model_itpm_limit`` and ``model_otpm_limit`` give the model separate input and output token windows
  (descriptors ``model_per_project_itpm`` and ``model_per_project_otpm``). The input window reserves a precise
  tokenizer estimate of the prompt; the output window reserves the explicit output cap (``max_tokens``) or, when
  the request has none, a floor of a quarter of the smallest OTPM limit, which is also written into the provider
  request as ``max_tokens``.
- When an ITPM or OTPM window refuses, the combined TPM reservation already made for the same request (key, team,
  ...) is refunded, so a refused request leaves no tokens behind.

These windows are new (#51608 added batch enqueued project limits on top) and the Rust port (#51537) keeps
project scopes apart in its saved reservations; these tests pin the request path.
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
)
from tests.integration.authorization.dec1_ratelim_support import redis_counters, tokens_counter


def test_a_projects_per_model_request_window_is_shared_by_its_keys(gateway: Gateway) -> None:
    """A project has ``model_rpm_limit={<model>: 1}`` and two keys. Key one is served; key two must be refused with
    ``Rate limit exceeded for model_per_project: <project_id>:<model>. Limit type: requests. Current limit: 1``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], model_rpm_limit={model: 1})
        first: Final = scenario.key(team_id=team, project_id=project)
        second: Final = scenario.key(team_id=team, project_id=project)
        assert chat(gateway, model, first).status_code == 200
        assert_limited(chat(gateway, model, second), "model_per_project", f"{project}:{model}", "requests", 1)


def test_a_projects_per_model_limit_does_not_count_other_models(gateway: Gateway) -> None:
    """A project limits model A to one request per minute. After a request to A, two requests to model B with the
    same key must be served: only listed models get a project descriptor."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        limited: Final = scenario.model(api_base=f"{wire.url}/v1")
        free: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[limited, free])
        project: Final = scenario.project(team, models=[limited, free], model_rpm_limit={limited: 1})
        key: Final = scenario.key(team_id=team, project_id=project)
        assert chat(gateway, limited, key).status_code == 200
        assert [chat(gateway, free, key).status_code for _ in range(2)] == [200, 200]


def test_a_projects_per_model_tpm_limit_refuses_a_request_too_large_for_it(gateway: Gateway) -> None:
    """A project has ``model_tpm_limit={<model>: 20}``. A request asking max_tokens=50 (estimate 60) must be refused
    for ``model_per_project: <project_id>:<model>`` with ``Limit type: tokens. Current limit: 20, Remaining: 20``
    and never reach the provider."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], model_tpm_limit={model: 20})
        key: Final = scenario.key(team_id=team, project_id=project)
        assert_limited(
            chat(gateway, model, key, max_tokens=50), "model_per_project", f"{project}:{model}", "tokens", 20, 20
        )
        assert posts(wire) == ()


def test_an_output_window_refuses_a_request_whose_max_tokens_exceeds_it(gateway: Gateway) -> None:
    """A project has ``model_otpm_limit={<model>: 10}``. A request asking max_tokens=50 must be refused with
    ``Rate limit exceeded for model_per_project_otpm: <project_id>:<model>. Limit type: tokens. Current limit: 10,
    Remaining: 10`` and header ``rate_limited_by: model_per_project_otpm``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], model_otpm_limit={model: 10})
        key: Final = scenario.key(team_id=team, project_id=project)
        refused: Final = chat(gateway, model, key, max_tokens=50)
        assert_limited(refused, "model_per_project_otpm", f"{project}:{model}", "tokens", 10, 10)
        assert refused.headers["rate_limited_by"] == "model_per_project_otpm"
        assert posts(wire) == ()


def test_an_input_window_refuses_a_prompt_larger_than_it(gateway: Gateway) -> None:
    """A project has ``model_itpm_limit={<model>: 1}``. Any non-empty prompt estimates above one input token, so
    the request must be refused for ``model_per_project_itpm: <project_id>:<model>`` with ``Limit type: tokens.
    Current limit: 1, Remaining: 1``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], model_itpm_limit={model: 1})
        key: Final = scenario.key(team_id=team, project_id=project)
        assert_limited(
            chat(gateway, model, key, max_tokens=5), "model_per_project_itpm", f"{project}:{model}", "tokens", 1, 1
        )
        assert posts(wire) == ()


def test_an_output_window_caps_an_uncapped_request_at_a_quarter_of_the_limit(gateway: Gateway) -> None:
    """A project has ``model_otpm_limit={<model>: 200}`` and a request sets no max_tokens. The provider must receive
    ``max_tokens: 50`` (``int(200 * 0.25)``, ``NO_MAX_TOKENS_TPM_RESERVATION_FRACTION``) so an unbounded generation
    cannot overrun the output window."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], model_otpm_limit={model: 200})
        key: Final = scenario.key(team_id=team, project_id=project)
        assert chat(gateway, model, key).status_code == 200
        (sent,) = posts(wire)
        assert sent["max_tokens"] == 50, sent


def test_a_served_response_reports_the_output_window(gateway: Gateway) -> None:
    """A project has ``model_otpm_limit={<model>: 1000}``; a request asks max_tokens=40. The 200 must carry
    ``x-ratelimit-model_per_project_otpm-limit-tokens: 1000`` and ``-remaining-tokens: 960``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], model_otpm_limit={model: 1000})
        key: Final = scenario.key(team_id=team, project_id=project)
        served: Final = chat(gateway, model, key, max_tokens=40)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "model_per_project_otpm") == {"remaining-tokens": "960", "limit-tokens": "1000"}


def test_the_output_window_settles_to_the_reported_completion_tokens(gateway: Gateway) -> None:
    """A project has ``model_otpm_limit={<model>: 1000}``; a request reserves max_tokens=40 and the provider reports
    7 completion tokens. ``{model_per_project_otpm:<project_id>:<model>}:tokens`` must settle to 7. UNVERIFIED
    that settlement lands on exactly the completion count (``_build_io_token_reservation_ops``)."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario, redis_counters() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], model_otpm_limit={model: 1000})
        key: Final = scenario.key(team_id=team, project_id=project)
        assert chat(gateway, model, key, max_tokens=40).status_code == 200
        settled: Final = eventually(
            lambda: tokens_counter(redis, "model_per_project_otpm", f"{project}:{model}"),
            lambda tokens: tokens == 7,
            seconds=20,
        )
        assert settled == 7


def test_the_input_window_settles_to_the_reported_prompt_tokens(gateway: Gateway) -> None:
    """A project has ``model_itpm_limit={<model>: 100000}``; the provider reports 11 prompt tokens.
    ``{model_per_project_itpm:<project_id>:<model>}:tokens`` must settle to 11. UNVERIFIED as above."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario, redis_counters() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], model_itpm_limit={model: 100000})
        key: Final = scenario.key(team_id=team, project_id=project)
        assert chat(gateway, model, key, max_tokens=40).status_code == 200
        settled: Final = eventually(
            lambda: tokens_counter(redis, "model_per_project_itpm", f"{project}:{model}"),
            lambda tokens: tokens == 11,
            seconds=20,
        )
        assert settled == 11


def test_an_output_window_refusal_refunds_the_keys_token_reservation(gateway: Gateway) -> None:
    """A key with tpm_limit=1000 belongs to a project with ``model_otpm_limit={<model>: 10}``. A request asking
    max_tokens=50 first reserves 60 on the key and is then refused by the output window. The key's
    ``{api_key:<hash>}:tokens`` must be back at 0 (or absent), and the next request asking max_tokens=5 must report
    ``x-ratelimit-api_key-remaining-tokens: 985``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario, redis_counters() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], model_otpm_limit={model: 10})
        key: Final = scenario.key(team_id=team, project_id=project, tpm_limit=1000)
        assert chat(gateway, model, key, max_tokens=50).status_code == 429
        assert (tokens_counter(redis, "api_key", hashed(key)) or 0) == 0
        served: Final = chat(gateway, model, key, max_tokens=5)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key")["remaining-tokens"] == "985"


def test_an_output_refusal_refunds_the_input_reservation_of_the_same_request(gateway: Gateway) -> None:
    """A project has ``model_itpm_limit={<model>: 100000}`` and ``model_otpm_limit={<model>: 10}``. A request
    asking max_tokens=50 reserves its input estimate and is then refused by the output window. The ITPM counter
    ``{model_per_project_itpm:<project_id>:<model>}:tokens`` must be back at 0 (or absent): ``reserve_io_tokens``
    rolls back the input half when the output half refuses."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario, redis_counters() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(
            team, models=[model], model_itpm_limit={model: 100000}, model_otpm_limit={model: 10}
        )
        key: Final = scenario.key(team_id=team, project_id=project)
        assert chat(gateway, model, key, max_tokens=50).status_code == 429
        assert (tokens_counter(redis, "model_per_project_itpm", f"{project}:{model}") or 0) == 0
