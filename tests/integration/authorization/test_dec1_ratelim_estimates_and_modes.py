"""Operator-declared output estimates, ``rpm_limit_type`` / ``tpm_limit_type`` modes and the over-budget throttle.

The contract:

- ``default_estimated_output_tokens`` and ``default_estimated_output_tokens_per_model`` on a key's or team's
  metadata replace the heuristic output part of the token reservation when a request has no ``max_tokens``.
  Priority: key per-model, key global, team per-model, team global (``get_estimated_output_tokens`` in
  ``litellm/proxy/auth/auth_utils.py``). Under a small TPM limit the implicit output cap written to the provider
  is raised to that estimate (``_apply_implicit_output_cap``).
- ``rpm_limit_type`` / ``tpm_limit_type`` set to ``dynamic`` on a key make its limit apply only while a
  deployment of the requested model has more than ``DYNAMIC_RATE_LIMIT_ERROR_THRESHOLD_PER_MINUTE`` (1) failures
  in the current minute (``_get_enforced_limit``). ``guaranteed_throughput`` refuses at key creation when the
  team's keys would be over-allocated.
- A key that opted in with ``metadata.throttle_on_budget_exceeded`` and is over its ``max_budget`` keeps serving at
  ``max(1, floor(limit * litellm_settings.budget_exceeded_throttle_percentage))`` instead of being blocked
  (``litellm/proxy/auth/budget_throttle.py``).

All three feed the descriptor builder and the reservation size, both rewritten in Rust by #51537
(``builder.rs``, ``estimate/``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limits_support import (
    assert_limited,
    chat,
    hashed,
    limits_proxy,
    posts,
    provider,
    scope_headers,
    statuses,
)


def test_a_keys_declared_output_estimate_replaces_the_heuristic_reservation(gateway: Gateway) -> None:
    """A key with tpm_limit=10000 has ``metadata.default_estimated_output_tokens=100`` and sends a 10-token prompt
    without max_tokens. The 200 must report ``x-ratelimit-api_key-remaining-tokens: 9890`` (10 + 100 reserved),
    not the 1024-token heuristic floor."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=10000, metadata={"default_estimated_output_tokens": 100})
        served: Final = chat(gateway, model, key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key")["remaining-tokens"] == "9890"


def test_a_per_model_output_estimate_wins_over_the_keys_global_estimate(gateway: Gateway) -> None:
    """A key with tpm_limit=10000 declares a global estimate of 100 and ``default_estimated_output_tokens_per_model
    = {<model>: 30}``. A 10-token prompt without max_tokens must report ``remaining-tokens: 9960`` (10 + 30)."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(
            models=[model],
            tpm_limit=10000,
            metadata={
                "default_estimated_output_tokens": 100,
                "default_estimated_output_tokens_per_model": {model: 30},
            },
        )
        served: Final = chat(gateway, model, key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key")["remaining-tokens"] == "9960"


def test_a_keys_output_estimate_wins_over_its_teams(gateway: Gateway) -> None:
    """A team declares ``default_estimated_output_tokens=500`` and its key declares 40; the key has tpm_limit=10000.
    A 10-token prompt without max_tokens must report ``remaining-tokens: 9950`` (10 + 40)."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], metadata={"default_estimated_output_tokens": 500})
        key: Final = scenario.key(team_id=team, tpm_limit=10000, metadata={"default_estimated_output_tokens": 40})
        served: Final = chat(gateway, model, key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key")["remaining-tokens"] == "9950"


def test_a_teams_output_estimate_applies_to_a_key_without_one(gateway: Gateway) -> None:
    """A team declares ``default_estimated_output_tokens=60`` and its key (tpm_limit=10000) declares none. A
    10-token prompt without max_tokens must report ``remaining-tokens: 9930`` (10 + 60)."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model], metadata={"default_estimated_output_tokens": 60})
        key: Final = scenario.key(team_id=team, tpm_limit=10000)
        served: Final = chat(gateway, model, key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key")["remaining-tokens"] == "9930"


def test_a_declared_estimate_raises_the_implicit_output_cap_under_a_small_limit(gateway: Gateway) -> None:
    """A key with tpm_limit=400 declares ``default_estimated_output_tokens=150``. A request without max_tokens must
    reach the provider with ``max_tokens: 150``: the cap is ``max(int(400 * 0.25), 150)``, not 100."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=400, metadata={"default_estimated_output_tokens": 150})
        assert chat(gateway, model, key).status_code == 200
        (sent,) = posts(wire)
        assert sent["max_tokens"] == 150, sent


def test_an_explicit_max_tokens_wins_over_the_declared_estimate(gateway: Gateway) -> None:
    """A key with tpm_limit=10000 declares an estimate of 500 and the request sets max_tokens=20. The 200 must
    report ``remaining-tokens: 9970`` (10 + 20) and the provider must receive ``max_tokens: 20``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=10000, metadata={"default_estimated_output_tokens": 500})
        served: Final = chat(gateway, model, key, max_tokens=20)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key")["remaining-tokens"] == "9970"
        (sent,) = posts(wire)
        assert sent["max_tokens"] == 20, sent


def test_a_declared_estimate_larger_than_the_limit_is_never_admitted(gateway: Gateway) -> None:
    """A key with tpm_limit=100 declares ``default_estimated_output_tokens=200``. A request without max_tokens
    reserves 210 and must be refused for ``api_key`` tokens with ``Current limit: 100, Remaining: 100`` on an
    empty window."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=100, metadata={"default_estimated_output_tokens": 200})
        assert_limited(chat(gateway, model, key), "api_key", hashed(key), "tokens", 100, 100)
        assert posts(wire) == ()


def test_a_dynamic_rpm_limit_is_not_enforced_while_the_model_is_healthy(gateway: Gateway) -> None:
    """A key has rpm_limit=1 with ``rpm_limit_type="dynamic"`` and its model has had no failures. Three requests
    must all be served, and no ``x-ratelimit-api_key-limit-requests`` header appears (the request limit is
    dropped from the descriptor)."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1, rpm_limit_type="dynamic")
        responses: Final = [chat(gateway, model, key) for _ in range(3)]
        assert statuses(responses) == [200, 200, 200]
        assert "limit-requests" not in scope_headers(responses[-1], "api_key")


def test_a_dynamic_rpm_limit_is_enforced_once_the_model_has_failures(gateway: Gateway) -> None:
    """A key with rpm_limit=1 and ``rpm_limit_type="dynamic"`` calls a model whose provider fails with 500. After
    two failures in the minute (more than the threshold of 1) the limit applies, so the next request must be
    refused for ``api_key`` requests. UNVERIFIED: depends on router failure tracking
    (``llm_router.health().failures_this_minute``) counting the 500s before the third request."""
    with wire_server(provider(status=500)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1, rpm_limit_type="dynamic")
        refused: Final = eventually(lambda: chat(gateway, model, key), lambda response: response.status_code == 429)
        assert refused.json()["error"]["message"].startswith(f"Rate limit exceeded for api_key: {hashed(key)}.")


def test_a_dynamic_tpm_limit_is_not_enforced_while_the_model_is_healthy(gateway: Gateway) -> None:
    """A key has tpm_limit=20 with ``tpm_limit_type="dynamic"``. A request asking max_tokens=50 (estimate 60) must
    be served, because a healthy model drops the token limit from the descriptor."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=20, tpm_limit_type="dynamic")
        assert chat(gateway, model, key, max_tokens=50).status_code == 200


def test_guaranteed_throughput_refuses_a_key_that_overallocates_its_team(gateway: Gateway) -> None:
    """A team has rpm_limit=10 and a key holding rpm_limit=6. Creating a second key with rpm_limit=5 and
    ``rpm_limit_type="guaranteed_throughput"`` must fail with 400 and the detail ``Allocated RPM limit=6 + Key RPM
    limit=5 is greater than team RPM limit=10``."""
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model], rpm_limit=10)
        scenario.key(team_id=team, rpm_limit=6)
        refused: Final = gateway.request(
            "POST", "/key/generate", {"team_id": team, "rpm_limit": 5, "rpm_limit_type": "guaranteed_throughput"}
        )
        assert refused.status_code == 400, refused.text
        assert "Allocated RPM limit=6 + Key RPM limit=5 is greater than team RPM limit=10" in refused.text


def test_best_effort_throughput_lets_keys_overallocate_their_team(gateway: Gateway) -> None:
    """The same team (rpm_limit=10, one key at 6) accepts a second key with rpm_limit=5 and
    ``rpm_limit_type="best_effort_throughput"``: the allocation check only runs for guaranteed throughput."""
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model], rpm_limit=10)
        scenario.key(team_id=team, rpm_limit=6)
        assert scenario.key(team_id=team, rpm_limit=5, rpm_limit_type="best_effort_throughput")


@pytest.fixture(scope="module")
def throttling(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory):  # noqa: ANN201
    """A proxy of our own with ``budget_exceeded_throttle_percentage: 0.5`` on the shared database and Redis."""
    directory: Final[Path] = tmp_path_factory.mktemp("ratelim-throttle")
    with limits_proxy(
        session_gateway, directory, litellm_settings={"budget_exceeded_throttle_percentage": 0.5}, shared=True
    ) as candidate:
        yield candidate


def _over_budget(gateway: Gateway, key: str) -> None:
    """Push the key past a max_budget of 0.000001 by recording spend directly through ``/key/update``."""
    gateway.post("/key/update", {"key": key, "spend": 1.0})


def test_an_opted_in_key_over_budget_keeps_serving_at_half_its_rpm(gateway: Gateway, throttling: Gateway) -> None:
    """On a proxy with throttle percentage 0.5, a key with rpm_limit=4, max_budget=0.000001 and
    ``metadata.throttle_on_budget_exceeded=true`` is over budget. Its requests must be served with
    ``x-ratelimit-api_key-limit-requests: 2`` and the third in the window refused with ``Current limit: 2``.
    UNVERIFIED: spend is set through ``/key/update`` rather than earned; the Rust budget engine (#51535) must still
    hand the throttle percentage to the limiter."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(
            models=[model], rpm_limit=4, max_budget=0.000001, metadata={"throttle_on_budget_exceeded": True}
        )
        _over_budget(gateway, key)
        served: Final = eventually(
            lambda: chat(throttling, model, key),
            lambda response: scope_headers(response, "api_key").get("limit-requests") == "2",
        )
        assert served.status_code == 200, served.text
        refused: Final = eventually(lambda: chat(throttling, model, key), lambda r: r.status_code == 429)
        assert_limited(refused, "api_key", hashed(key), "requests", 2)


def test_a_throttled_limit_never_drops_below_one(gateway: Gateway, throttling: Gateway) -> None:
    """On the same proxy, an opted-in key with rpm_limit=1 is over budget. It must still be served once per
    window with ``x-ratelimit-api_key-limit-requests: 1``: ``max(1, floor(1 * 0.5))``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(
            models=[model], rpm_limit=1, max_budget=0.000001, metadata={"throttle_on_budget_exceeded": True}
        )
        _over_budget(gateway, key)
        served: Final = chat(throttling, model, key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key")["limit-requests"] == "1"


def test_an_over_budget_key_that_did_not_opt_in_is_blocked_not_throttled(gateway: Gateway, throttling: Gateway) -> None:
    """On the same proxy, a key with rpm_limit=4 and max_budget=0.000001 but without the opt-in is over budget.
    Its request must be refused with a budget error (not a 429 rate limit) and never reach the provider.
    UNVERIFIED status code: the budget refusal is read as 400 ``budget_exceeded`` from auth_checks."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=4, max_budget=0.000001)
        _over_budget(gateway, key)
        refused: Final = eventually(lambda: chat(throttling, model, key), lambda r: r.status_code != 200)
        assert refused.status_code != 429, refused.text
        assert "budget" in refused.text.lower(), refused.text
        assert posts(wire) == ()


def test_an_opted_in_key_without_limits_is_still_blocked_over_budget(gateway: Gateway, throttling: Gateway) -> None:
    """On the same proxy, an opted-in key with no rpm or tpm limit is over budget. It must be blocked like any
    over-budget key, since there is nothing to scale down (``should_throttle_budget_exceeded``)."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], max_budget=0.000001, metadata={"throttle_on_budget_exceeded": True})
        _over_budget(gateway, key)
        refused: Final = eventually(lambda: chat(throttling, model, key), lambda r: r.status_code != 200)
        assert "budget" in refused.text.lower(), refused.text
        assert posts(wire) == ()
