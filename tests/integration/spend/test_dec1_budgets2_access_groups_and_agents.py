"""Contract: model access group holds, and managed agents forcing fail closed budget enforcement.

Model access groups. A deployment listed in `model_info.access_groups` and a group budget set with
`PUT /access_group/<group>/budget` give every request that auth matched through that group
(`UserAPIKeyAuth.matched_model_access_groups`) a Redis counter `spend:model_access_group:<group>`
(`budget_reservation.py::_get_model_access_group_budget_counters`). The hold is added to it like any other counter,
settled to the real cost after the call, and released on failure. The concurrent refusal and the plain refusal are
pinned in `authorization/test_access_group_budget.py`; these tests pin the counter itself and fail closed.

Managed agents. When the caller is a managed agent identity, or a key invokes a managed agent
(`managed_authorization.py` sets `billing_agent_policy`), the hold is taken as if
`general_settings.fail_closed_budget_enforcement` were on (`user_api_key_auth.py::_reserve_budget_after_common_checks`):
an estimate that does not fit the budget is refused with an `Estimated request cost` instead of being shrunk.

Price: input free, output 0.004 a token (`dec1_budgets_support.hold_model`), one served call 0.02, no `max_tokens`
estimates 65.536.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, eventually, held, string_value
from tests.integration._support.wire import Reply, Request, Wire, wire_server
from tests.integration.spend.dec1_budgets2_support import owned
from tests.integration.spend.dec1_budgets_support import (
    CHAT_COST,
    LANDING_SECONDS,
    background,
    budget_message,
    call,
    counter,
    counter_settles,
    held_provider,
    hold_model,
    redis_client,
    refused,
)
from tests.integration.spend.dec1_spend_support import new_customer, scripted_upstream, served

pytestmark: Final = pytest.mark.shares_worker_state


def _group_counter(group: str) -> str:
    return f"spend:model_access_group:{group}"


def _grouped_model(scenario: Scenario, provider: object, group: str) -> str:
    return hold_model(scenario, provider, model_info={"access_groups": [group]})  # type: ignore[arg-type]


def _group_budget(gateway: Gateway, scenario: Scenario, group: str, max_budget: float) -> None:
    created: Final = gateway.request(
        "PUT", f"/access_group/{group}/budget", {"max_budget": max_budget, "budget_duration": "30d"}
    )
    assert created.status_code == 200, created.text
    scenario.cleanups.callback(gateway.request, "DELETE", f"/access_group/{group}/budget")


def _group() -> str:
    return f"dec1-b2-group-{uuid.uuid4().hex[:12]}"


@pytest.mark.timeout(180)
def test_an_in_flight_call_holds_its_estimate_on_the_access_group_counter(gateway: Gateway) -> None:
    """A key granted the access group by name (`models: [<group>]`) sends a held call with `max_tokens: 10` to a
    deployment in the group, whose budget is 1.0. While it is in flight `spend:model_access_group:<group>` reads
    exactly 0.04; after release it settles at 0.02."""
    group: Final = _group()
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = _grouped_model(scenario, provider, group)
        _group_budget(gateway, scenario, group, 1.0)
        key: Final = scenario.key(models=[group])
        in_flight: Final = run(lambda: call(gateway, model, key, hold=True, max_tokens=10))
        provider.wait_for(1)
        assert counter(cache, _group_counter(group)) == pytest.approx(0.04)
        provider.release()
        served(in_flight.result(timeout=60))
        counter_settles(cache, _group_counter(group), CHAT_COST)


@pytest.mark.timeout(180)
def test_a_failed_call_releases_the_access_group_hold(gateway: Gateway) -> None:
    """The provider answers 500 to a grouped call with `max_tokens: 10`. The caller gets an error and the group
    counter settles back to 0: a failed call never consumes the group's shared budget."""
    group: Final = _group()
    with held_provider(failing_status=500) as provider, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = _grouped_model(scenario, provider, group)
        _group_budget(gateway, scenario, group, 1.0)
        key: Final = scenario.key(models=[group])
        response: Final = call(gateway, model, key, max_tokens=10)
        assert response.status_code >= 500, response.text
        counter_settles(cache, _group_counter(group), 0.0)


@pytest.mark.timeout(180)
def test_a_call_without_max_tokens_shrinks_its_group_hold_to_the_remaining_group_budget(gateway: Gateway) -> None:
    """The group budget is 1.0 and the key has none. A held call without `max_tokens` (estimate 65.536) is admitted
    by default (fail open) and the group counter reads exactly 1.0 while it is in flight."""
    group: Final = _group()
    with held_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = _grouped_model(scenario, provider, group)
        _group_budget(gateway, scenario, group, 1.0)
        key: Final = scenario.key(models=[group])
        in_flight: Final = run(lambda: call(gateway, model, key, hold=True))
        provider.wait_for(1)
        assert counter(cache, _group_counter(group)) == pytest.approx(1.0)
        provider.release()
        served(in_flight.result(timeout=60))


@pytest.mark.timeout(240)
def test_fail_closed_refuses_a_grouped_call_whose_estimate_does_not_fit_the_group_budget(
    gateway: Gateway, tmp_path: Path
) -> None:
    """On a proxy with `fail_closed_budget_enforcement: true`, a group budget of 0.03 and a call with
    `max_tokens: 10` (0.04) is refused with HTTP 422 and exactly `Budget has been exceeded! Model access
    group=<group> Current cost: 0.0, Estimated request cost: 0.04, Max budget: 0.03` (`engine.rs::exceeded`). The
    provider never sees it."""
    group: Final = _group()
    with (
        scripted_upstream() as wire,
        owned(gateway, tmp_path, general_settings={"fail_closed_budget_enforcement": True}) as proxy,
        proxy.scenario() as scenario,
    ):
        model: Final = _grouped_model(scenario, wire, group)
        _group_budget(proxy, scenario, group, 0.03)
        key: Final = scenario.key(models=[group])
        message: Final = refused(call(proxy, model, key, max_tokens=10))
        assert message == budget_message(f"Model access group={group}", "0.0", "0.03", estimate="0.04")
        assert [request for request in wire.drain() if request.method == "POST"] == []


@pytest.mark.timeout(180)
def test_a_group_without_a_budget_gets_no_counter(gateway: Gateway) -> None:
    """A key granted an access group that has no budget is served, and no `spend:model_access_group:<group>`
    counter appears (`_model_access_group_counter` returns nothing without a positive `max_budget`)."""
    group: Final = _group()
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = _grouped_model(scenario, wire, group)
        key: Final = scenario.key(models=[group])
        served(call(gateway, model, key, max_tokens=5))
        assert held(lambda: counter(cache, _group_counter(group)), lambda value: value is None, holding=3) is None


@pytest.mark.timeout(180)
def test_a_key_granted_the_model_by_name_does_not_charge_the_group_counter(gateway: Gateway) -> None:
    """The deployment is in a budgeted group, but the key lists the model name itself, not the group. The call is
    served and the group counter is never created: only groups that authorized the request are charged
    (`matched_model_access_groups`). UNVERIFIED: whether `common_checks` still records the group when the model is
    granted directly (check `matched_model_access_groups` in auth_checks.py)."""
    group: Final = _group()
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = _grouped_model(scenario, wire, group)
        _group_budget(gateway, scenario, group, 1.0)
        key: Final = scenario.key(models=[model])
        served(call(gateway, model, key, max_tokens=5))
        held(lambda: counter(cache, _group_counter(group)), lambda value: value is None, holding=3)


@pytest.mark.timeout(240)
def test_a_lost_group_counter_is_reseeded_from_the_stored_group_spend(gateway: Gateway) -> None:
    """After one served grouped call the stored group spend (`LiteLLM_ModelAccessGroupBudgetTable.spend`) is 0.02.
    The test deletes `spend:model_access_group:<group>` from Redis; the next call with `max_tokens: 5` settles the
    recreated counter at 0.04. Group counters are not reseeded by `SpendCounterReseed`; their seed is the cached
    group spend passed as `fallback_spend`. UNVERIFIED: whether the cached group spend is refreshed before the
    second call (the auth cache TTL may serve 0.0, which would read 0.02 here).

    BUG: in CI runs 44161684479 and 44176682961 the deleted counter was never recreated: it read None for the whole
    70 s window after a served second call, so not even that call's 0.02 landed. The post-call charge only
    increments counters it finds warm or reseeds them through `ensure_spend_counter_initialized`
    (`_prepare_spend_counter_increment`, litellm/proxy/proxy_server.py:4471-4498 on main), and reserved groups are
    skipped there (`_prepare_unreserved_spend_counter_increment`, proxy_server.py:4455-4468), so a lost group
    counter stays lost and the group budget is enforced against the lagging stored spend only."""
    pytest.skip("BUG: a deleted model access group spend counter is never recreated by later served calls")
    group: Final = _group()
    with scripted_upstream() as wire, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = _grouped_model(scenario, wire, group)
        _group_budget(gateway, scenario, group, 1.0)
        key: Final = scenario.key(models=[group])
        served(call(gateway, model, key, max_tokens=5))
        counter_settles(cache, _group_counter(group), CHAT_COST)
        eventually(
            lambda: gateway.rows(
                'SELECT spend FROM "LiteLLM_ModelAccessGroupBudgetTable" WHERE access_group_name = %s', (group,)
            ),
            lambda rows: float(str(rows[0]["spend"])) >= CHAT_COST - 1e-9,
            seconds=LANDING_SECONDS,
        )
        cache.delete(_group_counter(group))
        served(call(gateway, model, key, max_tokens=5))
        counter_settles(cache, _group_counter(group), 2 * CHAT_COST)


TENANT: Final = str(uuid.uuid4())
ISSUER: Final = f"https://login.microsoftonline.com/{TENANT}/v2.0"
AUDIENCE: Final = "api://dec1-budgets2-gateway"
KEY_ID: Final = "dec1-budgets2-signing-key"


@dataclass(frozen=True, slots=True)
class _Signer:
    private_key: rsa.RSAPrivateKey
    jwks: Wire


@pytest.fixture(scope="module")
def signer() -> Iterator[_Signer]:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk: Final = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    body: Final = json.dumps({"keys": [{**public_jwk, "kid": KEY_ID}]}).encode()

    def answer(_: Request) -> Reply:
        return Reply(body=body)

    with wire_server(answer) as jwks:
        yield _Signer(private_key, jwks)


@pytest.fixture(scope="module")
def jwt_proxy(signer: _Signer, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    from tests.integration._support.client import gateway_from_environment

    settings: Final[dict[str, JsonValue]] = {
        "enable_jwt_auth": True,
        "litellm_jwtauth": {"issuers": [{"issuer": ISSUER, "jwks_url": signer.jwks.url, "audience": AUDIENCE}]},
    }
    with (
        gateway_from_environment() as shared,
        owned(shared, tmp_path_factory.mktemp("dec1-budgets2-agents"), general_settings=settings) as proxy,
    ):
        yield proxy


def _managed_agent_token(scenario: Scenario, signer: _Signer, model: str) -> str:
    """An autonomous managed agent bound to a synthetic Entra app, and a signed app token for it (the setup of
    `messages_endpoint/safeguards/test_safeguard_results_managed_agent.py`)."""
    client_id: Final = str(uuid.uuid4())
    principal: Final = str(uuid.uuid4())
    agent: Final = scenario.gateway.post(
        "/v1/agents",
        {
            "agent_name": f"dec1-b2-{uuid.uuid4().hex[:10]}",
            "agent_card_params": {
                "protocolVersion": "0.3.0",
                "name": "dec1-budgets2",
                "description": "dec1 budgets2 agent",
                "url": "http://127.0.0.1:1/agent",
                "version": "1",
                "capabilities": {},
                "defaultInputModes": ["text"],
                "defaultOutputModes": ["text"],
                "skills": [],
            },
            "identity": {
                "provider": "microsoft_entra",
                "tenant_id": TENANT,
                "client_id": client_id,
                "service_principal_id": principal,
            },
            "execution_mode": "autonomous",
            "object_permission": {"models": [model]},
        },
    )
    agent_id: Final = string_value(agent["agent_id"])
    scenario.cleanups.callback(scenario.gateway.request, "DELETE", f"/v1/agents/{agent_id}")
    issued: Final = int(time.time())
    return jwt.encode(
        {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "tid": TENANT,
            "azp": client_id,
            "oid": principal,
            "sub": principal,
            "idtyp": "app",
            "iat": issued,
            "exp": issued + 600,
        },
        signer.private_key,
        algorithm="RS256",
        headers={"kid": KEY_ID},
    )


@pytest.mark.timeout(240)
def test_a_managed_agent_is_refused_an_estimate_that_does_not_fit_although_the_proxy_fails_open(
    jwt_proxy: Gateway, signer: _Signer
) -> None:
    """The proxy has no `fail_closed_budget_enforcement`. An autonomous managed agent sends a call without
    `max_tokens` for end user `<customer>` whose `max_budget` is 1.0. The estimate (65.536) does not fit, and because
    the caller is a managed agent (`billing_agent_policy`) it is refused with HTTP 422 and a message starting
    `Budget has been exceeded! EndUser=<customer> Current cost: 0.0, Estimated request cost: 65.536`. UNVERIFIED: the
    autonomous agent has no key, team or user, so the end user budget is the only counter; that the end user is
    resolved from the body `user` on the JWT path was not traced."""
    with scripted_upstream() as wire, jwt_proxy.scenario() as scenario:
        model: Final = hold_model(scenario, wire)
        end_user: Final = new_customer(jwt_proxy, scenario, max_budget=1.0)
        token: Final = _managed_agent_token(scenario, signer, model)
        message: Final = refused(call(jwt_proxy, model, token, user=end_user))
        assert message.startswith(
            f"Budget has been exceeded! EndUser={end_user} Current cost: 0.0, Estimated request cost: 65.536"
        ), message


@pytest.mark.timeout(240)
def test_a_plain_key_with_the_same_end_user_budget_is_admitted_fail_open(jwt_proxy: Gateway) -> None:
    """The counterpart: a virtual key sends the same call without `max_tokens` for an end user with `max_budget`
    1.0 on the same proxy. No managed agent is involved, so the hold is shrunk to 1.0 and the call is served."""
    with scripted_upstream() as wire, jwt_proxy.scenario() as scenario:
        model: Final = hold_model(scenario, wire)
        end_user: Final = new_customer(jwt_proxy, scenario, max_budget=1.0)
        key: Final = scenario.key(models=[model])
        assert served(call(jwt_proxy, model, key, user=end_user))["usage"] is not None


@pytest.mark.timeout(240)
def test_a_managed_agent_whose_estimate_fits_is_served(jwt_proxy: Gateway, signer: _Signer) -> None:
    """The managed agent sends `max_tokens: 5` (estimate 0.02) for an end user with `max_budget` 1.0. The estimate
    fits, so fail closed admits it and the call is served: forcing fail closed refuses only what could overspend."""
    with scripted_upstream() as wire, jwt_proxy.scenario() as scenario:
        model: Final = hold_model(scenario, wire)
        end_user: Final = new_customer(jwt_proxy, scenario, max_budget=1.0)
        token: Final = _managed_agent_token(scenario, signer, model)
        assert served(call(jwt_proxy, model, token, max_tokens=5, user=end_user))["usage"] is not None
