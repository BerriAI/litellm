"""
Gaps in the characterization of ``router_python_core.py`` lines 14000-17669 that the views answer:
``settings().allowed_fails_for`` under an allowed_fails_policy, ``settings().as_dict()`` for the
latency strategy and after a ``routing_strategy_args`` update, the ``registry()`` deployment
sources ``routable`` / ``alias`` / ``routing_groups`` / ``routed_group`` / ``for_request`` and team
aliases of ``all``, cached clients of a pod-discovery deployment, and PTU-share ceilings.
"""

from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.ptu_shares import PTUTeamCeiling
from litellm.proxy._types import UserAPIKeyAuth
from litellm.router_api import ChangeSet
from litellm.types.router import AllowedFailsPolicy
from tests.unit.test_router.core.scripted_host import (
    auth_error,
    bad_request,
    content_policy,
    context_window,
    deployment,
    rate_limit,
    server_error,
    timeout,
)
from tests.unit.test_router.core.selection_helpers import MakeCore, picked_ids

# --- settings().allowed_fails_for ----------------------------------------------------------------


def _bad_gateway() -> litellm.BadGatewayError:
    return litellm.BadGatewayError(message="bad gateway", llm_provider="openai", model="gpt-4o")


def _connection_error() -> litellm.APIConnectionError:
    return litellm.APIConnectionError(message="no route", llm_provider="openai", model="gpt-4o")


FULL_POLICY: Final = AllowedFailsPolicy(
    ContentPolicyViolationErrorAllowedFails=1,
    BadRequestErrorAllowedFails=2,
    BadGatewayErrorAllowedFails=3,
    RateLimitErrorAllowedFails=4,
)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (content_policy(), 1),
        (bad_request(), 2),
        # A context-window error is a BadRequestError, so it takes the bad-request allowance.
        (context_window(), 2),
        (_bad_gateway(), 3),
        (rate_limit(), 4),
        # Types the policy names no allowance for answer None, never the router's allowed_fails.
        (server_error(), None),
        (timeout(), None),
        (auth_error(), None),
        (_connection_error(), None),
    ],
    ids=[
        "content-policy",
        "bad-request",
        "context-window",
        "bad-gateway",
        "rate-limit",
        "server-error",
        "timeout",
        "auth",
        "connection",
    ],
)
def test_allowed_fails_for_reads_the_policy_field_of_the_error_type(
    make_core: MakeCore, error: Exception, expected: int | None
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], allowed_fails_policy=FULL_POLICY)
    assert core.settings().allowed_fails_for(error) == expected


def test_allowed_fails_for_a_content_policy_error_falls_back_to_the_bad_request_allowance(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a")], allowed_fails_policy=AllowedFailsPolicy(BadRequestErrorAllowedFails=2)
    )
    assert core.settings().allowed_fails_for(content_policy()) == 2


def test_allowed_fails_for_without_a_policy_is_none(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], allowed_fails=7)
    assert core.settings().allowed_fails_for(rate_limit()) is None


# --- settings().as_dict() and routing_strategy_args ----------------------------------------------

LATENCY_ARGS_DEFAULTS: Final = {"lowest_latency_buffer": 0, "max_latency_list_size": 10, "ttft_percentile": None}


def test_latency_routing_reports_the_strategys_full_routing_args(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a")], routing_strategy="latency-based-routing", routing_strategy_args={"ttl": 30}
    )
    assert core.settings().as_dict()["routing_strategy_args"] == {"ttl": 30.0, **LATENCY_ARGS_DEFAULTS}


def test_updating_routing_strategy_args_reaches_the_latency_strategy(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a")], routing_strategy="latency-based-routing", routing_strategy_args={"ttl": 30}
    )
    core.apply(ChangeSet(settings={"routing_strategy_args": {"ttl": 99}}))
    assert core.settings().as_dict()["routing_strategy_args"] == {"ttl": 99.0, **LATENCY_ARGS_DEFAULTS}


def test_routing_strategy_args_under_simple_shuffle_are_reported_as_given(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    assert core.settings().as_dict()["routing_strategy_args"] == {}
    core.apply(ChangeSet(settings={"routing_strategy_args": {"ttl": 99}}))
    assert core.settings().as_dict()["routing_strategy_args"] == {"ttl": 99}


def test_switching_to_latency_routing_takes_the_args_sent_with_it(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    core.apply(ChangeSet(settings={"routing_strategy": "latency-based-routing", "routing_strategy_args": {"ttl": 7}}))
    settings: Final = core.settings().as_dict()
    assert settings["routing_strategy"] == "latency-based-routing"
    assert settings["routing_strategy_args"] == {"ttl": 7.0, **LATENCY_ARGS_DEFAULTS}


# --- registry() deployment sources ---------------------------------------------------------------

TEAMS_AND_GROUPS: Final = [
    deployment("internal-x", "tx", model_info={"team_id": "team-x", "team_public_model_name": "pub"}),
    deployment("internal-y", "ty", model_info={"team_id": "team-y", "team_public_model_name": "pub"}),
    deployment("g", "a"),
    deployment("g", "b", model_info={"access_groups": ["ag"]}),
]

ROUTING_GROUP: Final = {"group_name": "grp", "models": ["g"], "routing_strategy": "simple-shuffle"}


def _ids(deployments: object) -> list[str]:  # mutable-ok: a fresh list
    return [deployment_id for _, deployment_id in _rows(deployments)]


def _rows(deployments: object) -> list[tuple[str, str]]:  # mutable-ok: a fresh list
    assert isinstance(deployments, (list, tuple))
    return [(str(row["model_name"]), str(row["model_info"]["id"])) for row in deployments]


def test_all_deployments_of_a_team_public_name_are_renamed_to_the_alias(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(TEAMS_AND_GROUPS))
    rows: Final = core.registry().deployments("pub", source="all", team_id="team-x", model_alias="nick")
    assert _rows(rows) == [("nick", "tx")]


def test_an_alias_to_a_routing_group_lists_the_members_under_the_alias_without_access_groups(
    make_core: MakeCore,
) -> None:
    core: Final = make_core(
        model_list=list(TEAMS_AND_GROUPS), routing_groups=[dict(ROUTING_GROUP)], model_group_alias={"al": "grp"}
    )
    rows: Final = core.registry().deployments("al")
    assert _rows(rows) == [("al", "a"), ("al", "b")]
    assert isinstance(rows, list)
    assert all("access_groups" not in row["model_info"] for row in rows)


def test_the_alias_source_lists_every_alias_row(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=list(TEAMS_AND_GROUPS),
        routing_groups=[dict(ROUTING_GROUP)],
        model_group_alias={"al": "grp", "al2": "g"},
    )
    rows: Final = core.registry().deployments(source="alias")
    assert _rows(rows) == [("al", "a"), ("al", "b"), ("al2", "a"), ("al2", "b")]
    # A plain alias row keeps the member's access groups; only routing-group rows drop them.
    assert isinstance(rows, list)
    assert rows[3]["model_info"].get("access_groups") == ["ag"]


def test_the_routing_groups_source_lists_members_under_the_group_without_access_groups(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(TEAMS_AND_GROUPS), routing_groups=[dict(ROUTING_GROUP)])
    rows: Final = core.registry().deployments(source="routing_groups")
    assert _rows(rows) == [("grp", "a"), ("grp", "b")]
    assert isinstance(rows, tuple)
    assert all("access_groups" not in row["model_info"] for row in rows)


def test_the_routed_group_of_a_team_public_name_is_the_teams_own_deployments(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(TEAMS_AND_GROUPS))
    assert _rows(core.registry().deployments("pub", source="routed_group", team_id="team-y")) == [("internal-y", "ty")]
    assert core.registry().deployments("pub", source="routed_group") == []


def test_the_routable_source_lists_the_deployments_routing_would_consider(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(TEAMS_AND_GROUPS))
    registry: Final = core.registry()
    assert _ids(registry.deployments("g", source="routable")) == ["a", "b"]
    assert registry.deployments("nope", source="routable") == ()
    assert _ids(registry.deployments("a", source="routable")) == ["a"]
    team_request: Final = {"metadata": {"user_api_key_team_id": "team-x"}}
    assert _ids(registry.deployments("pub", source="routable", request=team_request)) == ["tx"]


def test_the_routable_source_answers_nothing_for_a_name_several_teams_share_for_an_admin(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(TEAMS_AND_GROUPS))
    admin: Final = {"metadata": {"user_api_key_auth": UserAPIKeyAuth(user_role="proxy_admin")}}  # pyright: ignore[reportCallIssue]
    assert core.registry().deployments("pub", source="routable", request=admin) == ()


def test_the_for_request_source_finds_one_teams_public_name_for_a_proxy_admin(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[TEAMS_AND_GROUPS[0], *TEAMS_AND_GROUPS[2:]])
    admin: Final = {"metadata": {"user_api_key_auth": UserAPIKeyAuth(user_role="proxy_admin")}}  # pyright: ignore[reportCallIssue]
    assert _ids(core.registry().deployments("pub", source="for_request", request=admin)) == ["tx"]
    assert core.registry().deployments("pub", source="for_request", request={}) == []


def test_the_for_request_source_refuses_a_name_several_teams_share_for_a_proxy_admin(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(TEAMS_AND_GROUPS))
    admin: Final = {"metadata": {"user_api_key_auth": UserAPIKeyAuth(user_role="proxy_admin")}}  # pyright: ignore[reportCallIssue]
    with pytest.raises(litellm.BadRequestError) as raised:
        core.registry().deployments("pub", source="for_request", request=admin)
    assert str(raised.value) == (
        "litellm.BadRequestError: Model name 'pub' matches deployments from multiple teams. "
        "Specify the deployment ID directly to disambiguate."
    )


@pytest.mark.parametrize("client_type", ["async", None])
def test_a_pod_discovery_deployment_has_no_cached_client(make_core: MakeCore, client_type: str | None) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    pod_discovered: Final = deployment("g", "k", kubernetes_pod_discovery=True)
    assert core.registry().client(pod_discovered, client_type=client_type) is None


# --- PTU shares ----------------------------------------------------------------------------------


def _ptu_shared(model_name: str, deployment_id: str) -> dict[str, object]:  # mutable-ok: a model_list entry
    return deployment(
        model_name,
        deployment_id,
        model=f"azure/{deployment_id}",
        api_base="https://example.openai.azure.com",
        model_info={
            "ptu_shares": {"TA": 50},
            "ptu_count": 50,
            "cost_per_ptu_per_hour": 1.0,
            "ptu_effective_from": "2026-01-01T00:00:00Z",
            "base_model": "gpt-4o",
        },
    )


PTU_MODELS: Final = [_ptu_shared("gpt", "shared"), _ptu_shared("solo", "s2"), deployment("solo", "open")]


def test_a_share_holding_team_has_a_ptu_ceiling_on_its_group(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(PTU_MODELS))
    expected: Final = PTUTeamCeiling(
        model_group="gpt", tpm_limit=125000, output_to_input_ratio=4.0, cached_input_ratio=0.0
    )
    assert core.health().team_ptu_ceiling("TA", "gpt") == expected
    assert core.health().team_ptu_ceiling("TA", "solo") == PTUTeamCeiling(
        model_group="solo", tpm_limit=125000, output_to_input_ratio=4.0, cached_input_ratio=0.0
    )
    assert core.health().team_ptu_ceiling("TB", "gpt") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_kwargs", "group", "expected"),
    [
        ({"metadata": {"user_api_key_team_id": "TA"}}, "gpt", "shared"),
        ({"metadata": {"user_api_key_team_id": "TA"}}, "solo", "s2"),
        ({"metadata": {"user_api_key_team_id": "TB"}}, "solo", "open"),
        ({}, "solo", "open"),
    ],
    ids=["holder-shared", "holder-prefers-its-share", "other-team-open", "no-team-open"],
)
async def test_ptu_shares_route_holders_to_their_share_and_others_to_open_deployments(
    make_core: MakeCore, request_kwargs: dict[str, object], group: str, expected: str
) -> None:
    core: Final = make_core(model_list=list(PTU_MODELS))
    assert await picked_ids(core, group, request_kwargs=request_kwargs) == {expected}


@pytest.mark.asyncio
@pytest.mark.parametrize("request_kwargs", [{"metadata": {"user_api_key_team_id": "TB"}}, {}], ids=["other", "none"])
async def test_a_group_only_share_holders_may_use_refuses_everyone_else(
    make_core: MakeCore, request_kwargs: dict[str, object]
) -> None:
    core: Final = make_core(model_list=list(PTU_MODELS))
    with pytest.raises(litellm.BadRequestError) as raised:
        await core.select("gpt", dict(request_kwargs))
    assert str(raised.value) == (
        "litellm.BadRequestError: Deployment gpt is reserved for the teams holding a PTU share of it"
    )
