from typing import Final

import httpx
import pytest

from litellm.litellm_core_utils.oauth_token_exchange import OAuthTokenExchangeConfig
from litellm.proxy.control_plane.auth.rbac import (
    USAGE_VIEWER_ROLE,
    VIEWER_ROLE,
    build_principal,
    default_policy,
)
from litellm.proxy.control_plane.dataplanes.client import DataplaneClient, RetryPolicy
from litellm.proxy.control_plane.dataplanes.credentials import Caller, OidcTokenExchange
from litellm.proxy.control_plane.dataplanes.identity import USER_INFO_PATH
from litellm.proxy.control_plane.dataplanes.registry import Dataplane
from litellm.proxy.control_plane.fleet.service import fetch_teams
from tests.unit.proxy.control_plane.support import RoutedTransport, response, rig

EU_MODELS: Final = "https://eu.test/model/info"
US_MODELS: Final = "https://us.test/model/info"
EU_TEAMS: Final = "https://eu.test/team/list"
US_TEAMS: Final = "https://us.test/team/list"
EU_USER_INFO: Final = f"https://eu.test{USER_INFO_PATH}"


def _model(name: str, provider: str, model_id: str, upstream: str | None = None) -> dict[str, object]:
    return {
        "model_name": name,
        "litellm_params": {"model": upstream or f"{provider}/{name}", "custom_llm_provider": provider},
        "model_info": {"id": model_id, "mode": "chat", "db_model": True},
    }


async def _exchange(config: OAuthTokenExchangeConfig, subject_token: str) -> str:
    assert config.audience == "dataplane"
    assert subject_token == "idp-token"
    return "dataplane-token"


def test_fleet_models_unions_deployments_and_groups_them_by_public_name() -> None:
    client, _, minter = rig(
        {
            EU_MODELS: [
                response(200, {"data": [_model("gpt-4o", "azure", "m1"), _model("claude", "anthropic", "m2")]})
            ],
            US_MODELS: [response(200, {"data": [_model("gpt-4o", "openai", "m3")]})],
        }
    )
    body: Final = client.get("/api/v1/fleet/models", headers=minter.headers()).json()
    assert [(m["dataplane_id"], m["model_name"], m["provider"], m["model_id"]) for m in body["items"]] == [
        ("eu", "gpt-4o", "azure", "m1"),
        ("eu", "claude", "anthropic", "m2"),
        ("us", "gpt-4o", "openai", "m3"),
    ]
    assert body["groups"] == [
        {"model_name": "claude", "dataplanes": ["eu"]},
        {"model_name": "gpt-4o", "dataplanes": ["eu", "us"]},
    ]
    assert body["partial"] is False


def test_fleet_models_accepts_a_single_deployment_object_from_a_dataplane() -> None:
    client, _, minter = rig(
        {
            EU_MODELS: [response(200, {"data": _model("gpt-4o", "azure", "m1")})],
            US_MODELS: [response(200, {"data": []})],
        }
    )

    body: Final = client.get("/api/v1/fleet/models", headers=minter.headers()).json()

    assert body["partial"] is False
    assert body["unreachable"] == []
    assert body["items"] == [
        {
            "dataplane_id": "eu",
            "model_name": "gpt-4o",
            "litellm_model": "azure/gpt-4o",
            "provider": "azure",
            "model_id": "m1",
            "mode": "chat",
        }
    ]


def test_fleet_models_tolerates_an_unparseable_dataplane_and_reports_it() -> None:
    client, _, minter = rig(
        {
            EU_MODELS: [response(200, {"data": [_model("gpt-4o", "azure", "m1")]})],
            US_MODELS: [response(200, {"data": "x"})],
        }
    )
    body: Final = client.get("/api/v1/fleet/models", headers=minter.headers()).json()
    assert [m["model_id"] for m in body["items"]] == ["m1"] and body["partial"] is True
    assert body["unreachable"][0]["id"] == "us" and "validation error" in body["unreachable"][0]["reason"]


def test_fleet_teams_counts_members_and_keeps_budget_fields() -> None:
    team: Final = {
        "team_id": "t1",
        "team_alias": "ml",
        "organization_id": None,
        "spend": 12.5,
        "max_budget": 100.0,
        "members_with_roles": [{"user_id": "a", "role": "admin"}, {"user_id": "b", "role": "user"}],
        "models": ["gpt-4o"],
        "blocked": False,
        "metadata": {"ignored": True},
    }
    client, _, minter = rig({EU_TEAMS: [response(200, [team])], US_TEAMS: [response(200, [])]})
    body: Final = client.get("/api/v1/fleet/teams", headers=minter.headers()).json()
    assert body["items"] == [
        {
            "dataplane_id": "eu",
            "team_id": "t1",
            "team_alias": "ml",
            "organization_id": None,
            "spend": 12.5,
            "max_budget": 100.0,
            "members": 2,
            "models": ["gpt-4o"],
            "blocked": False,
        }
    ]


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize(
    "identity",
    [
        {"user_id": "alice"},
        {"user_info": {"user_id": "alice"}},
        {"user_id": "", "user_info": {"user_id": "alice"}},
        {"user_id": "alice", "user_info": {"user_id": "other"}},
    ],
)
def test_fleet_teams_falls_back_to_the_callers_own_teams_when_the_dataplane_refuses_to_list_all(
    identity: dict[str, object], status: int
) -> None:
    own: Final = {"team_id": "t-own", "members_with_roles": [{"user_id": "alice"}]}
    client, transport, minter = rig(
        {
            EU_TEAMS: [
                response(status, {"detail": {"error": "Only admin users can query all teams"}}),
                response(200, [own]),
            ],
            EU_USER_INFO: [response(200, identity)],
            US_TEAMS: [response(200, [])],
        }
    )
    body: Final = client.get("/api/v1/fleet/teams", headers=minter.headers()).json()
    assert [(t["dataplane_id"], t["team_id"], t["members"]) for t in body["items"]] == [("eu", "t-own", 1)]
    assert body["partial"] is False
    eu_calls: Final = [(url, params) for url, params, _ in transport.calls if url in {EU_TEAMS, EU_USER_INFO}]
    assert eu_calls == [
        (EU_TEAMS, None),
        (EU_USER_INFO, None),
        (EU_TEAMS, (("user_id", "alice"),)),
    ]


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize(
    "identity",
    [{}, {"user_id": ""}, {"user_id": "", "user_info": {"user_id": ""}}, {"user_info": {"user_id": ""}}],
)
def test_fleet_teams_keeps_the_refusal_when_user_info_has_no_resolvable_identity(
    identity: dict[str, object], status: int
) -> None:
    client, transport, minter = rig(
        {EU_TEAMS: [response(status)], EU_USER_INFO: [response(200, identity)], US_TEAMS: [response(200, [])]}
    )

    body: Final = client.get("/api/v1/fleet/teams", headers=minter.headers()).json()

    assert body["items"] == []
    assert body["partial"] is True
    assert [entry["id"] for entry in body["unreachable"]] == ["eu"]
    assert [(url, params) for url, params, _ in transport.calls if url in {EU_TEAMS, EU_USER_INFO}] == [
        (EU_TEAMS, None),
        (EU_USER_INFO, None),
    ]


def test_fleet_teams_treats_a_user_info_not_found_as_an_empty_reached_dataplane() -> None:
    client, _, minter = rig(
        {
            EU_TEAMS: [response(403)],
            EU_USER_INFO: [response(404)],
            US_TEAMS: [response(200, [])],
        }
    )

    body: Final = client.get("/api/v1/fleet/teams", headers=minter.headers()).json()

    assert body["items"] == []
    assert body["partial"] is False
    assert body["unreachable"] == []


@pytest.mark.asyncio
async def test_fleet_teams_uses_dataplane_identity_for_oidc_exchange_callers() -> None:
    team: Final = {"team_id": "t-own", "members_with_roles": [{"user_id": "dataplane-user"}]}
    routes: Final = {
        EU_TEAMS: iter(
            [
                response(401, {"detail": {"error": "Only admin users can query all teams"}}),
                response(200, [team]),
            ]
        ),
        EU_USER_INFO: iter([response(200, {"user_id": "dataplane-user"})]),
    }
    transport: Final = RoutedTransport(routes)
    client: Final = DataplaneClient(transport=transport, timeout_seconds=1, retry=RetryPolicy(attempts=1))
    credential: Final = OidcTokenExchange(
        OAuthTokenExchangeConfig(
            token_endpoint="https://identity.test/token",
            client_id="control-plane",
            client_secret="secret",
            profile="rfc8693",
            audience="dataplane",
            scope=None,
        ),
        _exchange,
    )
    dataplane: Final = Dataplane(
        id="eu",
        name="EU",
        url="https://eu.test",
        credential=credential,
        credential_kinds=("oidc_exchange",),
    )
    principal: Final = build_principal(
        default_policy(),
        "idp-subject",
        "idp@example.com",
        frozenset({VIEWER_ROLE}),
        None,
        frozenset({"eu"}),
        "oidc",
    )
    caller: Final = Caller(principal=principal, bearer="idp-token")

    teams: Final = await fetch_teams(client, dataplane, caller)

    assert isinstance(teams, tuple)
    assert tuple(team.team_id for team in teams) == ("t-own",)
    assert [(url, params) for url, params, _ in transport.calls] == [
        (EU_TEAMS, None),
        (EU_USER_INFO, None),
        (EU_TEAMS, (("user_id", "dataplane-user"),)),
    ]
    assert all(
        headers is not None and headers["Authorization"] == "Bearer dataplane-token"
        for _, _, headers in transport.calls
    )


def test_fleet_routes_need_fleet_read_and_503_when_nobody_answers() -> None:
    client, _, minter = rig({EU_TEAMS: [httpx.ConnectError("down")] * 3, US_TEAMS: [response(502, "bad")] * 3})
    denied: Final = client.get("/api/v1/fleet/teams", headers=minter.headers(role=USAGE_VIEWER_ROLE))
    assert denied.status_code == 403 and "fleet:read" in denied.json()["detail"]
    assert client.get("/api/v1/fleet/teams", headers=minter.headers()).status_code == 503


def test_a_team_with_null_spend_lists_as_zero_instead_of_failing_the_fleet() -> None:
    fresh: Final = {"team_id": "new", "spend": None}
    client, _, minter = rig({EU_TEAMS: [response(200, [fresh])], US_TEAMS: [response(200, [])]})
    reply: Final = client.get("/api/v1/fleet/teams", headers=minter.headers())
    assert reply.status_code == 200, reply.text
    body: Final = reply.json()
    assert [(item["team_id"], item["spend"]) for item in body["items"]] == [("new", 0.0)]
    assert body["partial"] is False
