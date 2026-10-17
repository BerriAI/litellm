import httpx

from litellm.proxy.control_plane.auth.rbac import USAGE_VIEWER_ROLE
from tests.unit.proxy.control_plane.support import response, rig

EU_MODELS = "https://eu.test/model/info"
US_MODELS = "https://us.test/model/info"
EU_TEAMS = "https://eu.test/team/list"
US_TEAMS = "https://us.test/team/list"


def _model(name: str, provider: str, model_id: str, upstream: str | None = None) -> dict[str, object]:
    return {
        "model_name": name,
        "litellm_params": {"model": upstream or f"{provider}/{name}", "custom_llm_provider": provider},
        "model_info": {"id": model_id, "mode": "chat", "db_model": True},
    }


def test_fleet_models_unions_deployments_and_groups_them_by_public_name() -> None:
    client, _, minter = rig(
        {
            EU_MODELS: [
                response(200, {"data": [_model("gpt-4o", "azure", "m1"), _model("claude", "anthropic", "m2")]})
            ],
            US_MODELS: [response(200, {"data": [_model("gpt-4o", "openai", "m3")]})],
        }
    )
    body = client.get("/api/v1/fleet/models", headers=minter.headers()).json()
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


def test_fleet_models_tolerates_an_unparseable_dataplane_and_reports_it() -> None:
    client, _, minter = rig(
        {
            EU_MODELS: [response(200, {"data": [_model("gpt-4o", "azure", "m1")]})],
            US_MODELS: [response(200, {"data": "x"})],
        }
    )
    body = client.get("/api/v1/fleet/models", headers=minter.headers()).json()
    assert [m["model_id"] for m in body["items"]] == ["m1"] and body["partial"] is True
    assert body["unreachable"][0]["id"] == "us" and "validation error" in body["unreachable"][0]["reason"]


def test_fleet_teams_counts_members_and_keeps_budget_fields() -> None:
    team = {
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
    body = client.get("/api/v1/fleet/teams", headers=minter.headers()).json()
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


def test_fleet_teams_falls_back_to_the_callers_own_teams_when_the_dataplane_refuses_to_list_all() -> None:
    own = {"team_id": "t-own", "members_with_roles": [{"user_id": "alice"}]}
    client, transport, minter = rig(
        {
            EU_TEAMS: [
                response(401, {"detail": {"error": "Only admin users can query all teams"}}),
                response(200, [own]),
            ],
            US_TEAMS: [response(200, [])],
        }
    )
    body = client.get("/api/v1/fleet/teams", headers=minter.headers()).json()
    assert [(t["dataplane_id"], t["team_id"], t["members"]) for t in body["items"]] == [("eu", "t-own", 1)]
    assert body["partial"] is False
    eu_params = [params for url, params, _ in transport.calls if url == EU_TEAMS]
    assert eu_params == [None, {"user_id": "alice"}]


def test_fleet_routes_need_fleet_read_and_503_when_nobody_answers() -> None:
    client, _, minter = rig({EU_TEAMS: [httpx.ConnectError("down")] * 3, US_TEAMS: [response(502, "bad")] * 3})
    denied = client.get("/api/v1/fleet/teams", headers=minter.headers(role=USAGE_VIEWER_ROLE))
    assert denied.status_code == 403 and "fleet:read" in denied.json()["detail"]
    assert client.get("/api/v1/fleet/teams", headers=minter.headers()).status_code == 503


def test_a_team_with_null_spend_lists_as_zero_instead_of_failing_the_fleet() -> None:
    fresh = {"team_id": "new", "spend": None}
    client, _, minter = rig({EU_TEAMS: [response(200, [fresh])], US_TEAMS: [response(200, [])]})
    reply = client.get("/api/v1/fleet/teams", headers=minter.headers())
    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert [(item["team_id"], item["spend"]) for item in body["items"]] == [("new", 0.0)]
    assert body["partial"] is False
