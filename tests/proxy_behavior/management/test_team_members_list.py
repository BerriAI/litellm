from dataclasses import dataclass

import pytest
from prisma import Json

from .actors import Actor
from .conftest import create_scratch_actor, create_scratch_team, create_scratch_user
from .test_team_info import _SCENARIOS as TEAM_INFO_SCENARIOS

pytestmark = pytest.mark.asyncio(loop_scope="session")


@dataclass(frozen=True)
class Roster:
    team_id: str
    default_budget_id: str
    custom_budget_id: str
    user_ids: tuple


async def _budget(prisma, budget_id: str, **limits) -> str:
    await prisma.db.litellm_budgettable.create(
        data={"budget_id": budget_id, "created_by": "phase4-scratch", "updated_by": "phase4-scratch", **limits}
    )
    return budget_id


async def _membership(prisma, team_id: str, user_id: str, spend: float, budget_id: str | None) -> None:
    data = {"user_id": user_id, "team_id": team_id, "spend": spend, "total_spend": spend * 3}
    if budget_id is not None:
        data["litellm_budget_table"] = {"connect": {"budget_id": budget_id}}
    await prisma.db.litellm_teammembership.create(data=data)


async def _seed_roster(prisma, scratch, world) -> Roster:
    """Seven members in roster order: an admin on a custom budget whose email lives only on the user row, a
    member linked to the team default, a member with no membership row, a member whose id holds LIKE
    wildcards, and three filler members for paging."""
    team_id = scratch.prefix
    default_budget_id = await _budget(prisma, scratch.tag("default-budget"), max_budget=10.0)
    custom_budget_id = await _budget(
        prisma,
        scratch.tag("custom-budget"),
        max_budget=25.0,
        budget_duration="30d",
        tpm_limit=100,
        rpm_limit=5,
        allowed_models=["model-a"],
    )
    ada = await create_scratch_user(
        prisma, scratch.prefix, suffix="ada", user_email=f"{scratch.prefix}-ada@example.com"
    )
    bea = await create_scratch_user(prisma, scratch.prefix, suffix="bea")
    cole = await create_scratch_user(prisma, scratch.prefix, suffix="cole")
    wild = await create_scratch_user(prisma, scratch.prefix, suffix="100%_wild")
    fillers = tuple([await create_scratch_user(prisma, scratch.prefix, suffix=f"filler-{n}") for n in range(3)])
    await create_scratch_team(prisma, team_id, organization_id=world.org_a_id)
    roster = [
        {"user_id": ada, "role": "admin"},
        {"user_id": bea, "user_email": f"{scratch.prefix}-BEA@Example.com", "role": "user"},
        {"user_id": cole, "role": "user"},
        {"user_id": wild, "role": "user"},
        *({"user_id": uid, "role": "user"} for uid in fillers),
    ]
    await prisma.db.litellm_teamtable.update(
        where={"team_id": team_id},
        data={"members_with_roles": Json(roster), "metadata": Json({"team_member_budget_id": default_budget_id})},
    )
    await _membership(prisma, team_id, ada, 2.5, custom_budget_id)
    await _membership(prisma, team_id, bea, 1.0, default_budget_id)
    await _membership(prisma, team_id, wild, 0.5, None)
    return Roster(
        team_id=team_id,
        default_budget_id=default_budget_id,
        custom_budget_id=custom_budget_id,
        user_ids=(ada, bea, cole, wild, *fillers),
    )


def _admin(world) -> dict:
    return {"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"}


async def _list(proxy_client, world, team_id: str, params: dict | None = None):
    return await proxy_client.get(f"/management/v1/teams/{team_id}/members", params=params, headers=_admin(world))


@pytest.mark.parametrize(
    "actor,target,expected_status",
    [(a, t, s) for (_id, a, t, s) in TEAM_INFO_SCENARIOS],
    ids=[s[0] for s in TEAM_INFO_SCENARIOS],
)
async def test_members_list_admits_exactly_the_team_info_readers(
    actor: Actor, target: str, expected_status: int, proxy_client, world
):
    team_id = {"alpha": world.team_alpha_id, "beta": world.team_beta_id, "gamma": world.team_gamma_id}[target]
    headers = {"Authorization": f"Bearer {world.keys[actor].cleartext}"}

    members = await proxy_client.get(f"/management/v1/teams/{team_id}/members", headers=headers)
    info = await proxy_client.get(f"/team/info?team_id={team_id}", headers=headers)

    assert members.status_code == info.status_code == expected_status, members.text
    if expected_status == 403:
        assert members.headers["content-type"] == "application/problem+json"
        assert members.json()["status"] == 403


async def test_members_list_admits_the_admin_viewer(proxy_client, prisma, scratch, world):
    viewer = await create_scratch_actor(prisma, scratch.prefix, user_role="proxy_admin_viewer")
    resp = await proxy_client.get(
        f"/management/v1/teams/{world.team_beta_id}/members",
        headers={"Authorization": f"Bearer {viewer.cleartext}"},
    )
    assert resp.status_code == 200, resp.text


async def test_members_list_unknown_team_is_a_404_problem(proxy_client, world):
    resp = await _list(proxy_client, world, "behavior-pin-no-such-team")
    assert resp.status_code == 404, resp.text
    assert resp.json()["type"].endswith("team-not-found")


async def test_members_list_agrees_with_team_info(proxy_client, prisma, scratch, world):
    roster = await _seed_roster(prisma, scratch, world)

    listed = await _list(proxy_client, world, roster.team_id, {"page_size": "100"})
    info = await proxy_client.get(f"/team/info?team_id={roster.team_id}", headers=_admin(world))
    assert listed.status_code == 200, listed.text
    assert info.status_code == 200, info.text

    items = listed.json()["data"]
    info_members = info.json()["team_info"]["members_with_roles"]
    memberships = {tm["user_id"]: tm for tm in info.json()["team_memberships"]}
    assert [(i["user_id"], i["user_email"], i["role"]) for i in items] == [
        (m["user_id"], m["user_email"], m["role"]) for m in info_members
    ]
    for item in items:
        tm = memberships.get(item["user_id"])
        if tm is None:
            assert item["spend"] == 0 and item["budget_id"] is None, item
            assert item["budget_source"] == "team_default", item
            continue
        budget = tm["litellm_budget_table"] or {}
        assert (item["spend"], item["total_spend"], item["budget_id"], item["budget_source"]) == (
            tm["spend"],
            tm["total_spend"],
            tm["budget_id"],
            tm["budget_source"],
        )
        assert (item["max_budget_in_team"], item["tpm_limit"], item["rpm_limit"], item["budget_duration"]) == (
            budget.get("max_budget"),
            budget.get("tpm_limit"),
            budget.get("rpm_limit"),
            budget.get("budget_duration"),
        )
        assert list(item["allowed_models"]) == (budget.get("allowed_models") or [])


async def test_members_list_derives_each_budget_source(proxy_client, prisma, scratch, world):
    roster = await _seed_roster(prisma, scratch, world)

    resp = await _list(proxy_client, world, roster.team_id)
    assert resp.status_code == 200, resp.text
    by_id = {i["user_id"]: i for i in resp.json()["data"]}
    ada, bea, cole, wild = roster.user_ids[:4]

    assert by_id[ada]["budget_source"] == "custom"
    assert by_id[ada]["max_budget_in_team"] == 25.0
    assert by_id[ada]["allowed_models"] == ["model-a"]
    assert by_id[ada]["user_email"] == f"{scratch.prefix}-ada@example.com"
    assert by_id[bea]["budget_source"] == "team_default"
    assert by_id[bea]["max_budget_in_team"] == 10.0
    assert by_id[cole]["budget_source"] == "team_default"
    assert by_id[wild]["budget_source"] == "team_default"


@pytest.mark.parametrize(
    "metadata",
    [pytest.param({}, id="no-team-default"), pytest.param({"team_member_budget_id": "gone"}, id="deleted-default")],
)
async def test_members_list_without_a_live_team_default_reads_none(metadata, proxy_client, prisma, scratch, world):
    member = await create_scratch_user(prisma, scratch.prefix, suffix="member")
    await create_scratch_team(prisma, scratch.prefix, member_user_ids=[member], metadata=metadata)

    resp = await _list(proxy_client, world, scratch.prefix)
    assert resp.status_code == 200, resp.text
    assert [i["budget_source"] for i in resp.json()["data"]] == ["none"]


async def test_members_list_pages_through_the_roster_in_order(proxy_client, prisma, scratch, world):
    roster = await _seed_roster(prisma, scratch, world)

    pages = [await _list(proxy_client, world, roster.team_id, {"page": str(n), "page_size": "3"}) for n in (1, 2, 3, 4)]

    assert [p.status_code for p in pages] == [200, 200, 200, 200]
    assert [len(p.json()["data"]) for p in pages] == [3, 3, 1, 0]
    assert {(p.json()["meta"]["total_count"], p.json()["meta"]["total_pages"]) for p in pages} == {(7, 3)}
    assert tuple(i["user_id"] for p in pages for i in p.json()["data"]) == roster.user_ids
    assert pages[0].json()["links"]["next"] is not None
    assert pages[2].json()["links"]["next"] is None


async def test_members_list_caps_page_size(proxy_client, prisma, scratch, world):
    roster = await _seed_roster(prisma, scratch, world)
    resp = await _list(proxy_client, world, roster.team_id, {"page_size": "1000"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["meta"]["page_size"] == 100


async def test_members_list_never_reads_another_teams_roster(proxy_client, prisma, scratch, world):
    roster = await _seed_roster(prisma, scratch, world)
    other = await create_scratch_user(prisma, scratch.prefix, suffix="other")
    await create_scratch_team(prisma, scratch.tag("other-team"), member_user_ids=[other])

    resp = await _list(proxy_client, world, roster.team_id, {"q": "other"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"] == [] and resp.json()["meta"]["total_count"] == 0


@pytest.mark.parametrize(
    "q,expected",
    [
        pytest.param("BEA@example", ("bea",), id="email-case-insensitive"),
        pytest.param("-Cole", ("cole",), id="user-id-case-insensitive"),
        pytest.param("ada@", ("ada",), id="email-only-on-the-user-row"),
        pytest.param("%_", ("100%_wild",), id="like-wildcards-match-literally"),
        pytest.param("filler", ("filler-0", "filler-1", "filler-2"), id="substring"),
    ],
)
async def test_members_list_search(q, expected, proxy_client, prisma, scratch, world):
    roster = await _seed_roster(prisma, scratch, world)

    resp = await _list(proxy_client, world, roster.team_id, {"q": q})
    assert resp.status_code == 200, resp.text
    assert tuple(i["user_id"] for i in resp.json()["data"]) == tuple(scratch.tag(s) for s in expected)
    assert resp.json()["meta"]["total_count"] == len(expected)


@pytest.mark.parametrize(
    "params,expected_count",
    [
        pytest.param({"filter[role]": "admin"}, 1, id="eq-admin"),
        pytest.param({"filter[role]": "user"}, 6, id="eq-user"),
        pytest.param({"filter[role][in]": "admin,user"}, 7, id="in"),
        pytest.param({"filter[role]": "user", "q": "filler"}, 3, id="with-search"),
    ],
)
async def test_members_list_role_filter(params, expected_count, proxy_client, prisma, scratch, world):
    roster = await _seed_roster(prisma, scratch, world)

    resp = await _list(proxy_client, world, roster.team_id, params)
    assert resp.status_code == 200, resp.text
    assert resp.json()["meta"]["total_count"] == expected_count
    assert len(resp.json()["data"]) == expected_count
    assert {i["role"] for i in resp.json()["data"]} <= set(params.get("filter[role]", "admin,user").split(","))


async def test_members_list_sorts_by_spend_then_roster_order(proxy_client, prisma, scratch, world):
    roster = await _seed_roster(prisma, scratch, world)

    resp = await _list(proxy_client, world, roster.team_id, {"sort": "-spend"})
    assert resp.status_code == 200, resp.text
    ada, bea, cole, wild, *fillers = roster.user_ids
    assert tuple(i["user_id"] for i in resp.json()["data"]) == (ada, bea, wild, cole, *fillers)


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"team_id": "other"}, id="unknown-param"),
        pytest.param({"sort": "budget_id"}, id="unsortable-field"),
        pytest.param({"filter[spend]": "1"}, id="unfilterable-field"),
        pytest.param({"filter[role]": "owner"}, id="unknown-role"),
        pytest.param({"page": "abc"}, id="non-integer-page"),
    ],
)
async def test_members_list_rejects_unsupported_params(params, proxy_client, prisma, scratch, world):
    roster = await _seed_roster(prisma, scratch, world)
    resp = await _list(proxy_client, world, roster.team_id, params)
    assert resp.status_code == 400, resp.text
    assert resp.headers["content-type"] == "application/problem+json"


async def test_members_list_refuses_a_non_reader_before_judging_their_params(proxy_client, world):
    resp = await proxy_client.get(
        f"/management/v1/teams/{world.team_alpha_id}/members",
        params={"sort": "budget_id"},
        headers={"Authorization": f"Bearer {world.keys[Actor.CROSS_ORG_USER].cleartext}"},
    )
    assert resp.status_code == 403, resp.text
