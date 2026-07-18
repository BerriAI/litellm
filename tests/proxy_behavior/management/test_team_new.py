from typing import Any, Dict

import pytest

from .actors import Actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


# POST /team/new — actor x org-target matrix (org_target picks the request's
# organization_id: none / ORG_A / ORG_B). Pinned against the role gate, which
# 401s every denial: PROXY_ADMIN always passes; any other caller must name an
# organization_id AND be ORG_ADMIN of that org.
_SCENARIOS = [
    ("none/proxy_admin", Actor.PROXY_ADMIN, "none", 200),
    ("none/org_admin", Actor.ORG_ADMIN, "none", 401),
    ("none/team_admin", Actor.TEAM_ADMIN, "none", 401),
    ("none/internal_user", Actor.INTERNAL_USER, "none", 401),
    ("none/owner", Actor.OWNER, "none", 401),
    ("none/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "none", 401),
    ("none/cross_org_user", Actor.CROSS_ORG_USER, "none", 401),
    ("none/service_account", Actor.SERVICE_ACCOUNT, "none", 401),
    ("none/org_b_admin", Actor.ORG_B_ADMIN, "none", 401),
    ("org_a/proxy_admin", Actor.PROXY_ADMIN, "org_a", 200),
    ("org_a/org_admin", Actor.ORG_ADMIN, "org_a", 200),
    ("org_a/team_admin", Actor.TEAM_ADMIN, "org_a", 401),
    ("org_a/internal_user", Actor.INTERNAL_USER, "org_a", 401),
    ("org_a/owner", Actor.OWNER, "org_a", 401),
    ("org_a/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "org_a", 401),
    ("org_a/cross_org_user", Actor.CROSS_ORG_USER, "org_a", 401),
    ("org_a/service_account", Actor.SERVICE_ACCOUNT, "org_a", 401),
    ("org_a/org_b_admin", Actor.ORG_B_ADMIN, "org_a", 401),
    ("org_b/proxy_admin", Actor.PROXY_ADMIN, "org_b", 200),
    ("org_b/org_admin", Actor.ORG_ADMIN, "org_b", 401),
    ("org_b/team_admin", Actor.TEAM_ADMIN, "org_b", 401),
    ("org_b/internal_user", Actor.INTERNAL_USER, "org_b", 401),
    ("org_b/owner", Actor.OWNER, "org_b", 401),
    ("org_b/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "org_b", 401),
    ("org_b/cross_org_user", Actor.CROSS_ORG_USER, "org_b", 401),
    ("org_b/service_account", Actor.SERVICE_ACCOUNT, "org_b", 401),
    ("org_b/org_b_admin", Actor.ORG_B_ADMIN, "org_b", 200),
]


@pytest.mark.parametrize(
    "actor,org_target,expected_status",
    [(a, o, s) for (_id, a, o, s) in _SCENARIOS],
    ids=[s[0] for s in _SCENARIOS],
)
async def test_team_new_authz_matrix(
    actor: Actor,
    org_target: str,
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
    world,
):
    caller = world.keys[actor]
    org_id = {
        "none": None,
        "org_a": world.org_a_id,
        "org_b": world.org_b_id,
    }[org_target]

    body: Dict[str, Any] = {"team_id": scratch.prefix, "team_alias": scratch.prefix}
    if org_id is not None:
        body["organization_id"] = org_id

    resp = await proxy_client.post(
        "/team/new",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json=body,
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value} org={org_target}: {resp.status_code} {resp.text}"

    row = await prisma.db.litellm_teamtable.find_unique(
        where={"team_id": scratch.prefix}
    )
    if expected_status == 200:
        assert row is not None
        assert row.organization_id == org_id
    else:
        assert row is None, f"{actor.value}: denied but team row leaked"


async def test_team_new_rejects_negative_budget(proxy_client, prisma, scratch, world):
    """Input-validation pin: max_budget < 0 is a 400, no row created."""
    resp = await proxy_client.post(
        "/team/new",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"team_id": scratch.prefix, "max_budget": -1},
    )
    assert resp.status_code == 400, resp.text
    row = await prisma.db.litellm_teamtable.find_unique(
        where={"team_id": scratch.prefix}
    )
    assert row is None


async def test_team_new_rejects_duplicate_team_id(proxy_client, prisma, scratch, world):
    """Input-validation pin: a colliding team_id is a 400 on the second call."""
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    first = await proxy_client.post(
        "/team/new",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"team_id": scratch.prefix, "team_alias": scratch.prefix},
    )
    assert first.status_code == 200, first.text

    second = await proxy_client.post(
        "/team/new",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"team_id": scratch.prefix, "team_alias": scratch.prefix},
    )
    assert second.status_code == 400, second.text


async def test_team_new_unknown_organization_is_500(
    proxy_client, prisma, scratch, world
):
    """SURFACED, NOT ENDORSED: a /team/new with an organization_id that does
    not exist currently fails 500 (the role-resolution layer raises before
    the handler's own 400 'Organization not found' check is reached)."""
    resp = await proxy_client.post(
        "/team/new",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={
            "team_id": scratch.prefix,
            "organization_id": scratch.tag("no-such-org"),
        },
    )
    assert resp.status_code == 500, resp.text
    row = await prisma.db.litellm_teamtable.find_unique(
        where={"team_id": scratch.prefix}
    )
    assert row is None
