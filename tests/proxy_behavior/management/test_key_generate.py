from typing import Any, Dict

import pytest

from .actors import TEAM_ALPHA, TEAM_BETA, Actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


# (id, actor, body_extras, expected_status). Status codes pinned to observed
# handler behavior — heterogeneous (200, 400, 401) because the handler routes
# denials through three different gates (role gate, user_id mismatch, team
# member permission).
_SCENARIOS = [
    ("self/proxy_admin", Actor.PROXY_ADMIN, {}, 200),
    ("self/org_admin", Actor.ORG_ADMIN, {}, 401),
    ("self/team_admin", Actor.TEAM_ADMIN, {}, 200),
    ("self/internal_user", Actor.INTERNAL_USER, {}, 200),
    ("self/owner", Actor.OWNER, {}, 200),
    ("self/unrelated_same_org", Actor.UNRELATED_SAME_ORG, {}, 200),
    ("self/cross_org_user", Actor.CROSS_ORG_USER, {}, 200),
    ("self/service_account", Actor.SERVICE_ACCOUNT, {}, 200),
    ("team_alpha/proxy_admin", Actor.PROXY_ADMIN, {"team_id": TEAM_ALPHA}, 200),
    ("team_alpha/org_admin", Actor.ORG_ADMIN, {"team_id": TEAM_ALPHA}, 401),
    ("team_alpha/team_admin", Actor.TEAM_ADMIN, {"team_id": TEAM_ALPHA}, 200),
    ("team_alpha/internal_user", Actor.INTERNAL_USER, {"team_id": TEAM_ALPHA}, 401),
    ("team_alpha/cross_org_user", Actor.CROSS_ORG_USER, {"team_id": TEAM_ALPHA}, 400),
    ("team_beta/proxy_admin", Actor.PROXY_ADMIN, {"team_id": TEAM_BETA}, 200),
    ("team_beta/org_admin", Actor.ORG_ADMIN, {"team_id": TEAM_BETA}, 401),
    ("team_beta/team_admin", Actor.TEAM_ADMIN, {"team_id": TEAM_BETA}, 400),
    ("team_beta/internal_user", Actor.INTERNAL_USER, {"team_id": TEAM_BETA}, 400),
    ("team_beta/cross_org_user", Actor.CROSS_ORG_USER, {"team_id": TEAM_BETA}, 401),
]


@pytest.mark.parametrize(
    "actor,body_extras,expected_status",
    [(actor, body, expected) for (_id, actor, body, expected) in _SCENARIOS],
    ids=[s[0] for s in _SCENARIOS],
)
async def test_key_generate_authz_matrix(
    actor: Actor,
    body_extras: Dict[str, Any],
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
    world,
):
    seeded = world.keys[actor]
    body: Dict[str, Any] = {"key_alias": scratch.prefix, **body_extras}

    resp = await proxy_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeded.cleartext}"},
        json=body,
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value} {body!r} → {resp.status_code}: {resp.text}"

    rows = await prisma.db.litellm_verificationtoken.find_many(
        where={"key_alias": scratch.prefix}
    )
    if expected_status == 200:
        cleartext = resp.json()["key"]
        assert cleartext.startswith("sk-")
        assert len(rows) == 1
    else:
        assert rows == [], f"{actor.value}: denied but row leaked"
