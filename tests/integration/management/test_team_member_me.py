import uuid

import pytest

from litellm.proxy.utils import hash_token

from tests.integration._support.actors import Actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


# GET /team/{team_id}/members/me resolves the CALLER's own membership row.
# A caller that is not a member of the team is 404 — even PROXY_ADMIN, which
# is not in any seeded team. The route is self-managed, so every actor reaches
# the handler. TEAM_GAMMA has no members, so every actor is 404 there.
_MEMBERS = {
    "alpha": {
        Actor.TEAM_ADMIN,
        Actor.INTERNAL_USER,
        Actor.OWNER,
        Actor.UNRELATED_SAME_ORG,
        Actor.SERVICE_ACCOUNT,
    },
    "beta": {Actor.CROSS_ORG_USER},
    "gamma": set(),
}

_CASES = [
    (f"{team}/{actor.value}", actor, team, 200 if actor in members else 404)
    for team, members in _MEMBERS.items()
    for actor in Actor
]


@pytest.mark.parametrize(
    "actor,team,expected_status",
    [(a, t, s) for (_id, a, t, s) in _CASES],
    ids=[c[0] for c in _CASES],
)
async def test_team_member_me_matrix(
    actor: Actor, team: str, expected_status: int, gateway_client, actors
):
    team_id = {
        "alpha": actors.team_alpha_id,
        "beta": actors.team_beta_id,
        "gamma": actors.team_gamma_id,
    }[team]
    caller = actors.keys[actor]

    resp = await gateway_client.get(
        f"/team/{team_id}/members/me",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value} -> {team}: {resp.status_code} {resp.text}"

    if expected_status == 200:
        body = resp.json()
        assert body["user_id"] == caller.user_id
        assert body["team_id"] == team_id


async def test_team_member_me_team_key_without_user_id_is_400(
    gateway_client, db, resources, actors
):
    """A key with no associated user_id (a team / service-account key) cannot
    resolve 'me' — the caller has no identity to look up — so it is 400."""
    cleartext = "sk-" + uuid.uuid4().hex
    await db.db.litellm_verificationtoken.create(
        data={
            "token": hash_token(cleartext),
            "key_name": f"{resources.prefix}-teamkey",
            "key_alias": f"{resources.prefix}-teamkey",
            "team_id": actors.team_alpha_id,
            "models": [],
        }
    )
    resp = await gateway_client.get(
        f"/team/{actors.team_alpha_id}/members/me",
        headers={"Authorization": f"Bearer {cleartext}"},
    )
    assert resp.status_code == 400, resp.text
