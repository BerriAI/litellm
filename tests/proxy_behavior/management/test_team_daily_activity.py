import pytest

from .actors import Actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


# GET /team/daily/activity. A proxy admin (admin view) sees activity for any
# team. A non-admin is scoped to user_info.teams: a bare query defaults to its
# own teams (200), and an explicit team_ids filter naming a team it does not
# belong to is 404 (the VERIA-43 fix). Org admins have no team memberships, so
# they behave like a non-member for any specific team.
_MEMBERS = {
    "alpha": {
        Actor.TEAM_ADMIN,
        Actor.INTERNAL_USER,
        Actor.OWNER,
        Actor.UNRELATED_SAME_ORG,
        Actor.SERVICE_ACCOUNT,
    },
    "beta": {Actor.CROSS_ORG_USER},
}


def _expected(actor: Actor, team: str) -> int:
    if team == "none" or actor == Actor.PROXY_ADMIN:
        return 200
    return 200 if actor in _MEMBERS.get(team, set()) else 404


_CASES = [
    (f"{team}/{actor.value}", actor, team, _expected(actor, team))
    for team in ("none", "alpha", "beta")
    for actor in Actor
]


# start_date / end_date are required by the handler — pin only the team-scope
# authz, not the date validation.
_DATES = "start_date=2024-01-01&end_date=2024-12-31"


@pytest.mark.parametrize(
    "actor,team,expected_status",
    [(a, t, s) for (_id, a, t, s) in _CASES],
    ids=[c[0] for c in _CASES],
)
async def test_team_daily_activity_matrix(
    actor: Actor, team: str, expected_status: int, proxy_client, world
):
    query = _DATES
    if team == "alpha":
        query += f"&team_ids={world.team_alpha_id}"
    elif team == "beta":
        query += f"&team_ids={world.team_beta_id}"

    resp = await proxy_client.get(
        f"/team/daily/activity?{query}",
        headers={"Authorization": f"Bearer {world.keys[actor].cleartext}"},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value} -> {team}: {resp.status_code} {resp.text}"
