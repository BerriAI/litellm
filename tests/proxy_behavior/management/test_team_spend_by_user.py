import pytest

from .actors import Actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


# GET /team/spend/by_user shares the team-scope resolver with
# /team/daily/activity, so the membership matrix must hold here too. team_ids
# is mandatory on this route (a per-user rollup with no team is meaningless),
# so the bare query is 400 for everyone instead of defaulting to own teams.
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
    if team == "none":
        return 400
    if actor == Actor.PROXY_ADMIN:
        return 200
    return 200 if actor in _MEMBERS.get(team, set()) else 404


_CASES = [
    (f"{team}/{actor.value}", actor, team, _expected(actor, team))
    for team in ("none", "alpha", "beta")
    for actor in Actor
]

_DATES = "start_date=2024-01-01&end_date=2024-12-31"


@pytest.mark.parametrize(
    "actor,team,expected_status",
    [(a, t, s) for (_id, a, t, s) in _CASES],
    ids=[c[0] for c in _CASES],
)
async def test_team_spend_by_user_matrix(actor: Actor, team: str, expected_status: int, proxy_client, world):
    team_id = {"alpha": world.team_alpha_id, "beta": world.team_beta_id}.get(team)
    query = _DATES if team_id is None else f"{_DATES}&team_ids={team_id}"

    resp = await proxy_client.get(
        f"/team/spend/by_user?{query}",
        headers={"Authorization": f"Bearer {world.keys[actor].cleartext}"},
    )
    assert resp.status_code == expected_status, f"{actor.value} -> {team}: {resp.status_code} {resp.text}"
    if expected_status == 200:
        body = resp.json()
        assert (body["start_date"], body["end_date"]) == ("2024-01-01", "2024-12-31")
        assert all(row["team_id"] == team_id for row in body["results"])
