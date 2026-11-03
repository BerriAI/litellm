import pytest

from tests.integration._support.actors import Actor

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
async def test_team_spend_by_user_matrix(actor: Actor, team: str, expected_status: int, gateway_client, actors):
    team_id = {"alpha": actors.team_alpha_id, "beta": actors.team_beta_id}.get(team)
    query = _DATES if team_id is None else f"{_DATES}&team_ids={team_id}"

    resp = await gateway_client.get(
        f"/team/spend/by_user?{query}",
        headers={"Authorization": f"Bearer {actors.keys[actor].cleartext}"},
    )
    assert resp.status_code == expected_status, f"{actor.value} -> {team}: {resp.status_code} {resp.text}"
    if expected_status == 200:
        body = resp.json()
        assert (body["start_date"], body["end_date"]) == ("2024-01-01", "2024-12-31")
        assert all(row["team_id"] == team_id for row in body["results"])


async def test_team_spend_by_user_timeout_seconds_runs_under_a_statement_timeout(gateway_client, actors):
    headers = {"Authorization": f"Bearer {actors.keys[Actor.PROXY_ADMIN].cleartext}"}
    query = f"{_DATES}&team_ids={actors.team_alpha_id}"

    resp = await gateway_client.get(f"/team/spend/by_user?{query}&timeout_seconds=60", headers=headers)
    assert resp.status_code == 200, resp.text
    assert all(row["team_id"] == actors.team_alpha_id for row in resp.json()["results"])

    too_small = await gateway_client.get(f"/team/spend/by_user?{query}&timeout_seconds=0", headers=headers)
    assert too_small.status_code == 422, too_small.text
