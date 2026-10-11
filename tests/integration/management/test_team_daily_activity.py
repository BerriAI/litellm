from typing import Final

import pytest

from tests.integration._support.actors import Actor
from tests.integration._support.database import write_rows

pytestmark = pytest.mark.asyncio(loop_scope="session")


# GET /team/daily/activity and its /aggregated variant (same shared scope
# resolver, so the matrix must hold for both). A proxy admin (admin view) sees
# activity for any team. A non-admin is scoped to user_info.teams: a bare query
# defaults to its own teams (200), and an explicit team_ids filter naming a
# team it does not belong to is 404 (the VERIA-43 fix). Org admins have no
# team memberships, so they behave like a non-member for any specific team.
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


_SEEDED_SPEND: Final = {"alpha": 1.25, "beta": 2.5}


@pytest.fixture(scope="module")
def seeded_team_spend(actors) -> None:
    team_ids: Final = {"alpha": actors.team_alpha_id, "beta": actors.team_beta_id}
    for team, spend in _SEEDED_SPEND.items():
        write_rows(
            'INSERT INTO "LiteLLM_DailyTeamSpend" (id, team_id, date, api_key, model, spend, updated_at) '
            "VALUES (%s, %s, '2024-06-01', 'integration-daily-activity', 'gpt-4o', %s, now()) "
            "ON CONFLICT (id) DO NOTHING",
            (f"integration-daily-activity-{team}", team_ids[team], str(spend)),
        )


@pytest.mark.parametrize(
    "endpoint",
    ("/team/daily/activity", "/team/daily/activity/aggregated"),
    ids=("paginated", "aggregated"),
)
@pytest.mark.parametrize(
    "actor,team,expected_status",
    [(a, t, s) for (_id, a, t, s) in _CASES],
    ids=[c[0] for c in _CASES],
)
async def test_team_daily_activity_matrix(
    actor: Actor, team: str, expected_status: int, endpoint: str, gateway_client, actors, seeded_team_spend
):
    query = _DATES
    if team == "alpha":
        query += f"&team_ids={actors.team_alpha_id}"
    elif team == "beta":
        query += f"&team_ids={actors.team_beta_id}"

    resp = await gateway_client.get(
        f"{endpoint}?{query}",
        headers={"Authorization": f"Bearer {actors.keys[actor].cleartext}"},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value} -> {team}: {resp.status_code} {resp.text}"
    if actor == Actor.PROXY_ADMIN and team in _SEEDED_SPEND:
        assert resp.json()["metadata"]["total_spend"] == _SEEDED_SPEND[team], resp.text
