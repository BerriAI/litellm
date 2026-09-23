import os
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter
from redis import Redis

from tests.integration._support.client import Gateway, eventually, object_value, string_value
from tests.integration._support.database import read_rows

_CACHED_BUDGET: Final = TypeAdapter(dict[str, JsonValue])


@pytest.mark.covers("mgmt.team_member_budget.default_budget_is_cached_in_redis_as_json")
def test_team_member_default_budget_lands_in_redis_after_first_member_call(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user()
        team: Final = scenario.team(team_member_budget=25)
        key: Final = scenario.key(team_id=team, user_id=user, models=[model])
        teams: Final = read_rows('SELECT metadata FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,))
        assert len(teams) == 1, teams
        budget_id: Final = string_value(object_value(teams[0]["metadata"])["team_member_budget_id"])
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "member budget cache"}]},
            key=key,
        )
        assert response.status_code == 200, response.text
        with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as cache:
            cached: Final = eventually(
                lambda: cache.get(f"team_member_default_budget:{budget_id}"),
                lambda value: value is not None,
                seconds=10,
            )
        assert isinstance(cached, bytes), cached
        budget: Final = _CACHED_BUDGET.validate_json(cached)
        assert budget["budget_id"] == budget_id, cached
        assert budget["max_budget"] == 25, cached
