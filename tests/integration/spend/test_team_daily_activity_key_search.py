import uuid
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from pydantic import JsonValue

_SEARCH_PATH: Final = "/team/daily/activity/aggregated/search"


def _range_around_today() -> dict[str, str]:
    today: Final = datetime.now(timezone.utc)
    return {
        "start_date": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
        "end_date": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
        "timezone": "0",
    }


def _team_key_breakdown(body: dict[str, JsonValue], team: str) -> dict[str, JsonValue]:
    results: Final = body["results"]
    assert isinstance(results, list) and len(results) == 1, body
    entities: Final = object_value(object_value(object_value(results[0])["breakdown"])["entities"])
    return object_value(object_value(entities[team])["api_key_breakdown"])


def test_team_key_search_returns_only_the_matching_key_spend_by_alias_and_by_hash(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team: Final = scenario.team(models=[model])
        needle_alias: Final = f"needle-{uuid.uuid4().hex}"
        needle: Final = scenario.key(team_id=team, models=[model], key_alias=needle_alias)
        other: Final = scenario.key(team_id=team, models=[model], key_alias=f"other-{uuid.uuid4().hex}")
        needle_digest: Final = sha256(needle.encode()).hexdigest()
        other_digest: Final = sha256(other.encode()).hexdigest()
        for key in (needle, other):
            reply: Final = gateway.chat(model, key=key, text=f"key search {uuid.uuid4().hex}")
            assert object_value(reply["usage"])["total_tokens"] == 40, reply
        daily: Final = eventually(
            lambda: read_rows('SELECT api_key, spend FROM "LiteLLM_DailyTeamSpend" WHERE team_id=%s', (team,)),
            lambda values: sorted(row["api_key"] for row in values) == sorted((needle_digest, other_digest)),
            seconds=70,
        )
        assert all(float(row["spend"]) == pytest.approx(0.06) for row in daily), daily
        for search in (needle_alias.upper(), needle_digest):
            response: Final = gateway.request(
                "GET", _SEARCH_PATH, params={"team_ids": team, "search": search, **_range_around_today()}
            )
            assert response.status_code == 200, response.text
            body: Final = object_value(response.json())
            assert object_value(body["metadata"])["total_spend"] == pytest.approx(0.06), response.text
            per_key: Final = _team_key_breakdown(body, team)
            assert set(per_key) == {needle_digest}, response.text
            assert object_value(object_value(per_key[needle_digest])["metrics"])["spend"] == pytest.approx(0.06)


def test_team_key_search_is_scoped_to_the_teams_the_caller_belongs_to(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team: Final = scenario.team(models=[model])
        needle_alias: Final = f"needle-{uuid.uuid4().hex}"
        needle: Final = scenario.key(team_id=team, models=[model], key_alias=needle_alias)
        needle_digest: Final = sha256(needle.encode()).hexdigest()
        reply: Final = gateway.chat(model, key=needle, text=f"key search {uuid.uuid4().hex}")
        assert object_value(reply["usage"])["total_tokens"] == 40, reply
        eventually(
            lambda: read_rows('SELECT api_key FROM "LiteLLM_DailyTeamSpend" WHERE team_id=%s', (team,)),
            lambda values: [row["api_key"] for row in values] == [needle_digest],
            seconds=70,
        )
        outsider: Final = scenario.user(user_role="internal_user")
        outsider_team: Final = scenario.team(models=[model], members_with_roles=[{"user_id": outsider, "role": "user"}])
        outsider_key: Final = scenario.key(user_id=outsider, team_id=outsider_team, models=[model])
        params: Final = {"search": needle_alias, **_range_around_today()}
        admin_view: Final = gateway.request("GET", _SEARCH_PATH, params={"team_ids": team, **params})
        assert admin_view.status_code == 200, admin_view.text
        assert set(_team_key_breakdown(object_value(admin_view.json()), team)) == {needle_digest}, admin_view.text
        own_teams_view: Final = gateway.request("GET", _SEARCH_PATH, params=params, key=outsider_key)
        assert own_teams_view.status_code == 200, own_teams_view.text
        own_teams_body: Final = object_value(own_teams_view.json())
        assert own_teams_body["results"] == [], own_teams_view.text
        assert object_value(own_teams_body["metadata"])["total_api_keys"] == 0, own_teams_view.text
        foreign_team_view: Final = gateway.request(
            "GET", _SEARCH_PATH, params={"team_ids": team, **params}, key=outsider_key
        )
        assert foreign_team_view.status_code == 404, foreign_team_view.text


def test_team_key_search_excludes_teams_inside_the_where(gateway: Gateway) -> None:
    """The dashboard always sends exclude_team_ids; a matching key in an excluded
    team with higher spend must not consume a take slot nor appear in the result."""
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team_keep: Final = scenario.team(models=[model])
        team_drop: Final = scenario.team(models=[model])
        shared_alias: Final = f"needle-{uuid.uuid4().hex}"
        keep: Final = scenario.key(team_id=team_keep, models=[model], key_alias=f"{shared_alias}-keep")
        drop: Final = scenario.key(team_id=team_drop, models=[model], key_alias=f"{shared_alias}-drop")
        keep_digest: Final = sha256(keep.encode()).hexdigest()
        drop_digest: Final = sha256(drop.encode()).hexdigest()
        for _ in range(2):
            reply: Final = gateway.chat(model, key=drop, text=f"key search {uuid.uuid4().hex}")
            assert object_value(reply["usage"])["total_tokens"] == 40, reply
        reply = gateway.chat(model, key=keep, text=f"key search {uuid.uuid4().hex}")
        assert object_value(reply["usage"])["total_tokens"] == 40, reply
        eventually(
            lambda: read_rows(
                'SELECT api_key, spend FROM "LiteLLM_DailyTeamSpend" WHERE team_id IN (%s, %s)',
                (team_keep, team_drop),
            ),
            lambda values: sorted(row["api_key"] for row in values) == sorted((keep_digest, drop_digest)),
            seconds=70,
        )
        response: Final = gateway.request(
            "GET",
            _SEARCH_PATH,
            params={"search": shared_alias, "exclude_team_ids": team_drop, **_range_around_today()},
        )
        assert response.status_code == 200, response.text
        body: Final = object_value(response.json())
        results: Final = body["results"]
        assert isinstance(results, list) and len(results) == 1, body
        entities: Final = object_value(object_value(object_value(results[0])["breakdown"])["entities"])
        assert set(entities) == {team_keep}, response.text
        assert set(_team_key_breakdown(body, team_keep)) == {keep_digest}, response.text
