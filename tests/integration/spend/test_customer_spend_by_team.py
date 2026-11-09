import uuid
from hashlib import sha256
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, string_value
from integration._support.database import read_rows, write_rows

from litellm.types.proxy.management_endpoints.common_daily_activity import SpendAnalyticsPaginatedResponse


@pytest.mark.parametrize(
    "route",
    (
        "/customer/daily/activity",
        "/end_user/daily/activity",
        "/customer/daily/activity/aggregated",
        "/end_user/daily/activity/aggregated",
    ),
)
def test_customer_spend_by_current_team_and_team_filter(gateway: Gateway, route: str) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team_a: Final = scenario.team(team_alias="customer-team-a")
        team_b: Final = scenario.team(team_alias="customer-team-b")
        key_a: Final = scenario.key(models=[model], team_id=team_a)
        key_b: Final = string_value(gateway.post("/key/generate", {"models": [model], "team_id": team_b})["key"])
        scenario.cleanups.callback(gateway.request, "POST", "/key/delete", {"keys": [key_b]})
        key_unassigned: Final = scenario.key(models=[model])
        customer: Final = f"customer-{uuid.uuid4().hex}"
        scenario.cleanups.callback(gateway.request, "POST", "/customer/delete", {"user_ids": [customer]})
        for index, key in enumerate((key_a, key_b, key_unassigned)):
            response: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": f"hello {index}"}], "user": customer},
                key=key,
            )
            assert response.status_code == 200, response.text
        eventually(
            lambda: read_rows(
                'SELECT SUM(api_requests)::int AS requests FROM "LiteLLM_DailyEndUserSpend" WHERE end_user_id=%s',
                (customer,),
            ),
            lambda rows: rows[0]["requests"] == 3,
            seconds=70,
        )
        today: Final = datetime.now(timezone.utc).date()
        start: Final = today - timedelta(days=1)
        end: Final = today + timedelta(days=1)
        query: Final = f"{route}?end_user_ids={customer}&start_date={start}&end_date={end}&page_size=100"
        result: Final = gateway.request("GET", query)
        assert result.status_code == 200, result.text
        body: Final = SpendAnalyticsPaginatedResponse.model_validate_json(result.content)
        assert body.metadata.total_spend == pytest.approx(3 * (20 * 0.001 + 20 * 0.002))
        assert len(body.results) == 1
        teams: Final = body.results[0].breakdown.teams
        assert teams is not None
        assert set(teams) == {team_a, team_b, "Unassigned"}
        assert teams[team_a].metadata["team_alias"] == "customer-team-a"
        assert teams[team_b].metadata["team_alias"] == "customer-team-b"
        assert sum(bucket.metrics.spend for bucket in teams.values()) == pytest.approx(body.metadata.total_spend)
        assert all(bucket.metrics.api_requests == 1 for bucket in teams.values())
        assert all(bucket.metrics.total_tokens == 40 for bucket in teams.values())
        filtered: Final = gateway.request("GET", f"{query}&team_ids={team_a}")
        assert filtered.status_code == 200, filtered.text
        filtered_body: Final = SpendAnalyticsPaginatedResponse.model_validate_json(filtered.content)
        assert filtered_body.metadata.total_spend == pytest.approx(20 * 0.001 + 20 * 0.002)
        assert set(filtered_body.results[0].breakdown.teams or {}) == {team_a}
        stored_key: Final = sha256(key_a.encode()).hexdigest()
        stored_model: Final = str(
            read_rows(
                'SELECT model FROM "LiteLLM_DailyEndUserSpend" WHERE end_user_id=%s AND api_key=%s',
                (customer, stored_key),
            )[0]["model"]
        )
        combined: Final = gateway.request(
            "GET",
            f"{query}&team_ids={team_a}&api_key={stored_key}&model={stored_model}",
        )
        assert combined.status_code == 200, combined.text
        assert SpendAnalyticsPaginatedResponse.model_validate_json(
            combined.content
        ).metadata.total_spend == pytest.approx(20 * 0.001 + 20 * 0.002)
        excluded: Final = gateway.request(
            "GET",
            f"{query}&team_ids={team_b}&api_key={stored_key}&model={stored_model}",
        )
        assert excluded.status_code == 200, excluded.text
        assert SpendAnalyticsPaginatedResponse.model_validate_json(excluded.content).metadata.total_spend == 0
        empty: Final = gateway.request("GET", f"{query}&team_ids=missing-team")
        assert empty.status_code == 200, empty.text
        assert SpendAnalyticsPaginatedResponse.model_validate_json(empty.content).metadata.total_spend == 0
        scenario.delete_key(key_b)
        deleted: Final = gateway.request("GET", f"{query}&team_ids={team_b}")
        assert deleted.status_code == 200, deleted.text
        deleted_body: Final = SpendAnalyticsPaginatedResponse.model_validate_json(deleted.content)
        assert deleted_body.metadata.total_spend == pytest.approx(20 * 0.001 + 20 * 0.002)
        assert set(deleted_body.results[0].breakdown.teams or {}) == {team_b}
        for key, team in ((key_a, team_a), (key_b, team_b)):
            legacy_key: Final = sha256(sha256(key.encode()).hexdigest().encode()).hexdigest()
            legacy_customer: Final = f"legacy-{uuid.uuid4().hex}"
            scenario.cleanups.callback(
                write_rows, 'DELETE FROM "LiteLLM_DailyEndUserSpend" WHERE end_user_id=%s', (legacy_customer,)
            )
            write_rows(
                'INSERT INTO "LiteLLM_DailyEndUserSpend" '
                "(id,end_user_id,date,api_key,model,spend,api_requests,updated_at) "
                "VALUES (gen_random_uuid()::text,%s,%s,%s,%s,0.25,1,now())",
                (legacy_customer, today.isoformat(), legacy_key, stored_model),
            )
            legacy_query: Final = f"{route}?end_user_ids={legacy_customer}&start_date={start}&end_date={end}"
            for suffix in ("", f"&team_ids={team}"):
                legacy: Final = gateway.request("GET", legacy_query + suffix)
                assert legacy.status_code == 200, legacy.text
                legacy_body: Final = SpendAnalyticsPaginatedResponse.model_validate_json(legacy.content)
                assert legacy_body.metadata.total_spend == 0.25
                legacy_teams: Final = legacy_body.results[0].breakdown.teams or {}
                assert set(legacy_teams) == {team}
                assert legacy_teams[team].metrics.spend == legacy_body.metadata.total_spend
