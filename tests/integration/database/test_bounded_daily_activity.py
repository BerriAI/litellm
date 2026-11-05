from collections.abc import Mapping
from datetime import date, timedelta
from hashlib import sha256
from typing import Final
from uuid import uuid4

import psycopg
import pytest
from integration._support.actors import ManagementDatabase
from integration._support.client import Gateway
from psycopg import sql
from pydantic import TypeAdapter

from litellm.repositories.bounded_daily_activity import BoundedDailyActivityRepository
from litellm.types.proxy.management_endpoints.common_daily_activity import BoundedDailyActivityResponse


@pytest.mark.asyncio(loop_scope="session")
async def test_bounded_activity_routes_keep_exact_totals_and_scoped_permissions(
    gateway: Gateway, db: ManagementDatabase
) -> None:
    repository: Final = BoundedDailyActivityRepository.from_prisma(db.prisma)
    identity: Final = uuid4().hex
    day: Final = date.today() - timedelta(days=3)
    with gateway.scenario() as scenario:
        owner: Final = scenario.user(user_role="internal_user")
        member: Final = scenario.user(user_role="internal_user")
        team: Final = scenario.team(
            members_with_roles=[{"user_id": owner, "role": "admin"}, {"user_id": member, "role": "user"}]
        )
        owner_key: Final = scenario.key(user_id=owner, team_id=team, key_type="management")
        member_key: Final = scenario.key(user_id=member, team_id=team, key_type="management")
        reporting_key: Final = sha256(owner_key.encode()).hexdigest()
        member_reporting_key: Final = sha256(member_key.encode()).hexdigest()
        for table, entity, column in (
            ("LiteLLM_DailyUserSpend", owner, "user_id"),
            ("LiteLLM_DailyTeamSpend", team, "team_id"),
        ):
            await repository.database.query_raw(
                f'''INSERT INTO "{table}" (id,{column},date,api_key,model,custom_llm_provider,
                    spend,api_requests,successful_requests,total_response_time_ms,timed_requests,updated_at)
                SELECT $1::text||'-'||m||'-'||d,$2::text,($3::date+d)::text,$4::text,
                    CASE WHEN m=1 THEN 'Other' WHEN m=137 THEN repeat('x',1025)
                         ELSE 'example-model-'||m END,'example-provider',
                    CASE WHEN m=1 THEN 10000 ELSE m END,10,10,2000,10,CURRENT_TIMESTAMP
                FROM generate_series(1,137) m CROSS JOIN generate_series(0,2) d''',
                identity,
                entity,
                day.isoformat(),
                reporting_key,
            )
        await repository.database.query_raw(
            """INSERT INTO "LiteLLM_DailyTeamSpend"
                (id,team_id,date,api_key,model,api_requests,successful_requests,updated_at)
                VALUES($1,$2,$3,$4,'member-model',7,7,CURRENT_TIMESTAMP)""",
            identity + "-member",
            team,
            day.isoformat(),
            member_reporting_key,
        )
        query: Final = {"start_date": day.isoformat(), "end_date": (day + timedelta(days=2)).isoformat()}
        try:
            path: Final = "/user/daily/activity/aggregated"
            legacy: Final = gateway.request("GET", path, params={**query, "api_key": reporting_key})
            assert legacy.status_code == 200 and "model_breakdown" not in legacy.json()
            result: Final = gateway.request(
                "GET",
                path,
                params={**query, "api_key": reporting_key, "bounded": "true", "include_model_breakdown": "true"},
                key=owner_key,
            )
            assert result.status_code == 200, result.text
            bounded: Final = BoundedDailyActivityResponse.model_validate(result.json())
            assert bounded.metadata.model_dump(exclude={"api_key_limit"}) == {
                key: value for key, value in legacy.json()["metadata"].items() if key != "api_key_limit"
            }
            models: Final = bounded.model_breakdown
            assert models is not None and models.total_models == 137 and len(models.models) == 20
            assert models.models[0].model == "Other" and models.other is not None and models.other.model_count == 117
            assert all(len(item.model) <= 1024 for item in models.models)
            assert (
                sum(item.metrics.spend for item in models.models) + models.other.metrics.spend
                == bounded.metadata.total_spend
            )
            assert len(result.content) < len(legacy.content) / 10
            assert bounded.metadata.total_api_requests == 4110 and len(bounded.results) == 3
            largest: Final = gateway.request(
                "GET",
                path,
                params={
                    **query,
                    "api_key": reporting_key,
                    "bounded": "true",
                    "include_model_breakdown": "true",
                    "model_limit": "100",
                },
            )
            assert largest.status_code == 200 and len(largest.json()["model_breakdown"]["models"]) == 100
            assert (
                gateway.request("GET", path, params={**query, "bounded": "true", "model_limit": "101"}).status_code
                == 422
            )
            assert (
                gateway.request(
                    "GET",
                    path,
                    params={**query, "bounded": "true", "end_date": (day + timedelta(days=366)).isoformat()},
                ).status_code
                == 422
            )
            team_query: Final = {**query, "team_ids": team, "bounded": "true"}
            team_path: Final = "/team/daily/activity/aggregated"
            team_admin: Final = gateway.request("GET", team_path, params=team_query, key=owner_key)
            team_member: Final = gateway.request("GET", team_path, params=team_query, key=member_key)
            assert team_admin.status_code == team_member.status_code == 200
            assert team_admin.json()["metadata"]["total_api_requests"] == 4117
            assert team_member.json()["metadata"]["total_api_requests"] == 7
            assert team_admin.json()["model_breakdown"] is None
            schemas: Final = TypeAdapter(tuple[Mapping[str, str], ...]).validate_python(
                await repository.database.query_raw("SELECT current_schema() AS name")
            )
            assert gateway.database_url is not None
            with psycopg.connect(gateway.database_url) as blocker:
                blocker.execute(
                    sql.SQL("LOCK TABLE {} IN ACCESS EXCLUSIVE MODE").format(
                        sql.Identifier(schemas[0]["name"], "LiteLLM_DailyUserSpend")
                    )
                )
                busy: Final = gateway.request("GET", path, params={**query, "bounded": "true"})
                assert busy.status_code == 503 and busy.headers["Retry-After"] == "5"
            assert gateway.request("GET", path, params={**query, "bounded": "true"}).status_code == 200
        finally:
            await repository.database.query_raw('DELETE FROM "LiteLLM_DailyUserSpend" WHERE user_id=$1', owner)
            await repository.database.query_raw('DELETE FROM "LiteLLM_DailyTeamSpend" WHERE team_id=$1', team)
