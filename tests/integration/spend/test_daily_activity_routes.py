import csv
import hashlib
import io
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from itertools import chain
from pathlib import Path
from typing import Final

import httpx
import pytest
from fastapi import FastAPI
from integration._support.client import JSON_OBJECT, Gateway, eventually, object_value, string_value
from integration._support.daily_activity import DAY, TAG_SPEND, daily_rows, seeded_row
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.upstream import JsonResponse, delete_scenario, register_scenario
from integration.spend.test_daily_activity_repository import _daily_activity_database, _PrismaDatabase, _repository
from pydantic import JsonValue

from litellm import constants
from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.daily_activity_routes import (
    get_daily_activity_prisma_client,
    get_daily_activity_repository,
)
from litellm.proxy.management_endpoints.daily_activity_routes import (
    router as daily_activity_router,
)
from litellm.proxy.management_endpoints.internal_user_endpoints import router as internal_user_router
from litellm.proxy.management_endpoints.team_endpoints import router as team_router
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    DailyActivityKeyPageResponse,
    SpendAnalyticsPaginatedResponse,
)


def _delete_organization(gateway: Gateway, organization_id: str) -> None:
    response: Final = gateway.request("DELETE", "/organization/delete", {"organization_ids": [organization_id]})
    assert response.status_code == 200, response.text


def _delete_tag(gateway: Gateway, tag: str) -> None:
    response: Final = gateway.request("POST", "/tag/delete", {"name": tag})
    assert response.status_code == 200, response.text


def _delete_end_user(gateway: Gateway, end_user_id: str) -> None:
    response: Final = gateway.request("POST", "/end_user/delete", {"user_ids": [end_user_id]})
    assert response.status_code == 200, response.text


def _delete_agent(gateway: Gateway, agent_id: str) -> None:
    response: Final = gateway.request("DELETE", f"/v1/agents/{agent_id}")
    assert response.status_code == 200, response.text


def _daily_activity_request(
    gateway: Gateway,
    *,
    model: str,
    key: str,
    end_user_id: str,
    tag: str,
    request_number: int,
) -> None:
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        JSON_OBJECT.validate_python(
            {
                "model": model,
                "messages": [{"role": "user", "content": f"daily activity {request_number}"}],
                "metadata": {"tags": [tag]},
                "user": end_user_id,
            }
        ),
        key=key,
    )
    assert response.status_code == 200, response.text


def _aggregate_result_api_keys(result: object) -> tuple[str, ...]:
    result_body: Final = object_value(result)
    breakdown: Final = object_value(result_body["breakdown"])
    api_keys: Final = object_value(breakdown["api_keys"])
    return tuple(api_keys)


def _aggregate_top_keys(results: object) -> frozenset[str]:
    assert isinstance(results, list)
    api_keys_by_result: Final = tuple(_aggregate_result_api_keys(result) for result in results)
    return frozenset(chain.from_iterable(api_keys_by_result))


def _assert_entity_activity_routes(
    gateway: Gateway,
    *,
    prefix: str,
    entity_param: str,
    entity_id: str,
    table: str,
    entity_column: str,
    date_params: dict[str, str],
    target_digest: str,
    model: str,
) -> None:
    params: Final = {**date_params, entity_param: entity_id}
    persisted_rows: Final = eventually(
        lambda: read_rows(
            f'SELECT api_key FROM "{table}" WHERE "{entity_column}"=%s AND date BETWEEN %s AND %s',
            (entity_id, date_params["start_date"], date_params["end_date"]),
        ),
        lambda rows: len(rows) == 6,
        seconds=70,
    )
    aggregated: Final = gateway.request(
        "GET",
        f"{prefix}/daily/activity/aggregated",
        params={**params, "api_key_limit": "3"},
    )
    assert aggregated.status_code == 200, aggregated.text
    aggregate_body: Final = object_value(aggregated.json())
    metadata: Final = object_value(aggregate_body["metadata"])
    total_api_keys: Final = metadata["total_api_keys"]
    api_key_limit: Final = metadata["api_key_limit"]
    assert isinstance(total_api_keys, int) and total_api_keys == 6, aggregated.text
    assert isinstance(api_key_limit, int) and api_key_limit == 3, aggregated.text
    assert total_api_keys > api_key_limit, aggregated.text
    assert metadata["total_api_requests"] == 8, aggregated.text
    top_api_keys: Final = _aggregate_top_keys(aggregate_body["results"])
    assert target_digest not in top_api_keys, aggregated.text
    ranked_rows: Final = read_rows(
        f'SELECT api_key FROM "{table}" WHERE "{entity_column}"=%s AND date BETWEEN %s AND %s '
        "AND api_key <> %s GROUP BY api_key ORDER BY SUM(spend::numeric) DESC, api_key",
        (
            entity_id,
            date_params["start_date"],
            date_params["end_date"],
            constants.PTU_SENTINEL_API_KEY,
        ),
    )
    ranked_keys: Final = tuple(string_value(row["api_key"]) for row in ranked_rows)
    page_responses: Final = tuple(
        gateway.request(
            "GET",
            f"{prefix}/daily/activity/aggregated/keys",
            params={**params, "offset": str(offset), "limit": "2"},
        )
        for offset in range(0, len(ranked_keys), 2)
    )
    assert all(response.status_code == 200 for response in page_responses), tuple(
        response.text for response in page_responses
    )
    page_bodies: Final = tuple(
        DailyActivityKeyPageResponse.model_validate_json(response.content) for response in page_responses
    )
    page_api_keys: Final = tuple(tuple(row.api_key for row in body.api_keys) for body in page_bodies)
    paged_keys: Final = tuple(chain.from_iterable(page_api_keys))
    assert tuple(body.total_api_keys for body in page_bodies) == (6,) * len(page_bodies)
    assert paged_keys == ranked_keys
    assert len(paged_keys) == len(frozenset(paged_keys))
    assert frozenset(paged_keys[:3]) == top_api_keys, aggregated.text

    key_details: Final = gateway.request(
        "GET",
        f"{prefix}/daily/activity/aggregated",
        params={**params, "api_key": target_digest},
    )
    assert key_details.status_code == 200, key_details.text
    key_details_body: Final = JSON_OBJECT.validate_json(key_details.content)
    assert object_value(key_details_body["metadata"])["total_api_keys"] == 1, key_details.text
    assert _aggregate_top_keys(key_details_body["results"]) == frozenset((target_digest,)), key_details.text

    searched: Final = gateway.request(
        "GET",
        f"{prefix}/daily/activity/aggregated/search",
        params={**params, "search": target_digest},
    )
    assert searched.status_code == 200, searched.text
    search_body: Final = object_value(searched.json())
    search_rows: Final = search_body["api_keys"]
    assert isinstance(search_rows, list) and len(search_rows) == 1, searched.text
    assert object_value(search_rows[0])["api_key"] == target_digest, searched.text

    top_keys: Final = gateway.request(
        "GET",
        f"{prefix}/daily/activity/aggregated/model_top_keys",
        params={**params, "model_group": model},
    )
    assert top_keys.status_code == 200, top_keys.text
    top_body: Final = object_value(top_keys.json())
    top_rows: Final = top_body["api_keys"]
    assert isinstance(top_rows, list) and len(top_rows) == 5, top_keys.text
    top_spends: Final = tuple(object_value(object_value(row)["metrics"])["spend"] for row in top_rows[:2])
    assert top_spends == (
        pytest.approx(0.12),
        pytest.approx(0.12),
    ), top_keys.text

    exported: Final = gateway.request(
        "GET",
        f"{prefix}/daily/activity/export",
        params={**params, "export_type": "daily_with_keys"},
    )
    assert exported.status_code == 200, exported.text
    export_rows: Final = tuple(csv.reader(io.StringIO(exported.text)))
    assert len(export_rows) == len(persisted_rows) + 1, exported.text


@pytest.mark.timeout(90)
def test_daily_activity_routes_cover_all_entities_and_bounded_key_search(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy(gateway, tmp_path, {}) as proxy:
        _assert_daily_activity_routes(proxy)


def _assert_daily_activity_routes(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        target_scenario_id: Final = f"usage-cache-{uuid.uuid4().hex}"
        target_response: Final = JsonResponse(
            content_type="application/json",
            body=JSON_OBJECT.validate_python(
                {
                    "id": "$UNIQUE_ID",
                    "object": "chat.completion",
                    "created": 1_700_000_000,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "cached response"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 40,
                        "completion_tokens": 20,
                        "total_tokens": 60,
                        "prompt_tokens_details": {"cached_tokens": 20},
                    },
                }
            ),
        )
        target_upstream: Final = register_scenario(target_scenario_id, target_response)
        scenario.cleanups.callback(delete_scenario, target_upstream)
        cache_model: Final = scenario.model(
            api_base=target_upstream.api_base(),
            api_key=target_scenario_id,
            input_cost_per_token=0.0,
            output_cost_per_token=0.0,
        )
        organization: Final = gateway.post(
            "/organization/new",
            {"organization_alias": f"integration-{uuid.uuid4().hex}"},
        )
        organization_id: Final = string_value(organization["organization_id"])
        scenario.cleanups.callback(_delete_organization, gateway, organization_id)
        team_id: Final = scenario.team(organization_id=organization_id)
        user_id: Final = scenario.user()
        tag: Final = f"integration-{uuid.uuid4().hex}"
        gateway.post("/tag/new", {"name": tag})
        scenario.cleanups.callback(_delete_tag, gateway, tag)
        end_user_id: Final = f"integration-{uuid.uuid4().hex}"
        gateway.post("/end_user/new", {"user_id": end_user_id})
        scenario.cleanups.callback(_delete_end_user, gateway, end_user_id)
        agent_response: Final = gateway.request(
            "POST",
            "/v1/agents",
            {
                "agent_name": f"integration-{uuid.uuid4().hex}",
                "agent_card_params": {
                    "protocolVersion": "0.3",
                    "name": "integration",
                    "description": "integration agent",
                    "url": "http://127.0.0.1:1/agent",
                    "version": "1",
                    "capabilities": {},
                    "defaultInputModes": ["text"],
                    "defaultOutputModes": ["text"],
                    "skills": [],
                },
            },
        )
        assert agent_response.status_code == 200, agent_response.text
        agent_id: Final = string_value(object_value(agent_response.json())["agent_id"])
        scenario.cleanups.callback(_delete_agent, gateway, agent_id)
        keys: Final = tuple(
            scenario.key(
                models=[model, cache_model],
                team_id=team_id,
                user_id=user_id,
                organization_id=organization_id,
                agent_id=agent_id,
            )
            for _ in range(5)
        )
        target_key: Final = scenario.key(
            models=[model, cache_model],
            team_id=team_id,
            user_id=user_id,
            organization_id=organization_id,
            agent_id=agent_id,
        )
        for request_number, key in enumerate(keys):
            _daily_activity_request(
                gateway,
                model=model,
                key=key,
                end_user_id=end_user_id,
                tag=tag,
                request_number=request_number,
            )
        for request_number, key in enumerate(keys[:2]):
            _daily_activity_request(
                gateway,
                model=model,
                key=key,
                end_user_id=end_user_id,
                tag=tag,
                request_number=100 + request_number,
            )
        target_request: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            JSON_OBJECT.validate_python(
                {
                    "model": cache_model,
                    "messages": [{"role": "user", "content": "cached activity response"}],
                    "metadata": {"tags": [tag]},
                    "user": end_user_id,
                }
            ),
            key=target_key,
        )
        assert target_request.status_code == 200, target_request.text

        today: Final = datetime.now(timezone.utc).date()
        start_date: Final = (today - timedelta(days=1)).isoformat()
        end_date: Final = (today + timedelta(days=1)).isoformat()
        date_params: Final = {"start_date": start_date, "end_date": end_date, "timezone": "0"}
        route_cases: Final = (
            ("/user", "user_id", user_id, "LiteLLM_DailyUserSpend", "user_id"),
            ("/team", "team_ids", team_id, "LiteLLM_DailyTeamSpend", "team_id"),
            ("/tag", "tags", tag, "LiteLLM_DailyTagSpend", "tag"),
            (
                "/organization",
                "organization_ids",
                organization_id,
                "LiteLLM_DailyOrganizationSpend",
                "organization_id",
            ),
            ("/customer", "end_user_ids", end_user_id, "LiteLLM_DailyEndUserSpend", "end_user_id"),
            ("/agent", "agent_ids", agent_id, "LiteLLM_DailyAgentSpend", "agent_id"),
        )
        target_digest: Final = hashlib.sha256(target_key.encode()).hexdigest()
        for prefix, entity_param, entity_id, table, entity_column in route_cases:
            _assert_entity_activity_routes(
                gateway,
                prefix=prefix,
                entity_param=entity_param,
                entity_id=entity_id,
                table=table,
                entity_column=entity_column,
                date_params=date_params,
                target_digest=target_digest,
                model=model,
            )
        user_cache_keys: Final = gateway.request(
            "GET",
            "/user/daily/activity/aggregated/cache_leakage_keys",
            params={**date_params, "user_id": user_id},
        )
        assert user_cache_keys.status_code == 200, user_cache_keys.text
        cache_rows: Final = object_value(user_cache_keys.json())["api_keys"]
        assert isinstance(cache_rows, list) and cache_rows, user_cache_keys.text
        cache_api_keys: Final = tuple(string_value(object_value(row)["api_key"]) for row in cache_rows)
        assert target_digest in cache_api_keys, user_cache_keys.text


def _daily_tag_metrics(
    spend: float,
    prompt_tokens: int,
    completion_tokens: int,
    api_requests: int = 1,
) -> dict[str, JsonValue]:
    return {
        "spend": spend,
        "flat_cost": 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "compression_saved_tokens": 0,
        "compression_savings_spend": 0.0,
        "prompt_caching_savings_spend": 0.0,
        "gateway_injected_caching_savings_spend": 0.0,
        "autorouter_savings_spend": 0.0,
        "total_tokens": prompt_tokens + completion_tokens,
        "successful_requests": api_requests,
        "failed_requests": 0,
        "api_requests": api_requests,
        "total_response_time_ms": 0,
        "timed_requests": 0,
    }


def _tag_activity(
    gateway: Gateway,
    *,
    key: str,
    tags: str,
    api_key: str | None = None,
    model: str | None = None,
    page: int = 1,
    page_size: int = 10,
) -> httpx.Response:
    params: Final = {
        "tags": tags,
        "start_date": DAY,
        "end_date": DAY,
        "page": str(page),
        "page_size": str(page_size),
        **({"api_key": api_key} if api_key is not None else {}),
        **({"model": model} if model is not None else {}),
    }
    return gateway.request("GET", "/tag/daily/activity", params=params, key=key)


def test_tag_daily_activity_scopes_internal_user_keys_and_filters(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        user_one: Final = scenario.user(user_role="internal_user")
        user_two: Final = scenario.user(user_role="internal_user")
        key_one_a: Final = scenario.key(user_id=user_one, key_alias="tag-activity-a")
        key_one_b: Final = scenario.key(user_id=user_one, key_alias="tag-activity-b")
        key_two: Final = scenario.key(user_id=user_two, key_alias="tag-activity-other")
        digest_one_a: Final = hashlib.sha256(key_one_a.encode()).hexdigest()
        digest_one_b: Final = hashlib.sha256(key_one_b.encode()).hexdigest()
        digest_two: Final = hashlib.sha256(key_two.encode()).hexdigest()
        standard_model_row: Final = seeded_row(TAG_SPEND, "tag", "alpha", digest_one_a, DAY)
        rows: Final = (
            standard_model_row,
            seeded_row(TAG_SPEND, "tag", "beta", digest_one_b, DAY),
            seeded_row(TAG_SPEND, "tag", "gamma", digest_one_a, DAY),
            seeded_row(TAG_SPEND, "tag", "alpha", digest_two, DAY),
            replace(standard_model_row, model="gpt-4o"),
        )
        with daily_rows(rows):
            scoped: Final = _tag_activity(
                gateway,
                key=key_one_a,
                tags="alpha,beta",
            )
            assert scoped.status_code == 200, scoped.text
            scoped_body: Final = SpendAnalyticsPaginatedResponse.model_validate_json(scoped.content)
            assert scoped_body.metadata.total_spend == 0.75, scoped.text
            assert scoped_body.metadata.total_prompt_tokens == 30, scoped.text
            assert scoped_body.metadata.total_completion_tokens == 15, scoped.text
            assert scoped_body.metadata.total_tokens == 45, scoped.text
            assert scoped_body.metadata.total_api_requests == 3, scoped.text
            assert len(scoped_body.results) == 1, scoped.text
            scoped_day: Final = scoped_body.results[0]
            assert scoped_day.date.isoformat() == DAY, scoped.text
            assert scoped_day.metrics.model_dump(mode="json") == _daily_tag_metrics(0.75, 30, 15, 3), scoped.text
            assert tuple(sorted(scoped_day.breakdown.entities)) == ("alpha", "beta"), scoped.text
            assert scoped_day.breakdown.entities["alpha"].metrics.model_dump(mode="json") == _daily_tag_metrics(
                0.5, 20, 10, 2
            ), scoped.text
            assert scoped_day.breakdown.entities["beta"].metrics.model_dump(mode="json") == _daily_tag_metrics(
                0.25, 10, 5
            ), scoped.text

            hidden_key: Final = _tag_activity(
                gateway,
                key=key_one_a,
                tags="alpha,beta",
                api_key=digest_two,
            )
            assert hidden_key.status_code == 200, hidden_key.text
            assert SpendAnalyticsPaginatedResponse.model_validate_json(hidden_key.content).model_dump(
                mode="json"
            ) == SpendAnalyticsPaginatedResponse(results=[]).model_dump(mode="json"), hidden_key.text

            selected_key: Final = _tag_activity(
                gateway,
                key=key_one_a,
                tags="alpha,beta",
                api_key=digest_one_b,
            )
            assert selected_key.status_code == 200, selected_key.text
            selected_body: Final = SpendAnalyticsPaginatedResponse.model_validate_json(selected_key.content)
            assert selected_body.metadata.total_spend == 0.25, selected_key.text
            assert selected_body.metadata.total_api_requests == 1, selected_key.text
            assert selected_body.results[0].breakdown.entities.keys() == {"beta"}, selected_key.text
            assert selected_body.results[0].metrics.model_dump(mode="json") == _daily_tag_metrics(0.25, 10, 5), (
                selected_key.text
            )

            model_filter: Final = _tag_activity(
                gateway,
                key=key_one_a,
                tags="alpha,beta",
                model="gpt-4o",
            )
            assert model_filter.status_code == 200, model_filter.text
            model_body: Final = SpendAnalyticsPaginatedResponse.model_validate_json(model_filter.content)
            assert model_body.metadata.total_spend == 0.25, model_filter.text
            assert model_body.results[0].metrics.model_dump(mode="json") == _daily_tag_metrics(0.25, 10, 5), (
                model_filter.text
            )

            first_page: Final = _tag_activity(
                gateway,
                key=key_one_a,
                tags="alpha",
                page=1,
                page_size=1,
            )
            second_page: Final = _tag_activity(
                gateway,
                key=key_one_a,
                tags="alpha",
                page=2,
                page_size=1,
            )
            assert first_page.status_code == 200, first_page.text
            assert second_page.status_code == 200, second_page.text
            page_one: Final = SpendAnalyticsPaginatedResponse.model_validate_json(first_page.content)
            page_two: Final = SpendAnalyticsPaginatedResponse.model_validate_json(second_page.content)
            assert (
                page_one.metadata.page,
                page_one.metadata.total_pages,
                page_one.metadata.has_more,
                page_two.metadata.page,
                page_two.metadata.total_pages,
                page_two.metadata.has_more,
            ) == (1, 2, True, 2, 2, False), (first_page.text, second_page.text)
            page_results: Final = (*page_one.results, *page_two.results)
            models_seen: Final = frozenset(
                model_name for result in page_results for model_name in result.breakdown.models
            )
            assert len(page_results) == 2 and models_seen == frozenset(("gpt-4o-mini", "gpt-4o")), (
                first_page.text,
                second_page.text,
            )

            admin_control: Final = _tag_activity(
                gateway,
                key=gateway.key,
                tags="alpha",
                api_key=digest_two,
            )
            assert admin_control.status_code == 200, admin_control.text
            admin_body: Final = SpendAnalyticsPaginatedResponse.model_validate_json(admin_control.content)
            assert admin_body.metadata.total_spend == 0.25, admin_control.text
            assert admin_body.results[0].metrics.model_dump(mode="json") == _daily_tag_metrics(0.25, 10, 5), (
                admin_control.text
            )


async def _assert_route_matches_golden(client: httpx.AsyncClient, route: str, golden_name: str) -> None:
    response: Final = await client.get(
        route,
        params={"start_date": "2026-06-01", "end_date": "2026-06-01"},
    )
    assert response.status_code == 200, response.text
    golden: Final = (Path(__file__).parent / "golden" / golden_name).read_text()
    expected: Final = JSON_OBJECT.validate_json(golden)
    actual: Final = object_value(response.json())
    assert actual == expected, route


@pytest.mark.asyncio
async def test_existing_activity_routes_match_base_branch_goldens(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _daily_activity_database() as database:
        repository: Final = _repository(database)
        app: Final = FastAPI()
        app.include_router(internal_user_router)
        app.include_router(team_router)
        app.include_router(daily_activity_router)
        monkeypatch.setattr(proxy_server, "prisma_client", _PrismaDatabase(database))
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
            user_id="integration-admin", user_role=LitellmUserRoles.PROXY_ADMIN
        )
        app.dependency_overrides[get_daily_activity_prisma_client] = lambda: _PrismaDatabase(database)
        app.dependency_overrides[get_daily_activity_repository] = lambda: repository

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            route_goldens: Final = (
                ("/user/daily/activity", "daily_activity_user_paginated.json"),
                ("/user/daily/activity/aggregated", "daily_activity_user_aggregated.json"),
                ("/team/daily/activity", "daily_activity_team_paginated.json"),
                ("/team/daily/activity/aggregated", "daily_activity_team_aggregated.json"),
            )
            for route, golden_name in route_goldens:
                await _assert_route_matches_golden(client, route, golden_name)


@pytest.mark.asyncio
async def test_user_key_pages_and_details_respect_caller_scope() -> None:
    async with _daily_activity_database() as database:
        await database.query_raw(
            'INSERT INTO "LiteLLM_UserTable" (user_id, user_email, models) VALUES ($1, $2, $3)',
            "user-2",
            "other@example.test",
            [],
        )
        await database.query_raw(
            """
            INSERT INTO "LiteLLM_VerificationToken"
                (token, key_alias, team_id, user_id, metadata, models)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6)
            """,
            "key-other-user",
            "Other user key",
            None,
            "user-2",
            "{}",
            [],
        )
        await database.query_raw(
            """
            INSERT INTO "LiteLLM_DailyUserSpend"
                (id, user_id, date, api_key, model, model_group, custom_llm_provider,
                 mcp_namespaced_tool_name, endpoint, prompt_tokens, completion_tokens,
                 cache_read_input_tokens, cache_creation_input_tokens, spend, api_requests,
                 successful_requests, failed_requests, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18::timestamp)
            """,
            "other-user-row",
            "user-2",
            "2026-06-01",
            "key-other-user",
            "model",
            "",
            "provider-a",
            None,
            "/v1/chat/completions",
            1,
            1,
            0,
            0,
            50.0,
            1,
            1,
            0,
            "2026-06-01 12:00:00",
        )
        repository: Final = _repository(database)
        app: Final = FastAPI()
        app.include_router(daily_activity_router)
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
            user_id="user-1",
            user_role=LitellmUserRoles.INTERNAL_USER,
        )
        app.dependency_overrides[get_daily_activity_prisma_client] = lambda: _PrismaDatabase(database)
        app.dependency_overrides[get_daily_activity_repository] = lambda: repository

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            params: Final = {"start_date": "2026-06-01", "end_date": "2026-06-01"}
            page: Final = await client.get(
                "/user/daily/activity/aggregated/keys",
                params={**params, "user_id": "user-1", "limit": 100},
            )
            assert page.status_code == 200, page.text
            page_body: Final = DailyActivityKeyPageResponse.model_validate_json(page.content)
            page_keys: Final = frozenset(row.api_key for row in page_body.api_keys)
            assert page_body.total_api_keys == len(page_keys) == 5
            assert "key-other-user" not in page_keys

            denied: Final = await client.get(
                "/user/daily/activity/aggregated/keys",
                params={**params, "user_id": "user-2"},
            )
            assert denied.status_code == 403, denied.text

            own_details: Final = await client.get(
                "/user/daily/activity/aggregated",
                params={**params, "user_id": "user-1", "api_key": "key-a"},
            )
            assert own_details.status_code == 200, own_details.text
            own_body: Final = JSON_OBJECT.validate_json(own_details.content)
            assert object_value(own_body["metadata"])["total_api_keys"] == 1
            assert _aggregate_top_keys(own_body["results"]) == frozenset(("key-a",))

            other_details: Final = await client.get(
                "/user/daily/activity/aggregated",
                params={**params, "user_id": "user-1", "api_key": "key-other-user"},
            )
            assert other_details.status_code == 200, other_details.text
            other_body: Final = JSON_OBJECT.validate_json(other_details.content)
            assert object_value(other_body["metadata"])["total_api_keys"] == 0
            assert _aggregate_top_keys(other_body["results"]) == frozenset()


@pytest.mark.asyncio
async def test_team_routes_exclusion_keeps_unassigned_keys() -> None:
    async with _daily_activity_database(include_team_exclusion_activity=True) as database:
        repository: Final = _repository(database)
        app: Final = FastAPI()
        app.include_router(daily_activity_router)
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
            user_id="integration-admin", user_role=LitellmUserRoles.PROXY_ADMIN
        )
        app.dependency_overrides[get_daily_activity_prisma_client] = lambda: _PrismaDatabase(database)
        app.dependency_overrides[get_daily_activity_repository] = lambda: repository

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            params: Final = {
                "start_date": "2026-06-04",
                "end_date": "2026-06-04",
                "exclude_team_ids": "litellm-dashboard",
            }
            surviving_keys: Final = frozenset(("key-excluded-null", "key-excluded-empty", "key-excluded-normal"))

            aggregated: Final = await client.get("/team/daily/activity/aggregated", params=params)
            assert aggregated.status_code == 200, aggregated.text
            aggregated_body: Final = JSON_OBJECT.validate_json(aggregated.content)
            assert object_value(aggregated_body["metadata"])["total_spend"] == 23.0
            assert object_value(aggregated_body["metadata"])["total_api_keys"] == 3
            assert _aggregate_top_keys(aggregated_body["results"]) == surviving_keys

            page: Final = await client.get("/team/daily/activity/aggregated/keys", params={**params, "limit": 10})
            assert page.status_code == 200, page.text
            page_body: Final = DailyActivityKeyPageResponse.model_validate_json(page.content)
            assert page_body.total_api_keys == 3
            assert frozenset(row.api_key for row in page_body.api_keys) == surviving_keys
