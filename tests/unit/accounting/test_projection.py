import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm.accounting.projection import project
from litellm.constants import BILLED_RETRY_ATTEMPT_METADATA_KEY, INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.proxy_server import proxy_logging_obj
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.proxy.utils import PrismaClient
from litellm.types.utils import ModelResponse, Usage

_NOW: Final = datetime(2026, 11, 2, tzinfo=timezone.utc)
_FIELDS: Final = TypeAdapter(dict[str, object])


@pytest.mark.parametrize("status", ["success", "failure"])
@pytest.mark.parametrize("origin", [None, "auto_router", "billed_retry"])
@pytest.mark.parametrize("spend", [0.0, 0.25])
@pytest.mark.asyncio
async def test_daily_projection_matches_legacy_writer(status: str, origin: str | None, spend: float) -> None:
    payload: Final = get_logging_payload(
        kwargs={
            "model": "projection-test",
            "custom_llm_provider": "openai",
            "call_type": "acompletion",
            "litellm_call_id": "projection-request",
            "litellm_params": {"metadata": {"user_api_key": "key", "user_api_key_user_id": "user", "status": status}},
        },
        response_obj=ModelResponse(
            id="projection-request", model="projection-test", usage=Usage(prompt_tokens=10, completion_tokens=5)
        ),
        start_time=_NOW,
        end_time=_NOW,
    )
    metadata: Final = {
        **_FIELDS.validate_json(payload["metadata"]),
        **({BILLED_RETRY_ATTEMPT_METADATA_KEY: True} if origin == "billed_retry" else {}),
        **({INTERNAL_CALL_ORIGIN_METADATA_KEY: origin} if origin not in (None, "billed_retry") else {}),
    }
    row: Final = {**payload, "spend": spend, "metadata": json.dumps(metadata), "request_duration_ms": 25}
    legacy: Final = await DBSpendUpdateWriter()._common_add_spend_log_transaction_to_daily_transaction(
        row, PrismaClient(database_url="postgresql://unused", proxy_logging_obj=proxy_logging_obj), "user"
    )
    assert legacy is not None
    projected: Final = next(item for item in project(row) if item.table == "LiteLLM_DailyUserSpend")
    assert dict(projected.increments) == {name: legacy[name] for name, _ in projected.increments}
    assert dict(projected.key)["date"] == legacy["date"]
    assert dict(projected.key)["api_key"] == legacy["api_key"]


def test_projection_uses_frozen_savings_and_keeps_duplicate_entity_tags() -> None:
    row: Final[Mapping[str, object]] = {
        "request_id": "saved",
        "startTime": _NOW,
        "api_key": "key",
        "user": "user",
        "model": "unpriced-projection-test",
        "model_group": "group",
        "custom_llm_provider": "openai",
        "request_tags": '["tag", "tag"]',
        "spend": 0.5,
        "metadata": json.dumps(
            {
                "accounting_savings": {
                    "compression": 1.0,
                    "prompt_caching": -2.0,
                    "gateway_injected_caching": 3.0,
                    "autorouter": 4.0,
                },
                "autorouter_savings": 9000,
            }
        ),
    }
    contributions: Final = project(row)
    tags: Final = tuple(item for item in contributions if item.table == "LiteLLM_TagTable")
    daily_tags: Final = tuple(item for item in contributions if item.table == "LiteLLM_DailyTagSpend")
    assert sum(dict(item.increments)["spend"] for item in tags) == 1.0
    assert len(daily_tags) == 1
    assert {key: value for key, value in daily_tags[0].increments if key.endswith("savings_spend")} == {
        "compression_savings_spend": 1.0,
        "prompt_caching_savings_spend": -2.0,
        "gateway_injected_caching_savings_spend": 3.0,
        "autorouter_savings_spend": 4.0,
    }


def test_projection_carries_windows_and_does_not_reprice_guardrails() -> None:
    row: Final = {
        "request_id": "guarded",
        "startTime": _NOW,
        "api_key": "key",
        "spend": 0.75,
        "metadata": json.dumps(
            {
                "accounting_windows": [
                    {
                        "entity_type": "key",
                        "entity_id": "key",
                        "window_duration": "1d",
                        "window_start": _NOW.isoformat(),
                    }
                ],
                "guardrail_information": [
                    {
                        "guardrail_id": "guard",
                        "guardrail_usage": {"text": 3, "images": 2},
                        "guardrail_cost_by_unit": {"text": 0.3, "images": None},
                    }
                ],
            }
        ),
    }
    contributions: Final = project(row)
    window: Final = next(item for item in contributions if item.table == "LiteLLM_BudgetWindowSpend")
    units: Final = {
        dict(item.key)["usage_unit"]: dict(item.increments)
        for item in contributions
        if item.table == "LiteLLM_DailyGuardrailUsageUnits"
    }
    assert dict(window.increments) == {"spend": 0.75}
    assert dict(window.values)["window_start"] == _NOW.isoformat()
    assert units == {
        "text": {"units": 3, "cost": 0.3, "untracked_units": 0},
        "images": {"units": 2, "cost": 0.0, "untracked_units": 2},
    }


@pytest.mark.asyncio
async def test_entity_projection_matches_real_request_queues(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server
    from litellm.responses.streaming_iterator import drain_logging_tasks

    client: Final = PrismaClient(database_url="postgresql://unused", proxy_logging_obj=proxy_logging_obj)
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    monkeypatch.setattr(proxy_server, "disable_spend_logs", False)
    monkeypatch.setattr(client, "spend_log_transactions", [])
    monkeypatch.setattr(client, "model_usage_transactions", [])
    writer: Final = DBSpendUpdateWriter()
    charged: Final = await writer.update_database(
        token="sk-projection",
        user_id="user",
        end_user_id="end-user",
        team_id="team",
        org_id="org",
        project_id="project",
        kwargs={
            "model": "projection-test",
            "custom_llm_provider": "openai",
            "call_type": "acompletion",
            "litellm_call_id": "entity-request",
            "agent_id": "agent",
            "user": "end-user",
            "litellm_params": {
                "metadata": {
                    "user_api_key": "sk-projection",
                    "user_api_key_user_id": "user",
                    "user_api_key_team_id": "team",
                    "user_api_key_org_id": "org",
                    "user_api_key_project_id": "project",
                    "user_api_key_end_user_id": "end-user",
                    "tags": ["tag", "tag"],
                },
                "user": "end-user",
            },
        },
        completion_response=ModelResponse(
            id="entity-request", model="projection-test", usage=Usage(prompt_tokens=10, completion_tokens=5)
        ),
        start_time=_NOW,
        end_time=_NOW,
        response_cost=0.25,
    )
    await drain_logging_tasks()
    assert charged
    assert len(client.spend_log_transactions) == 1
    queued: Final = await writer.spend_update_queue.flush_all_updates_from_in_memory_queue()
    row: Final = _FIELDS.validate_python(client.spend_log_transactions[0])
    projected: Final = project(row)
    legacy: Final = {(item["entity_type"].value, item["entity_id"]): item["response_cost"] for item in queued}
    assert legacy[("key", proxy_server.hash_token("sk-projection"))] == next(
        dict(item.increments)["spend"] for item in projected if item.table == "LiteLLM_VerificationToken"
    )
    for table, entity, identity in (
        ("LiteLLM_UserTable", "user", "user"),
        ("LiteLLM_TeamTable", "team", "team"),
        ("LiteLLM_OrganizationTable", "organization", "org"),
        ("LiteLLM_EndUserTable", "end_user", "end-user"),
        ("LiteLLM_AgentsTable", "agent", "agent"),
        ("LiteLLM_ProjectTable", "project", "project"),
    ):
        assert legacy[(entity, identity)] == next(
            dict(item.increments)["spend"] for item in projected if item.table == table
        )
    assert sum(dict(item.increments)["spend"] for item in projected if item.table == "LiteLLM_TagTable") == sum(
        item["response_cost"] or 0 for item in queued if item["entity_type"].value == "tag"
    )

    daily_queues: Final = (
        ("LiteLLM_DailyUserSpend", writer.daily_spend_update_queue),
        ("LiteLLM_DailyTeamSpend", writer.daily_team_spend_update_queue),
        ("LiteLLM_DailyOrganizationSpend", writer.daily_org_spend_update_queue),
        ("LiteLLM_DailyEndUserSpend", writer.daily_end_user_spend_update_queue),
        ("LiteLLM_DailyAgentSpend", writer.daily_agent_spend_update_queue),
        ("LiteLLM_DailyTagSpend", writer.daily_tag_spend_update_queue),
    )
    for table, queue in daily_queues:
        transactions = await queue.flush_and_get_aggregated_daily_spend_update_transactions()
        matching = tuple(item for item in projected if item.table == table)
        assert matching, table
        contribution = matching[0]
        assert len(transactions) == 1
        transaction = next(iter(transactions.values()))
        assert dict(contribution.increments) == {name: transaction[name] for name, _ in contribution.increments}
    assert len(client.model_usage_transactions) == 1
    model: Final = client.model_usage_transactions[0]
    model_projection: Final = next(item for item in projected if item.table == "LiteLLM_DailyModelUsage")
    assert dict(model_projection.increments) == {
        "spend": model.spend,
        "prompt_tokens": model.prompt_tokens,
        "completion_tokens": model.completion_tokens,
        "request_count": int(model.counts_as_request),
        "successful_requests": int(model.successful),
        "failed_requests": int(not model.successful),
    }


@pytest.mark.parametrize("served_id", ["requested", "fallback", "missing"])
@pytest.mark.asyncio
async def test_projection_preserves_fallback_narrowed_model_access_groups(served_id: str) -> None:
    from litellm import Router
    from litellm.proxy.db.db_spend_update_writer import debitable_model_access_groups

    router: Final = Router(
        model_list=[
            {
                "model_name": "first",
                "litellm_params": {"model": "openai/projection-test", "api_key": "unused"},
                "model_info": {"id": "requested", "access_groups": ["first-group"]},
            },
            {
                "model_name": "second",
                "litellm_params": {"model": "openai/projection-test", "api_key": "unused"},
                "model_info": {"id": "fallback", "access_groups": ["second-group", "unattributed"]},
            },
        ]
    )
    attributed: Final = ("first-group", "second-group", "first-group")
    writer: Final = DBSpendUpdateWriter()
    client: Final = PrismaClient(database_url="postgresql://unused", proxy_logging_obj=proxy_logging_obj)
    await writer._update_model_access_group_db(0.5, attributed, served_id, client, router)
    queued: Final = await writer.spend_update_queue.flush_all_updates_from_in_memory_queue()
    frozen_groups: Final = debitable_model_access_groups(attributed, served_id, router)
    projected: Final = project(
        {"spend": 0.5, "metadata": json.dumps({"accounting_model_access_groups": frozen_groups})}
    )
    assert {dict(item.key)["access_group_name"]: dict(item.increments)["spend"] for item in projected} == {
        item["entity_id"]: item["response_cost"] for item in queued
    }
    assert "unattributed" not in frozen_groups


def test_guardrail_projection_skips_unmeasured_counters() -> None:
    row: Final = {
        "request_id": "guarded",
        "startTime": _NOW,
        "metadata": json.dumps(
            {
                "guardrail_information": [
                    {"guardrail_usage": {"text": 3}},
                    {
                        "guardrail_id": "guard",
                        "guardrail_usage": {"zero": 0, "negative": -1, "bool": True, "unknown": None, "text": 2},
                    },
                ]
            }
        ),
    }
    units: Final = tuple(item for item in project(row) if item.table == "LiteLLM_DailyGuardrailUsageUnits")
    assert len(units) == 1
    assert dict(units[0].key)["usage_unit"] == "text"
    assert dict(units[0].increments) == {"units": 2, "cost": 0.0, "untracked_units": 2}


def test_lean_row_without_model_does_not_fabricate_daily_entity_usage() -> None:
    projected: Final = project({"request_id": "lean", "api_key": "key", "startTime": _NOW, "spend": 0.5})
    assert tuple(item.table for item in projected) == ("LiteLLM_VerificationToken", "LiteLLM_DailyModelUsage")


@pytest.mark.parametrize(
    "cost_fields",
    [
        {"guardrail_cost_in_spend": False, "guardrail_cost_by_unit": {"text": 1.0}},
        {"guardrail_cost_by_unit": {"text": -1.0}},
    ],
)
def test_report_only_or_invalid_guardrail_prices_remain_untracked(cost_fields: Mapping[str, object]) -> None:
    row: Final = {
        "request_id": "guarded",
        "startTime": _NOW,
        "request_tags": '["task:debugging"]',
        "metadata": json.dumps(
            {
                "guardrail_information": [
                    {"guardrail_id": "guard", "guardrail_usage": {"text": 2}, **cost_fields},
                ]
            }
        ),
    }
    contributions: Final = project(row)
    guardrail: Final = next(item for item in contributions if item.table == "LiteLLM_DailyGuardrailUsageUnits")
    model: Final = next(item for item in contributions if item.table == "LiteLLM_DailyModelUsage")
    assert dict(guardrail.increments) == {"units": 2, "cost": 0.0, "untracked_units": 2}
    assert dict(model.key)["task_type"] == "debugging"
