import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm.accounting.projection import project
from litellm.accounting.snapshot import build_accounting_payload
from litellm.constants import BILLED_RETRY_ATTEMPT_METADATA_KEY, INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.proxy._types import SpendLogsPayload
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
@pytest.mark.parametrize(
    ("guardrail_information", "billed_guardrail_cost"),
    [
        (None, 0.0),
        ([{"guardrail_name": "guard", "guardrail_cost": 0.1}], 0.1),
        ([{"guardrail_name": "guard", "guardrail_cost": 0.5}], 0.5),
        ([{"guardrail_name": "guard", "guardrail_cost": 0.1, "guardrail_cost_in_spend": False}], 0.0),
    ],
)
@pytest.mark.asyncio
async def test_daily_projection_matches_legacy_writer(
    status: str,
    origin: str | None,
    spend: float,
    guardrail_information: list[dict[str, object]] | None,
    billed_guardrail_cost: float,
) -> None:
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
        **({"guardrail_information": guardrail_information} if guardrail_information is not None else {}),
    }
    row: Final = {**payload, "spend": spend, "metadata": json.dumps(metadata), "request_duration_ms": 25}
    legacy: Final = await DBSpendUpdateWriter()._common_add_spend_log_transaction_to_daily_transaction(
        row, PrismaClient(database_url="postgresql://unused", proxy_logging_obj=proxy_logging_obj), "user"
    )
    assert legacy is not None
    accounting_input: Final[SpendLogsPayload] = {**row}
    accounting: Final = build_accounting_payload(
        accounting_input,
        entities={"api_key": payload["api_key"], "user": payload["user"]},
        model_access_groups=(),
        windows=(),
        router=lambda: None,
    )
    restored: Final = _FIELDS.validate_json(json.dumps(accounting, default=str))
    stored_daily: Final = next(item for item in project(restored) if item.table == "LiteLLM_DailyUserSpend")
    assert dict(stored_daily.increments) == {name: legacy[name] for name, _ in stored_daily.increments}
    assert dict(stored_daily.increments)["guardrail_spend"] == min(billed_guardrail_cost, spend)
    assert dict(stored_daily.key)["date"] == legacy["date"]
    assert dict(stored_daily.key)["api_key"] == legacy["api_key"]
    assert not {"messages", "response", "proxy_server_request", "requester_ip_address"}.intersection(accounting)


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
    monkeypatch.setitem(proxy_server.general_settings, "use_spend_logs_ledger", False)
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
    logged: Final[SpendLogsPayload] = client.spend_log_transactions[0]
    projected: Final = project(
        _FIELDS.validate_json(
            json.dumps(
                build_accounting_payload(
                    logged,
                    entities={
                        "api_key": logged["api_key"],
                        "user": "user",
                        "end_user": "end-user",
                        "team_id": "team",
                        "organization_id": "org",
                        "project_id": "project",
                        "agent_id": "agent",
                    },
                    model_access_groups=(),
                    windows=(),
                    router=lambda: None,
                ),
                default=str,
            )
        )
    )
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
    projected: Final = project(
        {
            "request_id": "lean",
            "api_key": "key",
            "startTime": _NOW,
            "spend": 0.5,
            "metadata": {"accounting_entities": {"api_key": "key"}},
        }
    )
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


@pytest.mark.parametrize("organization_id", [None, "", "charged-org"])
def test_daily_organization_uses_frozen_charged_identity(organization_id: str | None) -> None:
    row: Final[Mapping[str, object]] = {
        "startTime": _NOW,
        "api_key": "key",
        "model": "projection-test",
        "model_group": "group",
        "custom_llm_provider": "openai",
        "organization_id": "stored-org",
        "spend": 0.25,
        "metadata": {"accounting_entities": {"organization_id": organization_id}},
    }
    daily: Final = tuple(item for item in project(row) if item.table == "LiteLLM_DailyOrganizationSpend")
    if organization_id is None:
        assert daily == ()
    else:
        assert len(daily) == 1
        assert dict(daily[0].key)["organization_id"] == organization_id
        assert dict(daily[0].increments)["spend"] == 0.25


@pytest.mark.parametrize("start", [_NOW, "2026-11-02T12:00:00+00:00", "2026-11-02 12:00:00+00:00"])
def test_every_daily_row_uses_the_calendar_date_of_the_request(start: object) -> None:
    row: Final[Mapping[str, object]] = {
        "request_id": "dated",
        "startTime": start,
        "api_key": "key",
        "user": "user",
        "team_id": "team",
        "model": "unpriced-projection-test",
        "model_group": "group",
        "custom_llm_provider": "openai",
        "spend": 0.5,
        "metadata": "{}",
    }
    dates: Final = {
        item.table: dict(item.key)["date"] for item in project(row) if item.table.startswith("LiteLLM_Daily")
    }
    assert {"LiteLLM_DailyUserSpend", "LiteLLM_DailyModelUsage"} <= dates.keys()
    assert set(dates.values()) == {"2026-11-02"}


def test_unparseable_start_time_skips_only_guardrail_usage() -> None:
    projected: Final = project(
        {
            "request_id": "bad-time",
            "startTime": "yesterday at noon",
            "spend": 0.5,
            "metadata": {
                "accounting_entities": {"api_key": "key"},
                "guardrail_information": [{"guardrail_id": "guard", "guardrail_usage": {"text": 2}}],
            },
        }
    )
    tables: Final = {item.table for item in projected}
    assert "LiteLLM_VerificationToken" in tables
    assert "LiteLLM_DailyGuardrailUsageUnits" not in tables


def test_guardrail_usage_is_dated_in_utc() -> None:
    projected: Final = project(
        {
            "request_id": "offset",
            "startTime": "2026-11-02T22:30:00-05:00",
            "metadata": {"guardrail_information": [{"guardrail_id": "guard", "guardrail_usage": {"text": 1}}]},
        }
    )
    assert [dict(item.key)["date"] for item in projected if item.table == "LiteLLM_DailyGuardrailUsageUnits"] == [
        "2026-11-03"
    ]


@pytest.mark.parametrize(
    ("untrusted", "metadata"),
    [
        ({"request_tags": json.dumps(["x" * 16384, "normal"])}, {}),
        ({"end_user": "x" * 3072}, {}),
        ({}, {"compression_savings": {"tokens_saved": 2**63}}),
    ],
)
def test_untrusted_analytics_values_never_block_the_charge(
    untrusted: Mapping[str, str], metadata: Mapping[str, object]
) -> None:
    row: Final[Mapping[str, object]] = {
        "request_id": "untrusted",
        "startTime": _NOW,
        "api_key": "key",
        "user": "user",
        "team_id": "team",
        "model": "unpriced-projection-test",
        "model_group": "group",
        "custom_llm_provider": "openai",
        "spend": 0.5,
        **untrusted,
        "metadata": {
            **metadata,
            "accounting_entities": {
                "api_key": "key",
                "user": "user",
                "team_id": "team",
                "end_user": untrusted.get("end_user"),
            },
        },
    }
    contributions: Final = project(row)
    texts: Final = tuple(value for item in contributions for _, value in (*item.key, *item.values) if value)
    numbers: Final = tuple(value for item in contributions for _, value in item.increments)
    assert {"LiteLLM_VerificationToken", "LiteLLM_TeamTable", "LiteLLM_UserTable"} <= {
        item.table for item in contributions
    }
    assert "LiteLLM_DailyUserSpend" in {item.table for item in contributions}
    assert all(len(text.encode()) <= 2000 for text in texts)
    assert all(abs(number) <= 2**40 for number in numbers)


def test_stored_payload_keeps_guardrail_billing_but_not_guardrail_content() -> None:
    payload: Final = get_logging_payload(
        kwargs={
            "model": "projection-test",
            "custom_llm_provider": "openai",
            "call_type": "acompletion",
            "litellm_call_id": "guardrail-request",
            "litellm_params": {"metadata": {"user_api_key": "key", "user_api_key_user_id": "user"}},
        },
        response_obj=ModelResponse(
            id="guardrail-request", model="projection-test", usage=Usage(prompt_tokens=10, completion_tokens=5)
        ),
        start_time=_NOW,
        end_time=_NOW,
    )
    guardrail: Final = {
        "guardrail_name": "pii",
        "guardrail_usage": {"requests": 1},
        "guardrail_cost_by_unit": {"requests": 0.25},
        "guardrail_response": "card 4111 1111 1111 1111",
        "match_details": [{"matched": "4111 1111 1111 1111"}],
    }
    headroom: Final = {
        "guardrail_name": "compress",
        "guardrail_provider": "headroom",
        "guardrail_response": {"tokens_saved": 7, "compressed_prompt": "card 4111 1111 1111 1111"},
    }
    metadata: Final = {**_FIELDS.validate_json(payload["metadata"]), "guardrail_information": [guardrail, headroom]}
    accounting_input: Final[SpendLogsPayload] = {
        **payload,
        "spend": 0.5,
        "metadata": json.dumps(metadata),
        "request_tags": '["nul\x00tag"]',
    }
    accounting: Final = build_accounting_payload(
        accounting_input, entities={"api_key": "key"}, model_access_groups=(), windows=(), router=lambda: None
    )
    stored: Final = json.dumps(accounting, default=str)
    restored: Final = _FIELDS.validate_json(stored)
    usage_rows: Final = tuple(item for item in project(restored) if item.table == "LiteLLM_DailyGuardrailUsageUnits")
    assert "4111" not in stored
    assert "\\u0000" not in stored
    assert [dict(item.increments) for item in usage_rows] == [{"units": 1, "cost": 0.25, "untracked_units": 0}]
    daily_user: Final = next(item for item in project(restored) if item.table == "LiteLLM_DailyUserSpend")
    assert dict(daily_user.increments)["compression_saved_tokens"] == 7


def test_caller_supplied_savings_beyond_json_range_are_clamped_and_never_drop_the_charge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm

    monkeypatch.setitem(
        litellm.model_cost,
        "savings-overflow-test",
        {"input_cost_per_token": 0.002, "output_cost_per_token": 0.004, "litellm_provider": "openai", "mode": "chat"},
    )
    payload: Final[SpendLogsPayload] = {
        **get_logging_payload(
            kwargs={
                "model": "savings-overflow-test",
                "custom_llm_provider": "openai",
                "call_type": "acompletion",
                "litellm_call_id": "overflow-request",
                "litellm_params": {"metadata": {"user_api_key": "key"}},
            },
            response_obj=ModelResponse(
                id="overflow-request", model="savings-overflow-test", usage=Usage(prompt_tokens=10, completion_tokens=5)
            ),
            start_time=_NOW,
            end_time=_NOW,
        ),
        "spend": 0.5,
        "metadata": json.dumps({"compression_savings": {"tokens_saved": 10**400}}),
    }
    accounting: Final = build_accounting_payload(
        payload, entities={"api_key": "key"}, model_access_groups=(), windows=(), router=lambda: None
    )
    restored: Final = _FIELDS.validate_json(json.dumps(accounting, default=str))
    assert dict(restored["metadata"])["accounting_savings"]["compression"] == 0.0
    assert dict(restored["metadata"])["compression_savings"] == {"tokens_saved": 2**53 - 1}
    assert dict(restored["metadata"])["compression_savings"] == {"tokens_saved": 2**53 - 1}
    assert [
        dict(item.increments)["spend"] for item in project(restored) if item.table == "LiteLLM_VerificationToken"
    ] == [0.5]
