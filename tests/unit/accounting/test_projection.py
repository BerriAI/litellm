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
    frozen_groups: Final = debitable_model_access_groups(attributed, served_id, router)
    projected: Final = project(
        {"spend": 0.5, "metadata": json.dumps({"accounting_model_access_groups": frozen_groups})}
    )
    assert {dict(item.key)["access_group_name"]: dict(item.increments)["spend"] for item in projected} == {
        group: 0.5 for group in {"requested": ("first-group",), "fallback": ("second-group",), "missing": ("first-group", "second-group")}[served_id]
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


def test_legacy_takeover_applies_the_new_row_without_projecting_old_metadata():
    from litellm.accounting.delivery_projection import project_deliveries
    from litellm.repositories.accounting_delivery_repository import DeliveryProjection

    new = {"request_id": "batch", "api_key": "key", "spend": 0.25}
    old = {"request_id": "batch", "api_key": "old-key", "spend": 99, "metadata": {}}
    contributions = project_deliveries((DeliveryProjection(new, old, legacy_takeover=True),))
    assert contributions == project_deliveries((DeliveryProjection(new, None),))
    assert [(dict(item.key)["token"], dict(item.increments)["spend"]) for item in contributions if item.table == "LiteLLM_VerificationToken"] == [("key", 0.25)]
