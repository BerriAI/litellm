import json
import os
import time
from unittest.mock import AsyncMock

import pytest
from httpx import Request, Response

from litellm.integrations.datadog.datadog_cost_management import (
    DatadogCostManagementLogger,
)
from litellm.types.utils import StandardLoggingPayload


@pytest.fixture
def clean_env():
    # Save original env
    original_api_key = os.environ.get("DD_API_KEY")
    original_app_key = os.environ.get("DD_APP_KEY")
    original_site = os.environ.get("DD_SITE")

    # Set test env
    os.environ["DD_API_KEY"] = "test_api_key"
    os.environ["DD_APP_KEY"] = "test_app_key"
    os.environ["DD_SITE"] = "test.datadoghq.com"

    yield

    # Restore original env
    if original_api_key:
        os.environ["DD_API_KEY"] = original_api_key
    else:
        del os.environ["DD_API_KEY"]

    if original_app_key:
        os.environ["DD_APP_KEY"] = original_app_key
    else:
        del os.environ["DD_APP_KEY"]

    if original_site:
        os.environ["DD_SITE"] = original_site
    else:
        del os.environ["DD_SITE"]


@pytest.mark.asyncio
async def test_init(clean_env):
    """
    Test initialization sets up clients and url correctly
    """
    logger = DatadogCostManagementLogger()
    assert logger.dd_api_key == "test_api_key"
    assert logger.dd_app_key == "test_app_key"
    assert (
        logger.upload_url == "https://api.test.datadoghq.com/api/v2/cost/custom_costs"
    )


@pytest.mark.asyncio
async def test_aggregate_costs(clean_env):
    """
    Test that costs are correctly aggregated by provider, model, and date
    """
    logger = DatadogCostManagementLogger()

    # Mock some log payloads
    now = time.time()
    day_str = time.strftime("%Y-%m-%d", time.localtime(now))

    logs = [
        StandardLoggingPayload(
            custom_llm_provider="openai",
            model="gpt-4",
            response_cost=0.01,
            startTime=now,
            metadata={"user_api_key_team_alias": "team-a"},
        ),
        StandardLoggingPayload(
            custom_llm_provider="openai",
            model="gpt-4",
            response_cost=0.02,
            startTime=now,
            metadata={"user_api_key_team_alias": "team-a"},
        ),
        StandardLoggingPayload(
            custom_llm_provider="anthropic",
            model="claude-3",
            response_cost=0.05,
            startTime=now,
        ),
    ]

    aggregated = logger._aggregate_costs(logs)

    assert len(aggregated) == 2

    # Check OpenAI entry
    openai_entry = next(e for e in aggregated if e["ProviderName"] == "openai")
    assert openai_entry["BilledCost"] == 0.03
    assert openai_entry["ChargeDescription"] == "LLM Usage for gpt-4"
    assert openai_entry["ChargePeriodStart"] == day_str
    assert openai_entry["Tags"]["team"] == "team-a"
    assert "env" in openai_entry["Tags"]
    assert "service" in openai_entry["Tags"]

    # Check Anthropic entry
    anthropic_entry = next(e for e in aggregated if e["ProviderName"] == "anthropic")
    assert anthropic_entry["BilledCost"] == 0.05


@pytest.mark.asyncio
async def test_async_log_success_event(clean_env):
    """
    Test that logs are added to queue
    """
    logger = DatadogCostManagementLogger(batch_size=10)

    await logger.async_log_success_event(
        kwargs={"standard_logging_object": {"response_cost": 0.01}},
        response_obj={},
        start_time=time.time(),
        end_time=time.time(),
    )

    assert len(logger.log_queue) == 1
    assert logger.log_queue[0]["response_cost"] == 0.01

    # Test zero cost ignored
    await logger.async_log_success_event(
        kwargs={"standard_logging_object": {"response_cost": 0.0}},
        response_obj={},
        start_time=time.time(),
        end_time=time.time(),
    )

    assert len(logger.log_queue) == 1


@pytest.mark.asyncio
async def test_async_send_batch(clean_env):
    """
    Test that batch is aggregated and uploaded
    """
    logger = DatadogCostManagementLogger()
    logger.async_client = AsyncMock()
    logger.async_client.put.return_value = Response(
        202,
        request=Request(
            "PUT", "https://api.test.datadoghq.com/api/v2/cost/custom_costs"
        ),
        json={"status": "ok"},
    )

    # Add logs directly to queue
    logger.log_queue = [
        StandardLoggingPayload(
            custom_llm_provider="openai",
            model="gpt-4",
            response_cost=0.01,
            startTime=time.time(),
        )
    ]

    await logger.async_send_batch()

    # Verify API called
    assert logger.async_client.put.called
    call_args = logger.async_client.put.call_args
    assert call_args[0][0] == "https://api.test.datadoghq.com/api/v2/cost/custom_costs"

    # Use call_args.kwargs['content']
    content = json.loads(call_args[1]["content"])
    assert content[0]["ProviderName"] == "openai"
    assert content[0]["BilledCost"] == 0.01
    assert logger.log_queue == []


@pytest.mark.asyncio
async def test_async_send_batch_preserves_events_added_during_upload(clean_env):
    logger = DatadogCostManagementLogger()
    logger.log_queue = [
        StandardLoggingPayload(
            custom_llm_provider="openai",
            model="gpt-4",
            response_cost=0.01,
            startTime=time.time(),
        )
    ]

    async def _mock_upload(payload):
        logger.log_queue.append(
            StandardLoggingPayload(
                custom_llm_provider="anthropic",
                model="claude-3",
                response_cost=0.02,
                startTime=time.time(),
            )
        )

    logger._upload_to_datadog = AsyncMock(side_effect=_mock_upload)

    await logger.async_send_batch()

    logger._upload_to_datadog.assert_awaited_once()
    assert len(logger.log_queue) == 1
    assert logger.log_queue[0]["model"] == "claude-3"


@pytest.mark.asyncio
async def test_async_send_batch_requeues_batch_on_upload_error(clean_env):
    logger = DatadogCostManagementLogger()
    logger.log_queue = [
        StandardLoggingPayload(
            custom_llm_provider="openai",
            model="gpt-4",
            response_cost=0.01,
            startTime=time.time(),
        )
    ]
    logger._upload_to_datadog = AsyncMock(side_effect=RuntimeError("boom"))

    await logger.async_send_batch()

    assert len(logger.log_queue) == 1
    assert logger.log_queue[0]["model"] == "gpt-4"


@pytest.mark.asyncio
async def test_extract_tags_includes_model_request_and_metadata_finops_tags(clean_env):
    logger = DatadogCostManagementLogger()
    payload = StandardLoggingPayload(
        custom_llm_provider="openai",
        model="gpt-4",
        model_group="customer-facing",
        model_id="model-123",
        response_cost=0.01,
        startTime=time.time(),
        request_tags=["ai_product:chat", "feature:summarize", "purpose:support"],
        metadata={
            "user_api_key_team_alias": "team-a",
            "environment": "prod",
            "spend_logs_metadata": {"cost_center": "ml-platform"},
            "requester_metadata": {"region": "us-east-1"},
        },
    )

    tags = logger._extract_tags(payload)

    assert tags["provider"] == "openai"
    assert tags["model"] == "gpt-4"
    assert tags["model_group"] == "customer-facing"
    assert tags["model_id"] == "model-123"
    assert tags["team"] == "team-a"
    assert tags["ai_product"] == "chat"
    assert tags["feature"] == "summarize"
    assert tags["purpose"] == "support"
    assert tags["environment"] == "prod"
    assert tags["cost_center"] == "ml-platform"
    assert tags["region"] == "us-east-1"
