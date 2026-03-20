import os
import time
from unittest.mock import AsyncMock

import pytest
from httpx import Response

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
    logger.async_client.put.return_value = Response(202, json={"status": "ok"})

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

    import json

    # Use call_args.kwargs['content']
    content = json.loads(call_args[1]["content"])
    assert content[0]["ProviderName"] == "openai"
    assert content[0]["BilledCost"] == 0.01
