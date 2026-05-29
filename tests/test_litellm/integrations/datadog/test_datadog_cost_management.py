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


_PUT_REQUEST = Request("PUT", "https://api.test.datadoghq.com/api/v2/cost/custom_costs")


@pytest.mark.asyncio
async def test_async_send_batch_clears_queue_on_success(clean_env):
    """Bug 1 regression: log_queue must be empty after a successful upload."""
    logger = DatadogCostManagementLogger()
    logger.async_client = AsyncMock()
    logger.async_client.put.return_value = Response(
        202, json={"status": "ok"}, request=_PUT_REQUEST
    )
    logger.log_queue = [
        StandardLoggingPayload(
            custom_llm_provider="openai",
            model="gpt-4",
            response_cost=0.01,
            startTime=time.time(),
        )
    ]
    await logger.async_send_batch()
    assert logger.log_queue == []


@pytest.mark.asyncio
async def test_async_send_batch_preserves_events_added_during_upload(clean_env):
    """Events appended while the upload is in flight survive (land on the cleared queue)."""
    logger = DatadogCostManagementLogger()

    later_event = StandardLoggingPayload(
        custom_llm_provider="anthropic",
        model="claude-3",
        response_cost=0.02,
        startTime=time.time(),
    )

    async def slow_put(*args, **kwargs):
        logger.log_queue.append(later_event)
        return Response(202, json={"status": "ok"}, request=_PUT_REQUEST)

    logger.async_client = AsyncMock()
    logger.async_client.put.side_effect = slow_put
    logger.log_queue = [
        StandardLoggingPayload(
            custom_llm_provider="openai",
            model="gpt-4",
            response_cost=0.01,
            startTime=time.time(),
        )
    ]
    await logger.async_send_batch()
    assert logger.log_queue == [later_event]


@pytest.mark.asyncio
async def test_async_send_batch_requeues_on_upload_failure(clean_env):
    """Failed upload requeues the original batch (no data loss)."""
    logger = DatadogCostManagementLogger()
    logger.async_client = AsyncMock()
    logger.async_client.put.side_effect = Exception("boom")
    original = StandardLoggingPayload(
        custom_llm_provider="openai",
        model="gpt-4",
        response_cost=0.01,
        startTime=time.time(),
    )
    logger.log_queue = [original]
    await logger.async_send_batch()
    assert logger.log_queue == [original]


@pytest.mark.asyncio
async def test_extract_tags_emits_canonical_focus_dimensions(clean_env):
    """provider, model, model_id always emitted regardless of cost_tag_keys."""
    logger = DatadogCostManagementLogger()
    log = StandardLoggingPayload(
        custom_llm_provider="openai",
        model="gpt-4o",
        model_id="router-id-123",
        response_cost=0.01,
        startTime=time.time(),
    )
    tags = logger._extract_tags(log)
    assert tags["provider"] == "openai"
    assert tags["model"] == "gpt-4o"
    assert tags["model_id"] == "router-id-123"


@pytest.mark.asyncio
async def test_extract_tags_allowlist_filters_request_tags(clean_env):
    """Only request_tags whose key is in cost_tag_keys reach the Tags dict."""
    logger = DatadogCostManagementLogger(cost_tag_keys=["capability", "tier"])
    log = StandardLoggingPayload(
        custom_llm_provider="openai",
        model="gpt-4",
        response_cost=0.01,
        startTime=time.time(),
        request_tags=["capability:chat", "tier:gold", "secret:disallowed"],
    )
    tags = logger._extract_tags(log)
    assert tags["capability"] == "chat"
    assert tags["tier"] == "gold"
    assert "secret" not in tags


@pytest.mark.asyncio
async def test_extract_tags_allowlist_filters_metadata(clean_env):
    """Only metadata keys in cost_tag_keys flow through; others (and dict/list values) are dropped."""
    logger = DatadogCostManagementLogger(cost_tag_keys=["capability", "owner"])
    log = StandardLoggingPayload(
        custom_llm_provider="openai",
        model="gpt-4",
        response_cost=0.01,
        startTime=time.time(),
        metadata={
            "capability": "chat",
            "owner": "team-x",
            "secret_field": "sensitive",
            "nested_obj": {"a": 1},
        },
    )
    tags = logger._extract_tags(log)
    assert tags["capability"] == "chat"
    assert tags["owner"] == "team-x"
    assert "secret_field" not in tags
    assert "nested_obj" not in tags


@pytest.mark.asyncio
async def test_extract_tags_empty_allowlist_default(clean_env):
    """With no cost_tag_keys, request_tags and arbitrary metadata.* do NOT leak into Tags."""
    logger = DatadogCostManagementLogger()
    log = StandardLoggingPayload(
        custom_llm_provider="openai",
        model="gpt-4",
        response_cost=0.01,
        startTime=time.time(),
        request_tags=["capability:chat"],
        metadata={"capability": "chat", "user_api_key_alias": "alice"},
    )
    tags = logger._extract_tags(log)
    assert "capability" not in tags
    # Backwards-compat keys still flow:
    assert tags["user"] == "alice"


@pytest.mark.asyncio
async def test_extract_tags_nested_metadata_allowlisted(clean_env):
    """spend_logs_metadata and requester_metadata get spread one level under the allowlist."""
    logger = DatadogCostManagementLogger(cost_tag_keys=["env", "platform"])
    log = StandardLoggingPayload(
        custom_llm_provider="openai",
        model="gpt-4",
        response_cost=0.01,
        startTime=time.time(),
        metadata={
            "spend_logs_metadata": {"platform": "web", "ignored": "x"},
            "requester_metadata": {"env": "prod"},
        },
    )
    tags = logger._extract_tags(log)
    assert tags["platform"] == "web"
    # "env" is a reserved trusted dimension — requester_metadata.env must NOT
    # overwrite the value sourced from get_datadog_env().
    assert tags["env"] != "prod"
    assert "ignored" not in tags


@pytest.mark.asyncio
async def test_extract_tags_allowlist_cannot_override_reserved_dimensions(clean_env):
    """
    Reserved tag keys (env, service, host, pod_name, provider, model, model_id,
    team, user, model_group) must not be overwritten by user-controlled
    request_tags or metadata, even when listed in cost_tag_keys.
    """
    reserved = [
        "env",
        "service",
        "host",
        "pod_name",
        "provider",
        "model",
        "model_id",
        "team",
        "user",
        "model_group",
    ]
    logger = DatadogCostManagementLogger(cost_tag_keys=reserved)

    metadata_attack = {k: f"attacker-meta-{k}" for k in reserved}
    metadata_attack["user_api_key_alias"] = "trusted-user"
    metadata_attack["user_api_key_team_alias"] = "trusted-team"
    metadata_attack["model_group"] = "trusted-group"
    metadata_attack["spend_logs_metadata"] = {
        k: f"attacker-spend-{k}" for k in reserved
    }
    metadata_attack["requester_metadata"] = {k: f"attacker-req-{k}" for k in reserved}

    log = StandardLoggingPayload(
        custom_llm_provider="openai",
        model="gpt-4",
        model_id="router-id-123",
        response_cost=0.01,
        startTime=time.time(),
        request_tags=[f"{k}:attacker-rt-{k}" for k in reserved],
        metadata=metadata_attack,
    )

    tags = logger._extract_tags(log)

    # Canonical FOCUS dims keep their trusted (top-level payload) values.
    assert tags["provider"] == "openai"
    assert tags["model"] == "gpt-4"
    assert tags["model_id"] == "router-id-123"

    # Backwards-compat trusted dims keep their proxy-controlled metadata values.
    assert tags["user"] == "trusted-user"
    assert tags["team"] == "trusted-team"
    assert tags["model_group"] == "trusted-group"

    # No reserved key carries an attacker-supplied prefix from any path.
    for k in reserved:
        assert not tags[k].startswith("attacker-"), (
            f"reserved key {k!r} was overwritten by user-controlled input: "
            f"{tags[k]!r}"
        )
