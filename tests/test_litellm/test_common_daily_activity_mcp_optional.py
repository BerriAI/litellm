import sys
import os

sys.path.insert(0, os.path.abspath("../../"))

from litellm.types.proxy.management_endpoints.common_daily_activity import (
    BreakdownMetrics,
    SpendMetrics,
)
from litellm.proxy.management_endpoints.common_daily_activity import (
    update_breakdown_metrics,
)


class _RecordWithoutMCP:
    """Minimal record shape required by update_breakdown_metrics.

    Intentionally does NOT provide mcp_namespaced_tool_name to ensure the function
    handles missing field gracefully.
    """

    def __init__(self) -> None:
        self.model = "gpt-4o"
        self.model_group = "gpt-4o"
        self.api_key = "vk_test"
        self.custom_llm_provider = "openai"
        self.spend = 1.23
        self.prompt_tokens = 100
        self.completion_tokens = 50
        self.cache_read_input_tokens = 10
        self.cache_creation_input_tokens = 5
        self.api_requests = 3
        self.successful_requests = 3
        self.failed_requests = 0


def test_update_breakdown_metrics_with_missing_mcp_field():
    record = _RecordWithoutMCP()
    breakdown = BreakdownMetrics()

    model_metadata = {}
    provider_metadata = {}
    api_key_metadata = {}

    # Should not raise even though the record lacks `mcp_namespaced_tool_name`
    updated = update_breakdown_metrics(
        breakdown=breakdown,
        record=record,
        model_metadata=model_metadata,
        provider_metadata=provider_metadata,
        api_key_metadata=api_key_metadata,
    )

    # Asserts: model and provider metrics got updated
    assert "gpt-4o" in updated.models
    assert "openai" in updated.providers
    # MCP breakdown remains empty without the field
    assert updated.mcp_servers == {}


