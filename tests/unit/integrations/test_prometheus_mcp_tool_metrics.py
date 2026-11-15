"""
Unit tests for MCP tool call Prometheus metrics (LIT-3765).

These metrics expose ``mcp_tool_call_metadata`` in Prometheus so Grafana
dashboards can break down MCP usage by server and tool name.

Run with:
    uv run pytest tests/unit/integrations/test_prometheus_mcp_tool_metrics.py -v
"""

from typing import get_args
from unittest.mock import MagicMock

import pytest

from litellm.types.integrations.prometheus import (
    DEFINED_PROMETHEUS_METRICS,
    PrometheusMetricLabels,
    UserAPIKeyLabelNames,
    UserAPIKeyLabelValues,
)


MCP_METRICS = (
    "litellm_mcp_tool_calls_total",
    "litellm_mcp_tool_call_spend_metric",
)


def _make_mock_logger():
    logger = MagicMock()
    for name in MCP_METRICS:
        setattr(logger, name, MagicMock())
    logger.get_labels_for_metric = MagicMock(
        return_value=PrometheusMetricLabels.litellm_mcp_tool_calls_total,
    )
    return logger


def _make_enum_values(
    *,
    mcp_tool_name: str = "get_weather",
    mcp_server_name: str = "weather-server",
) -> UserAPIKeyLabelValues:
    return UserAPIKeyLabelValues(
        mcp_tool_name=mcp_tool_name,
        mcp_server_name=mcp_server_name,
        hashed_api_key="sk-hash-123",
        api_key_alias="test-key",
        team="team-1",
        team_alias="Test Team",
        user="user-1",
        end_user="end-user-1",
    )


def _make_payload(
    *,
    mcp_tool_name: str = "get_weather",
    mcp_server_name: str = "weather-server",
    response_cost: float = 0.005,
) -> dict:
    return {
        "model": "gpt-4o",
        "model_group": "gpt-4o",
        "model_id": "model-123",
        "api_base": "https://api.openai.com",
        "custom_llm_provider": "openai",
        "response_cost": response_cost,
        "completion_tokens": 50,
        "prompt_tokens": 100,
        "total_tokens": 150,
        "request_tags": [],
        "stream": False,
        "metadata": {
            "user_api_key_hash": "sk-hash-123",
            "user_api_key_alias": "test-key",
            "user_api_key_team_id": "team-1",
            "user_api_key_team_alias": "Test Team",
            "user_api_key_user_id": "user-1",
            "user_api_key_user_email": None,
            "user_api_key_org_id": None,
            "user_api_key_org_alias": None,
            "mcp_tool_call_metadata": {
                "name": mcp_tool_name,
                "mcp_server_name": mcp_server_name,
                "namespaced_tool_name": f"{mcp_server_name}/{mcp_tool_name}",
                "arguments": {"city": "SF"},
                "result": {"temp": 72},
            },
        },
    }


class TestMCPMetricRegistration:
    def test_metrics_in_defined_prometheus_metrics(self):
        defined = get_args(DEFINED_PROMETHEUS_METRICS)
        for name in MCP_METRICS:
            assert name in defined, f"{name} missing from DEFINED_PROMETHEUS_METRICS"

    def test_metric_labels_defined(self):
        for name in MCP_METRICS:
            assert hasattr(PrometheusMetricLabels, name), f"{name} missing from PrometheusMetricLabels"

    def test_mcp_labels_include_tool_and_server_name(self):
        labels = PrometheusMetricLabels.litellm_mcp_tool_calls_total
        assert UserAPIKeyLabelNames.MCP_TOOL_NAME.value in labels
        assert UserAPIKeyLabelNames.MCP_SERVER_NAME.value in labels

    def test_spend_metric_shares_label_set_with_calls_metric(self):
        assert (
            PrometheusMetricLabels.litellm_mcp_tool_call_spend_metric
            == PrometheusMetricLabels.litellm_mcp_tool_calls_total
        )
        assert (
            PrometheusMetricLabels.litellm_mcp_tool_call_spend_metric
            is not PrometheusMetricLabels.litellm_mcp_tool_calls_total
        )

    def test_enum_values_accept_mcp_fields(self):
        vals = _make_enum_values()
        assert vals.mcp_tool_name == "get_weather"
        assert vals.mcp_server_name == "weather-server"

    def test_enum_values_default_mcp_fields_to_none(self):
        vals = UserAPIKeyLabelValues(user="u1")
        assert vals.mcp_tool_name is None
        assert vals.mcp_server_name is None







if __name__ == "__main__":
    pytest.main([__file__, "-v"])
