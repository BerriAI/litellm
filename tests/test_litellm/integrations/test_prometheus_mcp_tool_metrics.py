"""
Unit tests for MCP tool call Prometheus metrics (LIT-3765).

These metrics expose ``mcp_tool_call_metadata`` in Prometheus so Grafana
dashboards can break down MCP usage by server and tool name.

Run with:
    uv run pytest tests/test_litellm/integrations/test_prometheus_mcp_tool_metrics.py -v
"""

from typing import get_args
from unittest.mock import MagicMock

import pytest

from litellm.integrations.prometheus import PrometheusLogger
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


class TestIncrementMCPToolCallMetrics:
    def test_increments_calls_counter_when_mcp_metadata_present(self):
        logger = _make_mock_logger()
        payload = _make_payload()
        enum_values = _make_enum_values()

        PrometheusLogger._increment_mcp_tool_call_metrics(
            logger,
            standard_logging_payload=payload,
            enum_values=enum_values,
            response_cost=0.005,
        )

        logger.litellm_mcp_tool_calls_total.labels.assert_called_once()
        logger.litellm_mcp_tool_calls_total.labels().inc.assert_called_once_with(1.0)

    def test_increments_spend_counter_when_cost_positive(self):
        logger = _make_mock_logger()
        payload = _make_payload(response_cost=0.01)
        enum_values = _make_enum_values()

        PrometheusLogger._increment_mcp_tool_call_metrics(
            logger,
            standard_logging_payload=payload,
            enum_values=enum_values,
            response_cost=0.01,
        )

        logger.litellm_mcp_tool_call_spend_metric.labels.assert_called_once()
        logger.litellm_mcp_tool_call_spend_metric.labels().inc.assert_called_once_with(0.01)

    def test_skips_spend_counter_when_cost_zero(self):
        logger = _make_mock_logger()
        payload = _make_payload(response_cost=0.0)
        enum_values = _make_enum_values()

        PrometheusLogger._increment_mcp_tool_call_metrics(
            logger,
            standard_logging_payload=payload,
            enum_values=enum_values,
            response_cost=0.0,
        )

        logger.litellm_mcp_tool_calls_total.labels.assert_called_once()
        logger.litellm_mcp_tool_call_spend_metric.labels.assert_not_called()

    def test_noop_when_no_mcp_metadata(self):
        logger = _make_mock_logger()
        payload = _make_payload()
        payload["metadata"]["mcp_tool_call_metadata"] = None
        enum_values = _make_enum_values()

        PrometheusLogger._increment_mcp_tool_call_metrics(
            logger,
            standard_logging_payload=payload,
            enum_values=enum_values,
            response_cost=0.005,
        )

        for name in MCP_METRICS:
            getattr(logger, name).labels.assert_not_called()

    def test_noop_when_metadata_missing(self):
        logger = _make_mock_logger()
        payload = {"metadata": None}
        enum_values = _make_enum_values()

        PrometheusLogger._increment_mcp_tool_call_metrics(
            logger,
            standard_logging_payload=payload,
            enum_values=enum_values,
            response_cost=0.005,
        )

        for name in MCP_METRICS:
            getattr(logger, name).labels.assert_not_called()

    def test_label_values_carry_tool_and_server_name(self):
        logger = _make_mock_logger()
        payload = _make_payload(
            mcp_tool_name="search_docs",
            mcp_server_name="docs-mcp",
        )
        enum_values = _make_enum_values()

        PrometheusLogger._increment_mcp_tool_call_metrics(
            logger,
            standard_logging_payload=payload,
            enum_values=enum_values,
            response_cost=0.005,
        )

        labels_passed = logger.litellm_mcp_tool_calls_total.labels.call_args
        assert labels_passed.kwargs["mcp_tool_name"] == "search_docs"
        assert labels_passed.kwargs["mcp_server_name"] == "docs-mcp"

    def test_label_values_carry_team_and_key_from_parent(self):
        logger = _make_mock_logger()
        payload = _make_payload()
        enum_values = UserAPIKeyLabelValues(
            hashed_api_key="sk-parent-key",
            api_key_alias="parent-alias",
            team="parent-team",
            team_alias="Parent Team",
            user="parent-user",
            end_user="parent-end-user",
        )

        PrometheusLogger._increment_mcp_tool_call_metrics(
            logger,
            standard_logging_payload=payload,
            enum_values=enum_values,
            response_cost=0.005,
        )

        labels_passed = logger.litellm_mcp_tool_calls_total.labels.call_args
        assert labels_passed.kwargs["hashed_api_key"] == "sk-parent-key"
        assert labels_passed.kwargs["team"] == "parent-team"
        assert labels_passed.kwargs["team_alias"] == "Parent Team"
        assert labels_passed.kwargs["user"] == "parent-user"

    def test_handles_missing_server_name_gracefully(self):
        logger = _make_mock_logger()
        payload = _make_payload()
        payload["metadata"]["mcp_tool_call_metadata"] = {
            "name": "standalone_tool",
            "arguments": {},
            "result": {},
        }
        enum_values = _make_enum_values()

        PrometheusLogger._increment_mcp_tool_call_metrics(
            logger,
            standard_logging_payload=payload,
            enum_values=enum_values,
            response_cost=0.0,
        )

        labels_passed = logger.litellm_mcp_tool_calls_total.labels.call_args
        assert labels_passed.kwargs["mcp_tool_name"] == "standalone_tool"
        assert labels_passed.kwargs["mcp_server_name"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
