"""
Unit tests for custom_prometheus_metadata_labels on the failure and fallback metrics.

The configured labels were only populated by async_log_success_event and
async_log_failure_event. Every other emitter declared the label but never filled
it, so the series exported the literal string "None" on exactly the metrics an
operator reaches for when calls start failing. Issue #38660.
"""

import pytest
from prometheus_client import REGISTRY

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import UserAPIKeyAuth


@pytest.fixture(scope="function")
def prometheus_logger(monkeypatch):
    """
    The configured labels are baked into each metric's declared label set when the
    logger is constructed, so the setting has to be in place before instantiation.
    """
    monkeypatch.setattr("litellm.custom_prometheus_metadata_labels", ["caller"])
    for collector in list(REGISTRY._collector_to_names.keys()):
        REGISTRY.unregister(collector)
    return PrometheusLogger()


def _caller_labels(metric_name: str) -> list[str | None]:
    """Every `caller` label value emitted on the named metric."""
    for metric in REGISTRY.collect():
        if metric.name == metric_name:
            return [sample.labels.get("caller") for sample in metric.samples]
    return []


def _standard_logging_object() -> dict:
    return {
        "model_id": "resolved-deployment-id",
        "model_group": "my-model-group",
        "api_base": "https://example.com",
        "metadata": {"spend_logs_metadata": {"caller": "my-task"}},
    }


class TestCustomLabelsOnDeploymentFailureMetrics:
    def test_deployment_failure_responses_carries_custom_label(self, prometheus_logger):
        prometheus_logger.set_llm_deployment_failure_metrics(
            {
                "model": "my-model",
                "exception": Exception("upstream 503"),
                "litellm_params": {"custom_llm_provider": "vertex_ai", "metadata": {}},
                "standard_logging_object": _standard_logging_object(),
            }
        )
        assert "my-task" in _caller_labels("litellm_deployment_failure_responses")

    def test_deployment_total_requests_carries_custom_label(self, prometheus_logger):
        prometheus_logger.set_llm_deployment_failure_metrics(
            {
                "model": "my-model",
                "exception": Exception("upstream 503"),
                "litellm_params": {"custom_llm_provider": "vertex_ai", "metadata": {}},
                "standard_logging_object": _standard_logging_object(),
            }
        )
        assert "my-task" in _caller_labels("litellm_deployment_total_requests")

    def test_unset_custom_label_is_absent_rather_than_wrong(self, prometheus_logger):
        """No spend_logs_metadata means no value to attribute; it must not invent one."""
        payload = _standard_logging_object()
        payload["metadata"] = {}
        prometheus_logger.set_llm_deployment_failure_metrics(
            {
                "model": "my-model",
                "exception": Exception("upstream 503"),
                "litellm_params": {"custom_llm_provider": "vertex_ai", "metadata": {}},
                "standard_logging_object": payload,
            }
        )
        assert "my-task" not in _caller_labels("litellm_deployment_failure_responses")


class TestCustomLabelsOnFallbackMetrics:
    @pytest.mark.asyncio
    async def test_successful_fallback_carries_custom_label(self, prometheus_logger):
        await prometheus_logger.log_success_fallback_event(
            original_model_group="my-model-group",
            kwargs={
                "model": "fallback-model",
                "metadata": {"spend_logs_metadata": {"caller": "my-task"}},
            },
            original_exception=Exception("upstream 503"),
        )
        assert "my-task" in _caller_labels("litellm_deployment_successful_fallbacks")

    @pytest.mark.asyncio
    async def test_failed_fallback_carries_custom_label(self, prometheus_logger):
        await prometheus_logger.log_failure_fallback_event(
            original_model_group="my-model-group",
            kwargs={
                "model": "fallback-model",
                "metadata": {"spend_logs_metadata": {"caller": "my-task"}},
            },
            original_exception=Exception("upstream 503"),
        )
        assert "my-task" in _caller_labels("litellm_deployment_failed_fallbacks")


class TestCustomLabelsOnProxyFailureMetrics:
    @pytest.mark.asyncio
    async def test_proxy_failed_requests_carries_custom_label(self, prometheus_logger):
        await prometheus_logger.async_post_call_failure_hook(
            request_data={
                "model": "my-model",
                "metadata": {"spend_logs_metadata": {"caller": "my-task"}},
            },
            original_exception=Exception("upstream 503"),
            user_api_key_dict=UserAPIKeyAuth(token="test_token"),
        )
        assert "my-task" in _caller_labels("litellm_proxy_failed_requests_metric")

    @pytest.mark.asyncio
    async def test_proxy_total_requests_carries_custom_label(self, prometheus_logger):
        await prometheus_logger.async_post_call_failure_hook(
            request_data={
                "model": "my-model",
                "metadata": {"spend_logs_metadata": {"caller": "my-task"}},
            },
            original_exception=Exception("upstream 503"),
            user_api_key_dict=UserAPIKeyAuth(token="test_token"),
        )
        assert "my-task" in _caller_labels("litellm_proxy_total_requests_metric")
