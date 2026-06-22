"""
Tests for the opt-in `model_group` label on deployment-level Prometheus metrics
(issue #30748).

The label is gated behind `litellm.prometheus_emit_deployment_model_group_label`
(default off) so the historical label set of each metric is preserved across
upgrade, mirroring `prometheus_emit_rate_limit_labels`. These tests assert both
the label-list wiring and that the label actually flows onto the emitted series
when the flag is enabled.
"""

from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.router_utils.cooldown_callbacks import router_cooldown_event_callback
from litellm.types.integrations.prometheus import (
    PrometheusMetricLabels,
    UserAPIKeyLabelNames,
)
from litellm.types.router import ModelInfo

DEPLOYMENT_METRICS = [
    "litellm_deployment_state",
    "litellm_deployment_tpm_limit",
    "litellm_deployment_rpm_limit",
    "litellm_deployment_cooled_down",
    "litellm_deployment_latency_per_output_token",
]


def _logger_without_init() -> PrometheusLogger:
    with patch(
        "litellm.integrations.prometheus.PrometheusLogger.__init__", return_value=None
    ):
        return PrometheusLogger()


# ---------------------------------------------------------------------------
# Label-list wiring (flag on vs default off)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric_name", DEPLOYMENT_METRICS)
def test_model_group_included_when_flag_enabled(monkeypatch, metric_name):
    monkeypatch.setattr(litellm, "prometheus_emit_deployment_model_group_label", True)
    labels = PrometheusMetricLabels.get_labels(metric_name)
    assert UserAPIKeyLabelNames.MODEL_GROUP.value in labels
    # model_id must remain so a group can still be drilled down to a deployment.
    assert UserAPIKeyLabelNames.MODEL_ID.value in labels


@pytest.mark.parametrize("metric_name", DEPLOYMENT_METRICS)
def test_model_group_omitted_by_default_for_back_compat(metric_name):
    """Default-off preserves each metric's historical label set so existing
    dashboards / recording rules keep matching after upgrade."""
    assert litellm.prometheus_emit_deployment_model_group_label is False
    labels = PrometheusMetricLabels.get_labels(metric_name)
    assert UserAPIKeyLabelNames.MODEL_GROUP.value not in labels
    assert UserAPIKeyLabelNames.MODEL_ID.value in labels


# ---------------------------------------------------------------------------
# The label flows onto emitted series when enabled
# ---------------------------------------------------------------------------


def test_increment_deployment_cooled_down_emits_model_group(monkeypatch):
    monkeypatch.setattr(litellm, "prometheus_emit_deployment_model_group_label", True)
    logger = _logger_without_init()
    logger.litellm_deployment_cooled_down = MagicMock()
    logger.get_labels_for_metric = (
        lambda metric_name: PrometheusMetricLabels.get_labels(metric_name)
    )

    logger.increment_deployment_cooled_down(
        litellm_model_name="gpt-4o-mini",
        model_id="model-123",
        api_base="https://api.openai.com",
        api_provider="openai",
        exception_status="429",
        model_group="gpt-group",
    )

    labels = logger.litellm_deployment_cooled_down.labels.call_args.kwargs
    assert labels["model_group"] == "gpt-group"
    assert labels["litellm_model_name"] == "gpt-4o-mini"
    assert labels["model_id"] == "model-123"
    assert labels["exception_status"] == "429"
    logger.litellm_deployment_cooled_down.labels().inc.assert_called_once()


def test_set_litellm_deployment_state_emits_model_group(monkeypatch):
    monkeypatch.setattr(litellm, "prometheus_emit_deployment_model_group_label", True)
    logger = _logger_without_init()
    logger.litellm_deployment_state = MagicMock()
    logger.get_labels_for_metric = (
        lambda metric_name: PrometheusMetricLabels.get_labels(metric_name)
    )

    logger.set_litellm_deployment_state(
        state=2,
        litellm_model_name="gpt-4o-mini",
        model_id="model-123",
        api_base="https://api.openai.com",
        api_provider="openai",
        model_group="gpt-group",
    )

    labels = logger.litellm_deployment_state.labels.call_args.kwargs
    assert labels["model_group"] == "gpt-group"
    assert labels["model_id"] == "model-123"
    logger.litellm_deployment_state.labels().set.assert_called_with(2)


def test_set_deployment_tpm_rpm_limit_metrics_emit_model_group(monkeypatch):
    monkeypatch.setattr(litellm, "prometheus_emit_deployment_model_group_label", True)
    logger = _logger_without_init()
    logger.litellm_deployment_tpm_limit = MagicMock()
    logger.litellm_deployment_rpm_limit = MagicMock()
    logger.get_labels_for_metric = (
        lambda metric_name: PrometheusMetricLabels.get_labels(metric_name)
    )

    logger._set_deployment_tpm_rpm_limit_metrics(
        model_info={"tpm": 1000, "rpm": 60},
        litellm_params={},
        litellm_model_name="gpt-4o-mini",
        model_id="model-123",
        api_base="https://api.openai.com",
        llm_provider="openai",
        model_group="gpt-group",
    )

    assert (
        logger.litellm_deployment_tpm_limit.labels.call_args.kwargs["model_group"]
        == "gpt-group"
    )
    assert (
        logger.litellm_deployment_rpm_limit.labels.call_args.kwargs["model_group"]
        == "gpt-group"
    )


def test_deployment_metrics_omit_model_group_when_flag_disabled(monkeypatch):
    """With the flag off (default), the factory must not emit model_group even
    though the helper is handed a model_group value."""
    monkeypatch.setattr(litellm, "prometheus_emit_deployment_model_group_label", False)
    logger = _logger_without_init()
    logger.litellm_deployment_state = MagicMock()
    logger.get_labels_for_metric = (
        lambda metric_name: PrometheusMetricLabels.get_labels(metric_name)
    )

    logger.set_litellm_deployment_state(
        state=0,
        litellm_model_name="gpt-4o-mini",
        model_id="model-123",
        api_base="https://api.openai.com",
        api_provider="openai",
        model_group="gpt-group",
    )

    assert "model_group" not in logger.litellm_deployment_state.labels.call_args.kwargs


# ---------------------------------------------------------------------------
# Cooldown callback: alias -> model_group, underlying model -> litellm_model_name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_cooldown_callback_separates_alias_and_underlying_model(
    monkeypatch,
):
    """The deployment alias (model_name) becomes model_group while the
    prefix-stripped underlying provider model becomes litellm_model_name, and
    api_base is resolved from the underlying model rather than the alias."""
    mock_router = MagicMock()
    mock_router.get_deployment.return_value = {
        "litellm_params": {"model": "openai/gpt-4o-mini"},
        "model_name": "my-gpt-group",
        "model_info": ModelInfo(id="test-model-id"),
    }

    logger = _logger_without_init()
    logger.set_deployment_complete_outage = MagicMock()
    logger.increment_deployment_cooled_down = MagicMock()
    monkeypatch.setattr(litellm, "callbacks", [logger])

    with patch("litellm.get_api_base", return_value="https://api.openai.com") as gab:
        await router_cooldown_event_callback(
            litellm_router_instance=mock_router,
            deployment_id="test-deployment",
            exception_status="429",
            cooldown_time=60.0,
        )

    # api_base resolved from the underlying model, not the alias (P2).
    assert gab.call_args.kwargs["model"] == "gpt-4o-mini"

    for mock in (
        logger.set_deployment_complete_outage,
        logger.increment_deployment_cooled_down,
    ):
        kwargs = mock.call_args.kwargs
        assert kwargs["litellm_model_name"] == "gpt-4o-mini"
        assert kwargs["model_group"] == "my-gpt-group"
        assert kwargs["model_id"] == "test-model-id"
