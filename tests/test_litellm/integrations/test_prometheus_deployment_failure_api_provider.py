"""
Regression tests for #38531: the deployment-failure counter must never emit
api_provider="None". When a resolved deployment fails without
custom_llm_provider on litellm_params, fall back to the best-effort resolver
and coalesce to "" so alerting never sees a phantom all-failure series.
"""

import pytest
from prometheus_client import REGISTRY

from litellm.integrations.prometheus import PrometheusLogger


@pytest.fixture(scope="function")
def prometheus_logger():
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    return PrometheusLogger()


def _failure_samples(logger: PrometheusLogger):
    """Collected samples of the deployment-failure counter, as (labels, value)."""
    return [
        (dict(sample.labels), sample.value)
        for metric in logger.litellm_deployment_failure_responses.collect()
        for sample in metric.samples
        if sample.name == "litellm_deployment_failure_responses_total"
    ]


def _request_kwargs(model: str, litellm_params_provider: str | None = None) -> dict:
    """A deployment-failure request as the proxy hands it to the logger.

    Omitting litellm_params_provider leaves custom_llm_provider unset — the #38531 scenario.
    """
    return {
        "model": model,
        "exception": Exception("upstream 503"),
        "litellm_params": {
            "metadata": {},
            **({} if litellm_params_provider is None else {"custom_llm_provider": litellm_params_provider}),
        },
        "standard_logging_object": {
            "model_id": "resolved-deployment-id",
            "model_group": "my-model-group",
            "api_base": "https://example.com",
            "metadata": {},
        },
    }


def test_missing_provider_on_resolved_deployment_never_labels_none(prometheus_logger):
    """custom_llm_provider absent and the model name is not provider-prefixed:
    the label must be "" (not the literal "None")."""

    prometheus_logger.set_llm_deployment_failure_metrics(_request_kwargs("my-fallback-model"))

    samples = _failure_samples(prometheus_logger)
    assert len(samples) == 1
    assert samples[0][0]["api_provider"] == ""


def test_missing_provider_infers_from_provider_prefixed_model(prometheus_logger):
    """The resolver can still infer the provider from a prefixed model name."""

    prometheus_logger.set_llm_deployment_failure_metrics(_request_kwargs("openai/gpt-4o-mini"))

    samples = _failure_samples(prometheus_logger)
    assert len(samples) == 1
    assert samples[0][0]["api_provider"] == "openai"


def test_explicit_provider_still_wins(prometheus_logger):
    """Behavior is unchanged when custom_llm_provider is present."""

    prometheus_logger.set_llm_deployment_failure_metrics(
        _request_kwargs("openai/gpt-4o-mini", litellm_params_provider="azure")
    )

    samples = _failure_samples(prometheus_logger)
    assert len(samples) == 1
    assert samples[0][0]["api_provider"] == "azure"
