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
    samples = []
    for metric in logger.litellm_deployment_failure_responses.collect():
        for sample in metric.samples:
            if sample.name == "litellm_deployment_failure_responses_total":
                samples.append((dict(sample.labels), sample.value))
    return samples


def _request_kwargs(model: str, standard_custom_provider: str | None = None) -> dict:
    standard_logging_object = {
        "model_id": "resolved-deployment-id",
        "model_group": "my-model-group",
        "api_base": "https://example.com",
        "metadata": {},
    }
    if standard_custom_provider is not None:
        standard_logging_object["custom_llm_provider"] = standard_custom_provider
    return {
        "model": model,
        "exception": Exception("upstream 503"),
        "litellm_params": {
            # no "custom_llm_provider" key — the #38531 scenario
            "metadata": {},
        },
        "standard_logging_object": standard_logging_object,
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

    kwargs = _request_kwargs("openai/gpt-4o-mini")
    kwargs["litellm_params"]["custom_llm_provider"] = "azure"

    prometheus_logger.set_llm_deployment_failure_metrics(kwargs)

    samples = _failure_samples(prometheus_logger)
    assert len(samples) == 1
    assert samples[0][0]["api_provider"] == "azure"
