"""api_provider label is present on the latency histogram and the
proxy total/failed request counters.

Without it, Grafana dashboards cannot group these series by provider
(openai, anthropic, gemini, …) and have to fall back to regex on `model`,
which is brittle for cross-provider routing.
"""

from litellm.types.integrations.prometheus import (
    PrometheusMetricLabels,
)


def test_api_provider_on_total_latency_metric():
    assert (
        "api_provider"
        in PrometheusMetricLabels.litellm_request_total_latency_metric
    )


def test_api_provider_on_proxy_total_requests_metric():
    assert (
        "api_provider"
        in PrometheusMetricLabels.litellm_proxy_total_requests_metric
    )


def test_api_provider_on_proxy_failed_requests_metric():
    assert (
        "api_provider"
        in PrometheusMetricLabels.litellm_proxy_failed_requests_metric
    )
