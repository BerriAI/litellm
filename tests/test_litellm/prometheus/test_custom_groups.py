import sys, os
sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.types.integrations.prometheus import (
    PrometheusMetricLabels,
    CustomPrometheusMetricGroup,
)


def test_custom_metric_group_labels():
    # configure custom metric group
    litellm.custom_prometheus_metric_groups = [
        CustomPrometheusMetricGroup(
            group="service_metrics",
            metrics=[
                "litellm_deployment_failure_responses",
                "litellm_deployment_total_requests",
                "litellm_proxy_failed_requests_metric",
                "litellm_proxy_total_requests_metric",
            ],
            include_labels=[
                "litellm_model_name",
                "requested_model",
                "api_base",
                "api_provider",
                "exception_status",
                "exception_class",
            ],
        )
    ]

    labels = PrometheusMetricLabels.get_labels(
        "litellm_deployment_failure_responses"
    )

    for lbl in [
        "litellm_model_name",
        "requested_model",
        "api_base",
        "api_provider",
        "exception_status",
        "exception_class",
    ]:
        assert lbl in labels

    # cleanup
    litellm.custom_prometheus_metric_groups = []
