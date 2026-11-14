"""
Unit tests for prometheus metric labels configuration
"""

from typing import Final

import pytest

from litellm.types.integrations.prometheus import (
    PrometheusMetricLabels,
    UserAPIKeyLabelNames,
)


def _clear_prometheus_registry() -> None:
    from prometheus_client import REGISTRY

    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


def _collected_samples(metric_name: str):
    from prometheus_client import REGISTRY

    return [
        sample
        for metric in REGISTRY.collect()
        for sample in metric.samples
        if sample.name == metric_name
    ]


def test_user_email_in_required_metrics():
    """
    Test that user_email label is present in all the metrics that should have it:
    - litellm_proxy_total_requests_metric (already had it)
    - litellm_proxy_failed_requests_metric (added)
    - litellm_input_tokens_metric (added)
    - litellm_output_tokens_metric (added)
    - litellm_requests_metric (already had it)
    - litellm_spend_metric (added)
    """
    user_email_label = UserAPIKeyLabelNames.USER_EMAIL.value

    # Metrics that should have user_email
    metrics_with_user_email = [
        "litellm_proxy_total_requests_metric",
        "litellm_proxy_failed_requests_metric",
        "litellm_input_tokens_metric",
        "litellm_output_tokens_metric",
        "litellm_requests_metric",
        "litellm_spend_metric",
    ]

    for metric_name in metrics_with_user_email:
        labels = PrometheusMetricLabels.get_labels(metric_name)
        assert (
            user_email_label in labels
        ), f"Metric {metric_name} should contain user_email label"
        print(f"✅ {metric_name} contains user_email label")


def test_model_id_in_extended_metric_set():
    """
    Test that model_id label is present in all the metrics that should have it
    """
    model_id_label = UserAPIKeyLabelNames.MODEL_ID.value

    # Metrics that should have model_id
    metrics_with_model_id = [
        "litellm_proxy_total_requests_metric",
        "litellm_proxy_failed_requests_metric",
        "litellm_input_tokens_metric",
        "litellm_output_tokens_metric",
        "litellm_requests_metric",
        "litellm_spend_metric",
        "litellm_llm_api_latency_metric",
        "litellm_remaining_requests_metric",
        "litellm_deployment_successful_fallbacks",
        "litellm_cache_hits_metric",
        "litellm_cache_misses_metric",
        "litellm_remaining_api_key_requests_for_model",
        "litellm_remaining_api_key_tokens_for_model",
        "litellm_llm_api_failed_requests_metric",
    ]

    for metric_name in metrics_with_model_id:
        labels = PrometheusMetricLabels.get_labels(metric_name)
        assert (
            model_id_label in labels
        ), f"Metric {metric_name} should contain model_id label"
        print(f"✅ {metric_name} contains model_id label")


def test_api_provider_in_spend_and_requests_metrics():
    """
    Test that api_provider label is present in spend and requests metrics
    so users can build spend-by-provider and request-count-by-provider dashboards.
    """
    api_provider_label = UserAPIKeyLabelNames.API_PROVIDER.value

    for metric_name in ["litellm_spend_metric", "litellm_requests_metric"]:
        labels = PrometheusMetricLabels.get_labels(metric_name)
        assert (
            api_provider_label in labels
        ), f"Metric {metric_name} should contain api_provider label"
        print(f"✅ {metric_name} contains api_provider label")


def test_api_provider_in_token_latency_and_request_metrics():
    """
    Regression test for LIT-4178.

    These metrics are all emitted from the same call site (async_log_success_event
    / async_post_call_failure_hook) as litellm_spend_metric and
    litellm_requests_metric, which already carry api_provider. They were the odd
    ones out with no way to break down tokens, latency, request counts or cache
    hits by upstream provider, even though the provider is available on the
    payload as custom_llm_provider.
    """
    api_provider_label = UserAPIKeyLabelNames.API_PROVIDER.value

    metrics_with_api_provider = [
        "litellm_llm_api_latency_metric",
        "litellm_llm_api_time_to_first_token_metric",
        "litellm_request_total_latency_metric",
        "litellm_request_queue_time_seconds",
        "litellm_proxy_total_requests_metric",
        "litellm_proxy_failed_requests_metric",
        "litellm_input_tokens_metric",
        "litellm_total_tokens_metric",
        "litellm_output_tokens_metric",
        "litellm_cache_hits_metric",
        "litellm_cache_misses_metric",
        # The remaining cache metrics share _cache_metric_labels, so they pick up
        # api_provider from the same change. Assert them explicitly so the shared
        # list can't silently drop the label from them.
        "litellm_cached_tokens_metric",
        "litellm_provider_cache_read_input_tokens_metric",
        "litellm_provider_cache_creation_input_tokens_metric",
    ]

    for metric_name in metrics_with_api_provider:
        labels = PrometheusMetricLabels.get_labels(metric_name)
        assert (
            api_provider_label in labels
        ), f"Metric {metric_name} should contain api_provider label"




def test_extract_api_provider_from_request_data_failure_path():
    """
    On the client-side failure path the provider is not always known. Prefer the
    resolved custom_llm_provider on litellm_params, fall back to a partial
    standard_logging_object (e.g. a stream that broke mid-flight), then infer it
    from the requested model name, and return None only when nothing maps so the
    label emits empty rather than a guess.
    """
    from litellm.integrations.prometheus import PrometheusLogger

    extract = PrometheusLogger._extract_api_provider_from_request_data

    assert extract({"litellm_params": {"custom_llm_provider": "bedrock"}}) == "bedrock"
    assert (
        extract({"standard_logging_object": {"custom_llm_provider": "vertex_ai"}})
        == "vertex_ai"
    )
    # litellm_params wins over standard_logging_object when both are present
    assert (
        extract(
            {
                "litellm_params": {"custom_llm_provider": "openai"},
                "standard_logging_object": {"custom_llm_provider": "azure"},
            }
        )
        == "openai"
    )
    # Fallback: infer provider from the requested model name when the proxy's
    # failure request_data carries only the client-supplied model. This is the
    # common client-side failure case (e.g. an invalid param rejected with 400).
    assert extract({"model": "gpt-4o-mini"}) == "openai"
    assert extract({"litellm_params": {"model": "anthropic/claude-haiku-4-5"}}) == "anthropic"
    assert extract({}) is None
    # Unmappable model name -> None, not a guess
    assert extract({"model": "some-unknown-model-xyz"}) is None


_PASS_THROUGH_CALL_TYPE: Final = "pass_through_endpoint"


@pytest.mark.parametrize(
    "request_data, expected_provider",
    [
        ({"model": "gpt-4o-mini", "call_type": "acompletion", "custom_llm_provider": "anthropic"}, "openai"),
        ({"model": "gpt-4o-mini", "litellm_params": {"custom_llm_provider": "humanloop"}}, "humanloop"),
        ({"model": "gemini-3.8-flash", "call_type": _PASS_THROUGH_CALL_TYPE}, "vertex_ai"),
        (
            {"model": "gemini-3.8-flash", "call_type": _PASS_THROUGH_CALL_TYPE, "custom_llm_provider": "gemini"},
            "gemini",
        ),
        (
            {
                "call_type": _PASS_THROUGH_CALL_TYPE,
                "custom_llm_provider": "gemini",
                "standard_logging_object": {"custom_llm_provider": "vertex_ai"},
            },
            "vertex_ai",
        ),
    ],
)
def test_extract_api_provider_reads_the_request_declared_provider_only_on_pass_through_failures(
    request_data: dict[str, object], expected_provider: str
):
    from litellm.integrations.prometheus import PrometheusLogger

    assert PrometheusLogger._extract_api_provider_from_request_data(request_data) == expected_provider


@pytest.mark.parametrize(
    "declared_provider",
    ["azure", "bedrock", "openrouter", "client-supplied-value", "", {"nested": "object"}, 42],
)
def test_extract_api_provider_ignores_pass_through_declared_values_outside_gemini_and_vertex(
    declared_provider: object,
):
    from litellm.integrations.prometheus import PrometheusLogger

    extract = PrometheusLogger._extract_api_provider_from_request_data
    declared = {"call_type": _PASS_THROUGH_CALL_TYPE, "custom_llm_provider": declared_provider}

    assert extract({"model": "gpt-4o-mini", **declared}) == "openai"
    assert extract({"model": "no-such-model", **declared}) is None


def test_extract_api_provider_swallows_unknown_model_but_logs_unexpected_errors():
    """
    get_llm_provider raises BadRequestError for a model that maps to no
    provider; that is the expected miss and must resolve to None quietly. Any
    other exception is unexpected (e.g. a real bug) and must not vanish
    silently, nor break metric emission: it is logged and still returns None.
    """
    from unittest.mock import patch

    import litellm
    from litellm.integrations.prometheus import PrometheusLogger

    extract = PrometheusLogger._extract_api_provider_from_request_data

    # Expected miss: BadRequestError -> None, no log noise
    with patch.object(
        litellm,
        "get_llm_provider",
        side_effect=litellm.exceptions.BadRequestError(
            message="no provider", model="x", llm_provider="y"
        ),
    ):
        with patch("litellm.integrations.prometheus.verbose_logger") as mock_logger:
            assert extract({"model": "x"}) is None
            mock_logger.debug.assert_not_called()

    # Unexpected error: must be logged and still return None (never raised)
    with patch.object(litellm, "get_llm_provider", side_effect=RuntimeError("boom")):
        with patch("litellm.integrations.prometheus.verbose_logger") as mock_logger:
            assert extract({"model": "x"}) is None
            mock_logger.debug.assert_called_once()


def test_user_email_label_exists():
    """Test that the USER_EMAIL label is properly defined"""
    assert UserAPIKeyLabelNames.USER_EMAIL.value == "user_email"


def test_prometheus_metric_labels_structure():
    """Test that all required prometheus metrics have proper label structure"""
    from typing import get_args

    from litellm.types.integrations.prometheus import DEFINED_PROMETHEUS_METRICS

    # Test a few key metrics to ensure they have proper label structure
    test_metrics = [
        "litellm_proxy_total_requests_metric",
        "litellm_proxy_failed_requests_metric",
        "litellm_input_tokens_metric",
        "litellm_output_tokens_metric",
        "litellm_spend_metric",
    ]

    for metric_name in test_metrics:
        # Check metric is in DEFINED_PROMETHEUS_METRICS
        assert metric_name in get_args(
            DEFINED_PROMETHEUS_METRICS
        ), f"{metric_name} should be in DEFINED_PROMETHEUS_METRICS"

        # Check labels can be retrieved
        labels = PrometheusMetricLabels.get_labels(metric_name)
        assert isinstance(labels, list), f"Labels for {metric_name} should be a list"
        assert len(labels) > 0, f"Labels for {metric_name} should not be empty"

        # Check user_email is in the labels
        assert "user_email" in labels, f"{metric_name} should have user_email label"

        print(f"✅ {metric_name} has proper label structure with user_email")


def test_model_id_in_required_metrics():
    """
    Test that model_id label is present in all the metrics that should have it:
    - litellm_proxy_total_requests_metric
    - litellm_proxy_failed_requests_metric
    - litellm_request_total_latency_metric
    - litellm_llm_api_time_to_first_token_metric
    """
    model_id_label = UserAPIKeyLabelNames.MODEL_ID.value

    # Metrics that should have model_id
    metrics_with_model_id = [
        "litellm_proxy_total_requests_metric",
        "litellm_proxy_failed_requests_metric",
        "litellm_request_total_latency_metric",
        "litellm_llm_api_time_to_first_token_metric",
    ]

    for metric_name in metrics_with_model_id:
        labels = PrometheusMetricLabels.get_labels(metric_name)
        assert (
            model_id_label in labels
        ), f"Metric {metric_name} should contain model_id label"
        print(f"✅ {metric_name} contains model_id label")


def test_requested_model_in_spend_and_requests_metrics():
    """
    Regression test for LIT-3796.

    litellm_spend_metric and litellm_requests_metric must expose the
    requested_model label so spend and request counts can be grouped by the
    model alias the caller asked for, not just the backend deployment that
    served the request. The sibling token metrics (input/output/total)
    already carry this label; spend and requests were the odd ones out, even
    though both are emitted side-by-side from the same call site.
    """
    requested_model_label = UserAPIKeyLabelNames.REQUESTED_MODEL.value

    metrics_with_requested_model = [
        "litellm_spend_metric",
        "litellm_requests_metric",
        "litellm_input_tokens_metric",
        "litellm_output_tokens_metric",
        "litellm_total_tokens_metric",
    ]

    for metric_name in metrics_with_requested_model:
        labels = PrometheusMetricLabels.get_labels(metric_name)
        assert requested_model_label in labels, (
            f"Metric {metric_name} should contain requested_model label"
        )


def test_route_normalization_for_responses_api():
    """
    Test that route normalization prevents high cardinality in Prometheus metrics
    for the /v1/responses/{response_id} endpoint.

    Issue: https://github.com/BerriAI/litellm/issues/XXXX
    Each unique response ID was creating a separate metric line, causing the
    /metrics endpoint to grow to ~30MB and take ~40 seconds to respond.

    Fix: Routes are normalized to collapse dynamic IDs into placeholders.
    """
    from litellm.proxy.auth.auth_utils import normalize_request_route

    # Test responses API routes
    responses_routes = [
        ("/v1/responses/1234567890", "/v1/responses/{response_id}"),
        ("/v1/responses/9876543210", "/v1/responses/{response_id}"),
        ("/v1/responses/abcdefghij", "/v1/responses/{response_id}"),
        ("/v1/responses/resp_abc123", "/v1/responses/{response_id}"),
        ("/v1/responses/litellm_poll_xyz", "/v1/responses/{response_id}"),
    ]

    for original, expected in responses_routes:
        normalized = normalize_request_route(original)
        assert (
            normalized == expected
        ), f"Failed: {original} -> {normalized} (expected {expected})"

    # Verify cardinality reduction
    unique_normalized = set(
        normalize_request_route(route) for route, _ in responses_routes
    )
    assert (
        len(unique_normalized) == 1
    ), f"Expected 1 unique normalized route, got {len(unique_normalized)}: {unique_normalized}"

    print(
        f"✅ Responses API routes: {len(responses_routes)} different IDs normalized to 1 metric label"
    )


def test_route_normalization_for_sub_routes():
    """Test that sub-routes like /cancel and /input_items are normalized correctly"""
    from litellm.proxy.auth.auth_utils import normalize_request_route

    sub_routes = [
        ("/v1/responses/id1/cancel", "/v1/responses/{response_id}/cancel"),
        ("/v1/responses/id2/cancel", "/v1/responses/{response_id}/cancel"),
        ("/v1/responses/id3/input_items", "/v1/responses/{response_id}/input_items"),
        (
            "/openai/v1/responses/id4/input_items",
            "/openai/v1/responses/{response_id}/input_items",
        ),
    ]

    for original, expected in sub_routes:
        normalized = normalize_request_route(original)
        assert (
            normalized == expected
        ), f"Failed: {original} -> {normalized} (expected {expected})"

    print("✅ Sub-routes normalized correctly")


def test_route_normalization_preserves_static_routes():
    """Test that static routes are not affected by normalization"""
    from litellm.proxy.auth.auth_utils import normalize_request_route

    static_routes = [
        "/chat/completions",
        "/v1/chat/completions",
        "/v1/embeddings",
        "/health",
        "/metrics",
        "/v1/models",
        "/v1/responses",  # List endpoint without ID
    ]

    for route in static_routes:
        normalized = normalize_request_route(route)
        assert (
            normalized == route
        ), f"Static route should not be modified: {route} -> {normalized}"

    print(f"✅ {len(static_routes)} static routes preserved")


def test_route_normalization_other_dynamic_apis():
    """Test normalization for other OpenAI-compatible APIs with dynamic IDs"""
    from litellm.proxy.auth.auth_utils import normalize_request_route

    test_cases = [
        # Threads API
        ("/v1/threads/thread_123", "/v1/threads/{thread_id}"),
        ("/v1/threads/thread_abc/messages", "/v1/threads/{thread_id}/messages"),
        (
            "/v1/threads/thread_abc/runs/run_123",
            "/v1/threads/{thread_id}/runs/{run_id}",
        ),
        # Vector Stores API
        ("/v1/vector_stores/vs_123", "/v1/vector_stores/{vector_store_id}"),
        ("/v1/vector_stores/vs_123/files", "/v1/vector_stores/{vector_store_id}/files"),
        # Assistants API
        ("/v1/assistants/asst_123", "/v1/assistants/{assistant_id}"),
        # Files API
        ("/v1/files/file_123", "/v1/files/{file_id}"),
        ("/v1/files/file_123/content", "/v1/files/{file_id}/content"),
        # Batches API
        ("/v1/batches/batch_123", "/v1/batches/{batch_id}"),
        ("/v1/batches/batch_123/cancel", "/v1/batches/{batch_id}/cancel"),
    ]

    for original, expected in test_cases:
        normalized = normalize_request_route(original)
        assert (
            normalized == expected
        ), f"Failed: {original} -> {normalized} (expected {expected})"

    print(f"✅ {len(test_cases)} other API routes normalized correctly")






def test_prometheus_label_value_sanitization_unicode_paragraph_separator():
    """Test that U+2029 (Paragraph Separator) is also stripped."""
    from litellm.types.integrations.prometheus import sanitize_prometheus_label_value

    result = sanitize_prometheus_label_value("model\u2029name")
    assert result == "modelname"
    assert "\u2029" not in result

    print("✅ U+2029 Paragraph Separator is stripped")


def test_prometheus_label_value_sanitization_none():
    """Test that None values pass through unchanged."""
    from litellm.types.integrations.prometheus import sanitize_prometheus_label_value

    assert sanitize_prometheus_label_value(None) is None

    print("✅ None values pass through unchanged")


def test_prometheus_label_value_sanitization_non_string_types():
    """Test that non-string values (int, bool, etc.) are coerced to str."""
    from litellm.types.integrations.prometheus import sanitize_prometheus_label_value

    assert sanitize_prometheus_label_value(200) == "200"
    assert sanitize_prometheus_label_value(True) == "True"
    assert sanitize_prometheus_label_value(3.14) == "3.14"

    print("✅ Non-string values are coerced to str")










async def _failed_requests_api_provider_labels(
    request_data: dict[str, object],
    original_exception: Exception,
) -> list[str]:
    from litellm.integrations.prometheus import PrometheusLogger
    from litellm.proxy._types import UserAPIKeyAuth

    _clear_prometheus_registry()
    try:
        await PrometheusLogger().async_post_call_failure_hook(
            request_data=request_data,
            original_exception=original_exception,
            user_api_key_dict=UserAPIKeyAuth(token="tok"),
        )
        return [
            s.labels.get("api_provider")
            for s in _collected_samples("litellm_proxy_failed_requests_metric_total")
        ]
    finally:
        _clear_prometheus_registry()










def test_model_group_in_deployment_metrics():
    """
    Test that model_group label is present on the deployment-scoped metrics
    needed to build model-group dashboards (request counts, success/failure
    counts, tpm/rpm limits). These metrics previously only carried
    requested_model, litellm_model_name and model_id, none of which identify
    the model_group a pooled deployment belongs to.
    """
    model_group_label = UserAPIKeyLabelNames.MODEL_GROUP.value

    metrics_with_model_group = [
        "litellm_deployment_total_requests",
        "litellm_deployment_success_responses",
        "litellm_deployment_failure_responses",
        "litellm_deployment_tpm_limit",
        "litellm_deployment_rpm_limit",
    ]

    for metric_name in metrics_with_model_group:
        labels = PrometheusMetricLabels.get_labels(metric_name)
        assert (
            model_group_label in labels
        ), f"Metric {metric_name} should contain model_group label"
        print(f"✅ {metric_name} contains model_group label")






def _routed(deployment: object) -> dict[str, object]:
    return {"deployment": deployment, "model_info": {"id": "model-123"}}








def test_model_group_in_latency_metrics():
    """
    Test that model_group label is present on the end-to-end / per-call
    latency metrics needed to build model-group latency dashboards. These
    metrics previously only carried requested_model, litellm_model_name and
    model_id, none of which identify the model_group a pooled deployment
    belongs to -- only the proxy-overhead-only latency metrics
    (litellm_overhead_latency_metric and friends) carried model_group.
    """
    model_group_label = UserAPIKeyLabelNames.MODEL_GROUP.value

    metrics_with_model_group = [
        "litellm_llm_api_latency_metric",
        "litellm_llm_api_time_to_first_token_metric",
        "litellm_request_total_latency_metric",
        "litellm_deployment_latency_per_output_token",
    ]

    for metric_name in metrics_with_model_group:
        labels = PrometheusMetricLabels.get_labels(metric_name)
        assert (
            model_group_label in labels
        ), f"Metric {metric_name} should contain model_group label"
        print(f"✅ {metric_name} contains model_group label")








if __name__ == "__main__":
    test_user_email_in_required_metrics()
    test_user_email_label_exists()
    test_prometheus_metric_labels_structure()
    test_route_normalization_for_responses_api()
    test_route_normalization_for_sub_routes()
    test_route_normalization_preserves_static_routes()
    test_route_normalization_other_dynamic_apis()
    test_prometheus_label_value_sanitization_unicode_paragraph_separator()
    test_prometheus_label_value_sanitization_none()
    test_prometheus_label_value_sanitization_non_string_types()
    print("\n✅ All prometheus label tests passed!")
