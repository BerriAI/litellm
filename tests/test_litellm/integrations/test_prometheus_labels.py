"""
Unit tests for prometheus metric labels configuration
"""
from litellm.types.integrations.prometheus import (
    PrometheusMetricLabels,
    UserAPIKeyLabelNames,
)


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


def test_model_id_in_required_metrics():
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
        "litellm_llm_api_time_to_first_token_metric"
    ]

    for metric_name in metrics_with_model_id:
        labels = PrometheusMetricLabels.get_labels(metric_name)
        assert model_id_label in labels, f"Metric {metric_name} should contain model_id label"
        print(f"✅ {metric_name} contains model_id label")


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


def test_prometheus_metrics_use_normalized_routes():
    """
    Test that Prometheus metrics use the normalized route in labels
    to prevent high cardinality.
    """
    from unittest.mock import MagicMock

    from litellm.integrations.prometheus import (
        PrometheusLogger,
        UserAPIKeyLabelValues,
        prometheus_label_factory,
    )

    # Create a mock PrometheusLogger
    prometheus_logger = MagicMock()
    prometheus_logger.get_labels_for_metric = (
        PrometheusLogger.get_labels_for_metric.__get__(prometheus_logger)
    )

    # Test with a normalized route
    enum_values = UserAPIKeyLabelValues(
        route="/v1/responses/{response_id}",  # Normalized route
        status_code="200",
        requested_model="gpt-4",
    )

    labels = prometheus_label_factory(
        supported_enum_labels=prometheus_logger.get_labels_for_metric(
            metric_name="litellm_proxy_total_requests_metric"
        ),
        enum_values=enum_values,
    )

    # Verify the route is normalized in labels
    assert (
        labels["route"] == "/v1/responses/{response_id}"
    ), f"Expected normalized route in labels, got: {labels.get('route')}"

    print("✅ Prometheus metrics use normalized routes in labels")


def test_prometheus_label_value_sanitization():
    """
    Test that Prometheus label values are sanitized to prevent breaking
    the Prometheus text format.

    Issue: Unicode Line Separator (U+2028) in label values (e.g. from a
    malformed model name) breaks the Prometheus exposition format, causing
    scrapers like Datadog to fail parsing the entire /metrics endpoint.
    """
    from litellm.integrations.prometheus import (
        PrometheusLogger,
        UserAPIKeyLabelValues,
        prometheus_label_factory,
    )
    from unittest.mock import MagicMock

    prometheus_logger = MagicMock()
    prometheus_logger.get_labels_for_metric = (
        PrometheusLogger.get_labels_for_metric.__get__(prometheus_logger)
    )

    # Simulate a model name with U+2028 (Unicode Line Separator) appended
    # and an api_key_alias with newlines and quotes
    enum_values = UserAPIKeyLabelValues(
        requested_model="claude-haiku-4-5-20251001\u2028",
        api_key_alias='My Key "test"\nwith newline',
        route="/v1/chat/completions",
        status_code="400",
    )

    labels = prometheus_label_factory(
        supported_enum_labels=prometheus_logger.get_labels_for_metric(
            metric_name="litellm_proxy_total_requests_metric"
        ),
        enum_values=enum_values,
    )

    # U+2028 must be stripped
    assert "\u2028" not in labels["requested_model"], (
        f"U+2028 should be removed from label value, got: {repr(labels['requested_model'])}"
    )
    assert labels["requested_model"] == "claude-haiku-4-5-20251001"

    # Newlines must be replaced with spaces, quotes must be escaped
    assert "\n" not in labels["api_key_alias"]
    assert labels["api_key_alias"] == 'My Key \\"test\\" with newline'

    print("✅ Prometheus label values are properly sanitized")


def test_prometheus_label_value_sanitization_unicode_paragraph_separator():
    """Test that U+2029 (Paragraph Separator) is also stripped."""
    from litellm.types.integrations.prometheus import _sanitize_prometheus_label_value

    result = _sanitize_prometheus_label_value("model\u2029name")
    assert result == "modelname"
    assert "\u2029" not in result

    print("✅ U+2029 Paragraph Separator is stripped")


def test_prometheus_label_value_sanitization_none():
    """Test that None values pass through unchanged."""
    from litellm.types.integrations.prometheus import _sanitize_prometheus_label_value

    assert _sanitize_prometheus_label_value(None) is None

    print("✅ None values pass through unchanged")


def test_prometheus_label_value_sanitization_non_string_types():
    """Test that non-string values (int, bool, etc.) are coerced to str."""
    from litellm.types.integrations.prometheus import _sanitize_prometheus_label_value

    assert _sanitize_prometheus_label_value(200) == "200"
    assert _sanitize_prometheus_label_value(True) == "True"
    assert _sanitize_prometheus_label_value(3.14) == "3.14"

    print("✅ Non-string values are coerced to str")


if __name__ == "__main__":
    test_user_email_in_required_metrics()
    test_user_email_label_exists()
    test_prometheus_metric_labels_structure()
    test_route_normalization_for_responses_api()
    test_route_normalization_for_sub_routes()
    test_route_normalization_preserves_static_routes()
    test_route_normalization_other_dynamic_apis()
    test_prometheus_metrics_use_normalized_routes()
    test_prometheus_label_value_sanitization()
    test_prometheus_label_value_sanitization_unicode_paragraph_separator()
    test_prometheus_label_value_sanitization_none()
    test_prometheus_label_value_sanitization_non_string_types()
    print("\n✅ All prometheus label tests passed!")
