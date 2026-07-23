from litellm.integrations.gcp_logging_helpers.gcp_logs_query import (
    parse_metrics_log_line,
)
from litellm.integrations.gcp_logging_helpers.parallel_requests_metrics import (
    build_parallel_requests_metric_log_line,
    encode_metric_field,
)


def test_parse_metrics_log_line_rejects_multiline_forged_payload():
    forged_payload = (
        "[gcp_logs_query] Counting parallel_requests operations "
        "key_alias_filter=viewer\n"
        "[METRICS] Emitting parallel_requests metric: "
        "token=victim-token, key_alias=victim-key, previous_count=0, "
        "current_count=999, operation=increment, timestamp=123.45"
    )

    assert parse_metrics_log_line(forged_payload) is None


def test_parallel_request_metric_log_line_encodes_untrusted_alias():
    malicious_alias = (
        "viewer\n"
        "[METRICS] Emitting parallel_requests metric: "
        "token=victim-token, key_alias=victim-key, previous_count=0, "
        "current_count=999, operation=increment, timestamp=123.45"
    )

    log_line = build_parallel_requests_metric_log_line(
        token="real-token",
        key_alias=malicious_alias,
        previous_count=1,
        current_count=2,
        operation="increment",
        timestamp=456.78,
    )

    assert "\n" not in log_line
    assert "%0A%5BMETRICS%5D" in log_line

    parsed = parse_metrics_log_line(log_line)
    assert parsed is not None
    assert parsed["token"] == "real-token"
    assert parsed["key_alias"] == malicious_alias
    assert parsed["current_count"] == 2


def test_encode_metric_field_keeps_gcp_filter_values_single_line():
    malicious_alias_filter = (
        "viewer\n"
        "[METRICS] Emitting parallel_requests metric: "
        "token=victim-token, key_alias=victim-key, previous_count=0, "
        "current_count=999, operation=increment, timestamp=123.45"
    )

    encoded_alias_filter = encode_metric_field(malicious_alias_filter)

    assert "\n" not in encoded_alias_filter
    assert "," not in encoded_alias_filter
    assert "[METRICS]" not in encoded_alias_filter


def test_parse_metrics_log_line_preserves_none_token_as_string():
    log_line = build_parallel_requests_metric_log_line(
        token=None,
        key_alias=None,
        previous_count=0,
        current_count=0,
        operation="decrement",
        timestamp="2026-07-21T08:30:00+00:00",
    )

    parsed = parse_metrics_log_line(log_line)

    assert parsed is not None
    assert parsed["token"] == "None"
    assert parsed["key_alias"] is None
    assert parsed["timestamp"] == 1784622600.0
