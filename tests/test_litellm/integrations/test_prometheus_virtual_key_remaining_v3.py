"""
Regression tests for LIT-2577 — Prometheus per-(API key, model) remaining
RPM/TPM gauges returning `sys.maxsize` (~9.22e18) under the v3 rate limiter.

The legacy v1 limiter writes the values into `metadata` under
`litellm-key-remaining-{requests,tokens}-{model_group}`. The active v3 limiter
(`parallel_request_limiter_v3.async_post_call_success_hook`) instead writes
them to `response._hidden_params["additional_headers"]` under
`x-ratelimit-{model_per_key,key}-remaining-{requests,tokens}`. `prometheus.py`
must read both, otherwise the gauges always fall through to `sys.maxsize` when
v3 is in use and Prometheus/DataDog dashboards become useless.

See https://linear.app/litellm-ai/issue/LIT-2577.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from prometheus_client import REGISTRY

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.integrations.prometheus import (  # noqa: E402
    PrometheusLogger,
    _coerce_remaining_to_int,
    _get_additional_headers_from_kwargs,
    _get_remaining_value_from_v3_headers,
)


@pytest.fixture(scope="function")
def prometheus_logger():
    """Fresh PrometheusLogger so each test starts from a clean registry."""
    for collector in list(REGISTRY._collector_to_names.keys()):
        REGISTRY.unregister(collector)
    return PrometheusLogger()


def _v3_kwargs(
    model_group="gpt-4o",
    remaining_requests=7,
    remaining_tokens=1234,
    descriptor="model_per_key",
):
    """
    Build the kwargs shape `_set_virtual_key_rate_limit_metrics` sees when
    `parallel_request_limiter_v3` produced the response.
    """
    return {
        "model": model_group,
        "litellm_params": {"metadata": {"model_group": model_group}},
        "standard_logging_object": {
            "hidden_params": {
                "additional_headers": {
                    f"x-ratelimit-{descriptor}-remaining-requests": remaining_requests,
                    f"x-ratelimit-{descriptor}-remaining-tokens": remaining_tokens,
                    f"x-ratelimit-{descriptor}-limit-requests": 100,
                    f"x-ratelimit-{descriptor}-limit-tokens": 10000,
                },
            },
        },
    }


class TestV3RemainingHeadersFallback:
    """Cover the four code paths in `_set_virtual_key_rate_limit_metrics`."""

    def test_v3_additional_headers_populate_per_key_per_model_gauges(
        self, prometheus_logger
    ):
        """
        Bug repro from LIT-2577: with metadata empty and v3 additional_headers
        populated, the gauges must read from additional_headers — not fall
        through to sys.maxsize.
        """
        kwargs = _v3_kwargs(remaining_requests=7, remaining_tokens=1234)

        with patch.object(
            prometheus_logger, "litellm_remaining_api_key_requests_for_model"
        ) as req_gauge, patch.object(
            prometheus_logger, "litellm_remaining_api_key_tokens_for_model"
        ) as tok_gauge:
            prometheus_logger._set_virtual_key_rate_limit_metrics(
                user_api_key="hashed-key",
                user_api_key_alias="alias",
                kwargs=kwargs,
                metadata={},
                model_id="model-id",
            )

        req_gauge.labels.return_value.set.assert_called_once_with(7)
        tok_gauge.labels.return_value.set.assert_called_once_with(1234)

    def test_v3_key_scope_headers_used_when_model_per_key_absent(
        self, prometheus_logger
    ):
        """
        When only the broader `key` scope headers are present (per-key
        cross-model limit), the gauges fall back to those.
        """
        kwargs = _v3_kwargs(
            remaining_requests=42, remaining_tokens=555, descriptor="key"
        )

        with patch.object(
            prometheus_logger, "litellm_remaining_api_key_requests_for_model"
        ) as req_gauge, patch.object(
            prometheus_logger, "litellm_remaining_api_key_tokens_for_model"
        ) as tok_gauge:
            prometheus_logger._set_virtual_key_rate_limit_metrics(
                user_api_key="hashed-key",
                user_api_key_alias="alias",
                kwargs=kwargs,
                metadata={},
                model_id="model-id",
            )

        req_gauge.labels.return_value.set.assert_called_once_with(42)
        tok_gauge.labels.return_value.set.assert_called_once_with(555)

    def test_legacy_v1_metadata_takes_precedence_over_v3_headers(
        self, prometheus_logger
    ):
        """
        Back-compat: when both sources are present the v1 metadata values win,
        preserving the pre-fix contract for the legacy limiter.
        """
        kwargs = _v3_kwargs(remaining_requests=7, remaining_tokens=1234)
        metadata = {
            "litellm-key-remaining-requests-gpt-4o": 99,
            "litellm-key-remaining-tokens-gpt-4o": 8888,
        }

        with patch.object(
            prometheus_logger, "litellm_remaining_api_key_requests_for_model"
        ) as req_gauge, patch.object(
            prometheus_logger, "litellm_remaining_api_key_tokens_for_model"
        ) as tok_gauge:
            prometheus_logger._set_virtual_key_rate_limit_metrics(
                user_api_key="hashed-key",
                user_api_key_alias="alias",
                kwargs=kwargs,
                metadata=metadata,
                model_id="model-id",
            )

        req_gauge.labels.return_value.set.assert_called_once_with(99)
        tok_gauge.labels.return_value.set.assert_called_once_with(8888)

    def test_no_source_populated_still_emits_sys_maxsize(self, prometheus_logger):
        """
        With neither v1 metadata nor v3 additional_headers populated (no rate
        limit configured for this key/model), preserve the historical
        `sys.maxsize` sentinel.
        """
        import sys as _sys

        kwargs = {
            "model": "gpt-4o",
            "litellm_params": {"metadata": {"model_group": "gpt-4o"}},
            "standard_logging_object": {"hidden_params": {"additional_headers": {}}},
        }

        with patch.object(
            prometheus_logger, "litellm_remaining_api_key_requests_for_model"
        ) as req_gauge, patch.object(
            prometheus_logger, "litellm_remaining_api_key_tokens_for_model"
        ) as tok_gauge:
            prometheus_logger._set_virtual_key_rate_limit_metrics(
                user_api_key="hashed-key",
                user_api_key_alias="alias",
                kwargs=kwargs,
                metadata={},
                model_id="model-id",
            )

        req_gauge.labels.return_value.set.assert_called_once_with(_sys.maxsize)
        tok_gauge.labels.return_value.set.assert_called_once_with(_sys.maxsize)

    def test_partial_metadata_falls_back_to_v3_for_missing_field(
        self, prometheus_logger
    ):
        """
        If only one of the two metadata keys is set, the other still falls
        back to additional_headers rather than to sys.maxsize.
        """
        kwargs = _v3_kwargs(remaining_requests=7, remaining_tokens=1234)
        metadata = {
            # only the requests metadata key — tokens must come from v3 headers
            "litellm-key-remaining-requests-gpt-4o": 50,
        }

        with patch.object(
            prometheus_logger, "litellm_remaining_api_key_requests_for_model"
        ) as req_gauge, patch.object(
            prometheus_logger, "litellm_remaining_api_key_tokens_for_model"
        ) as tok_gauge:
            prometheus_logger._set_virtual_key_rate_limit_metrics(
                user_api_key="hashed-key",
                user_api_key_alias="alias",
                kwargs=kwargs,
                metadata=metadata,
                model_id="model-id",
            )

        req_gauge.labels.return_value.set.assert_called_once_with(50)
        tok_gauge.labels.return_value.set.assert_called_once_with(1234)

    def test_string_header_values_are_coerced_to_int(self, prometheus_logger):
        """
        Header values can be stringified upstream. The fix coerces with
        `int(...)` so the gauge stays numeric.
        """
        kwargs = _v3_kwargs(remaining_requests="42", remaining_tokens="9930")

        with patch.object(
            prometheus_logger, "litellm_remaining_api_key_requests_for_model"
        ) as req_gauge, patch.object(
            prometheus_logger, "litellm_remaining_api_key_tokens_for_model"
        ) as tok_gauge:
            prometheus_logger._set_virtual_key_rate_limit_metrics(
                user_api_key="hashed-key",
                user_api_key_alias="alias",
                kwargs=kwargs,
                metadata={},
                model_id="model-id",
            )

        req_gauge.labels.return_value.set.assert_called_once_with(42)
        tok_gauge.labels.return_value.set.assert_called_once_with(9930)


class TestPureHelpers:
    """Direct coverage of the static helpers so they stay simple to reason about."""

    def test_get_additional_headers_returns_empty_when_anything_missing(self):
        assert _get_additional_headers_from_kwargs({}) == {}
        assert _get_additional_headers_from_kwargs({"standard_logging_object": None}) == {}
        assert (
            _get_additional_headers_from_kwargs(
                {"standard_logging_object": {"hidden_params": None}}
            )
            == {}
        )
        assert (
            _get_additional_headers_from_kwargs(
                {
                    "standard_logging_object": {
                        "hidden_params": {"additional_headers": None}
                    }
                }
            )
            == {}
        )

    def test_get_additional_headers_returns_the_dict(self):
        headers = {"x-ratelimit-model_per_key-remaining-requests": 7}
        kwargs = {
            "standard_logging_object": {
                "hidden_params": {"additional_headers": headers}
            }
        }
        assert _get_additional_headers_from_kwargs(kwargs) == headers

    def test_remaining_value_prefers_model_per_key_over_key(self):
        headers = {
            "x-ratelimit-model_per_key-remaining-requests": 5,
            "x-ratelimit-key-remaining-requests": 99,
        }
        assert _get_remaining_value_from_v3_headers(headers, "requests") == 5

    def test_remaining_value_falls_back_to_key_scope(self):
        headers = {"x-ratelimit-key-remaining-tokens": 1234}
        assert _get_remaining_value_from_v3_headers(headers, "tokens") == 1234

    def test_remaining_value_returns_none_when_absent(self):
        assert _get_remaining_value_from_v3_headers({}, "requests") is None

    def test_coerce_remaining_handles_none_int_str_garbage(self):
        import sys as _sys

        assert _coerce_remaining_to_int(None) == _sys.maxsize
        assert _coerce_remaining_to_int(7) == 7
        assert _coerce_remaining_to_int("42") == 42
        assert _coerce_remaining_to_int("not-a-number") == _sys.maxsize
        assert _coerce_remaining_to_int(object()) == _sys.maxsize
