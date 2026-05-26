"""
Regression tests for LIT-2577.

Until this fix, ``PrometheusLogger._set_virtual_key_rate_limit_metrics``
only consulted the v1 ``parallel_request_limiter`` keys
(``litellm-key-remaining-{requests,tokens}-{model_group}`` in ``metadata``).
The active ``parallel_request_limiter_v3`` does not write those; it writes
to ``response._hidden_params["additional_headers"]`` under
``x-ratelimit-{model_per_key,key}-remaining-{requests,tokens}``. Result:
``litellm_remaining_api_key_{requests,tokens}_for_model`` always reported
``sys.maxsize`` (~9.22e18) in DataDog/Prometheus.

These tests pin the v3 fallback so the bug stays fixed.
"""
from __future__ import annotations

import sys
from typing import Optional

import pytest
from prometheus_client import REGISTRY

from litellm.integrations.prometheus import (
    PrometheusLogger,
    _get_additional_headers_from_kwargs,
    _get_remaining_from_v3_headers,
)

MODEL_GROUP = "gpt-4o-mini"


def _clear_prometheus_registry() -> None:
    for collector in list(REGISTRY._collector_to_names.keys()):
        REGISTRY.unregister(collector)


@pytest.fixture
def prometheus_logger() -> PrometheusLogger:
    _clear_prometheus_registry()
    return PrometheusLogger()


def _slp_with_additional_headers(additional_headers: Optional[dict]) -> dict:
    """Minimal standard_logging_object shape ``_set_virtual_key_rate_limit_metrics`` touches."""
    payload: dict = {
        "model_id": "model-123",
        "model_group": MODEL_GROUP,
        "metadata": {
            "user_api_key_hash": "test-hash",
            "user_api_key_alias": "test-alias",
            "user_api_key_team_id": None,
            "user_api_key_team_alias": None,
            "user_api_key_user_id": None,
            "user_api_key_user_email": None,
            "user_api_key_org_id": None,
            "requester_metadata": None,
            "user_api_key_auth_metadata": None,
            "spend_logs_metadata": None,
        },
    }
    if additional_headers is not None:
        payload["hidden_params"] = {"additional_headers": additional_headers}
    return payload


def _sample(metric_name: str):
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name == metric_name:
                return sample
    return None


def test_v3_additional_headers_populate_remaining_gauges(prometheus_logger):
    """v3 path: only ``additional_headers`` populated -> gauges read from there."""
    metadata = {"model_group": MODEL_GROUP}  # no legacy litellm-key-remaining-* keys
    kwargs = {
        "litellm_params": {"metadata": metadata},
        "standard_logging_object": _slp_with_additional_headers(
            {
                "x-ratelimit-model_per_key-remaining-requests": 7,
                "x-ratelimit-model_per_key-remaining-tokens": 1234,
                "x-ratelimit-model_per_key-limit-requests": 10,
                "x-ratelimit-model_per_key-limit-tokens": 2000,
            }
        ),
    }

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="test-hash",
        user_api_key_alias="test-alias",
        kwargs=kwargs,
        metadata=metadata,
        model_id="model-123",
    )

    req = _sample("litellm_remaining_api_key_requests_for_model")
    tok = _sample("litellm_remaining_api_key_tokens_for_model")
    assert req is not None and req.value == 7
    assert tok is not None and tok.value == 1234


def test_v3_key_descriptor_used_when_model_per_key_missing(prometheus_logger):
    """v3 path: per-(key, model) header missing -> fall back to per-key header."""
    metadata = {"model_group": MODEL_GROUP}
    kwargs = {
        "litellm_params": {"metadata": metadata},
        "standard_logging_object": _slp_with_additional_headers(
            {
                "x-ratelimit-key-remaining-requests": 42,
                "x-ratelimit-key-remaining-tokens": 9001,
            }
        ),
    }

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="test-hash",
        user_api_key_alias="test-alias",
        kwargs=kwargs,
        metadata=metadata,
        model_id="model-123",
    )

    assert _sample("litellm_remaining_api_key_requests_for_model").value == 42
    assert _sample("litellm_remaining_api_key_tokens_for_model").value == 9001


def test_model_per_key_preferred_over_key(prometheus_logger):
    """When both descriptors are present, ``model_per_key`` (more specific) wins."""
    metadata = {"model_group": MODEL_GROUP}
    kwargs = {
        "litellm_params": {"metadata": metadata},
        "standard_logging_object": _slp_with_additional_headers(
            {
                "x-ratelimit-key-remaining-requests": 100,
                "x-ratelimit-model_per_key-remaining-requests": 5,
                "x-ratelimit-key-remaining-tokens": 5000,
                "x-ratelimit-model_per_key-remaining-tokens": 100,
            }
        ),
    }

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="test-hash",
        user_api_key_alias="test-alias",
        kwargs=kwargs,
        metadata=metadata,
        model_id="model-123",
    )

    assert _sample("litellm_remaining_api_key_requests_for_model").value == 5
    assert _sample("litellm_remaining_api_key_tokens_for_model").value == 100


def test_v1_metadata_keys_still_take_precedence(prometheus_logger):
    """v1 metadata keys are the existing public contract -> still win over v3 fallback."""
    metadata = {
        "model_group": MODEL_GROUP,
        "litellm-key-remaining-requests-gpt-4o-mini": 11,
        "litellm-key-remaining-tokens-gpt-4o-mini": 22,
    }
    kwargs = {
        "litellm_params": {"metadata": metadata},
        "standard_logging_object": _slp_with_additional_headers(
            {
                "x-ratelimit-model_per_key-remaining-requests": 999,
                "x-ratelimit-model_per_key-remaining-tokens": 999,
            }
        ),
    }

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="test-hash",
        user_api_key_alias="test-alias",
        kwargs=kwargs,
        metadata=metadata,
        model_id="model-123",
    )

    assert _sample("litellm_remaining_api_key_requests_for_model").value == 11
    assert _sample("litellm_remaining_api_key_tokens_for_model").value == 22


def test_v3_zero_remaining_preserved(prometheus_logger):
    """``0`` is a meaningful value (key exhausted) and must not get coerced to sys.maxsize."""
    metadata = {"model_group": MODEL_GROUP}
    kwargs = {
        "litellm_params": {"metadata": metadata},
        "standard_logging_object": _slp_with_additional_headers(
            {
                "x-ratelimit-model_per_key-remaining-requests": 0,
                "x-ratelimit-model_per_key-remaining-tokens": 0,
            }
        ),
    }

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="test-hash",
        user_api_key_alias="test-alias",
        kwargs=kwargs,
        metadata=metadata,
        model_id="model-123",
    )

    assert _sample("litellm_remaining_api_key_requests_for_model").value == 0
    assert _sample("litellm_remaining_api_key_tokens_for_model").value == 0


def test_falls_back_to_maxsize_when_neither_path_has_data(prometheus_logger):
    """No v1 keys, no v3 keys -> existing behavior (sys.maxsize) preserved."""
    metadata = {"model_group": MODEL_GROUP}
    kwargs = {
        "litellm_params": {"metadata": metadata},
        "standard_logging_object": _slp_with_additional_headers({}),
    }

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="test-hash",
        user_api_key_alias="test-alias",
        kwargs=kwargs,
        metadata=metadata,
        model_id="model-123",
    )

    assert _sample("litellm_remaining_api_key_requests_for_model").value == pytest.approx(
        float(sys.maxsize)
    )
    assert _sample("litellm_remaining_api_key_tokens_for_model").value == pytest.approx(
        float(sys.maxsize)
    )


def test_string_values_in_additional_headers_are_coerced(prometheus_logger):
    """Headers travelling through HTTP/serialization can land as strings -> coerce to int."""
    metadata = {"model_group": MODEL_GROUP}
    kwargs = {
        "litellm_params": {"metadata": metadata},
        "standard_logging_object": _slp_with_additional_headers(
            {
                "x-ratelimit-model_per_key-remaining-requests": "8",
                "x-ratelimit-model_per_key-remaining-tokens": "777",
            }
        ),
    }

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="test-hash",
        user_api_key_alias="test-alias",
        kwargs=kwargs,
        metadata=metadata,
        model_id="model-123",
    )

    assert _sample("litellm_remaining_api_key_requests_for_model").value == 8
    assert _sample("litellm_remaining_api_key_tokens_for_model").value == 777


# ----- helper unit tests -------------------------------------------------


def test_get_additional_headers_tolerates_missing_layers():
    """Helper must not raise on partially-populated standard_logging_object."""
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
            {"standard_logging_object": {"hidden_params": {"additional_headers": None}}}
        )
        == {}
    )
    # non-dict types at each layer
    assert _get_additional_headers_from_kwargs({"standard_logging_object": "oops"}) == {}
    assert (
        _get_additional_headers_from_kwargs(
            {"standard_logging_object": {"hidden_params": "oops"}}
        )
        == {}
    )


def test_get_remaining_from_v3_headers_returns_none_when_absent():
    assert _get_remaining_from_v3_headers({}, "requests") is None
    assert (
        _get_remaining_from_v3_headers({"x-ratelimit-foo-remaining-requests": 1}, "requests")
        is None
    )


def test_get_remaining_from_v3_headers_prefers_model_per_key_over_key():
    headers = {
        "x-ratelimit-key-remaining-requests": 100,
        "x-ratelimit-model_per_key-remaining-requests": 5,
    }
    assert _get_remaining_from_v3_headers(headers, "requests") == 5
