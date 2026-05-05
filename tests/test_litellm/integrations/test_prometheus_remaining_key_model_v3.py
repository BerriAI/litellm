"""
Regression test for the bug where litellm_remaining_api_key_{requests,tokens}_for_model
emitted sys.maxsize (~9.22e18) when the v3 parallel_request_limiter was in use.

The v3 limiter writes the per-key-per-model remaining values into
    response._hidden_params["additional_headers"]
        ["x-ratelimit-model_per_key-remaining-{tokens,requests}"]
whereas the older v1 limiter wrote them into request metadata under
    "litellm-key-remaining-{tokens,requests}-{model_group}".

Prometheus must read both locations.
"""
import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
)


@pytest.fixture(scope="module")
def prometheus_logger():
    pytest.importorskip("prometheus_client")
    from litellm.integrations.prometheus import PrometheusLogger

    return PrometheusLogger()


def _gauge_value(gauge, labels):
    return gauge.labels(*labels)._value.get()


def test_v3_limiter_additional_headers_populates_remaining_gauges(prometheus_logger):
    """v3 limiter path: values present only in hidden_params.additional_headers."""
    kwargs = {
        "model": "gpt-4o-mini",
        "litellm_params": {"metadata": {"model_group": "gpt-4o-mini"}},
        "standard_logging_object": {
            "hidden_params": {
                "additional_headers": {
                    "x-ratelimit-model_per_key-remaining-tokens": 1234,
                    "x-ratelimit-model_per_key-remaining-requests": 17,
                    "x-ratelimit-model_per_key-limit-tokens": 100000,
                    "x-ratelimit-model_per_key-limit-requests": 60,
                }
            }
        },
    }
    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="hashed-v3",
        user_api_key_alias="alias-v3",
        kwargs=kwargs,
        metadata={},
        model_id="deployment-v3",
    )
    labels = ("hashed-v3", "alias-v3", "gpt-4o-mini", "deployment-v3")
    req = _gauge_value(
        prometheus_logger.litellm_remaining_api_key_requests_for_model, labels
    )
    tok = _gauge_value(
        prometheus_logger.litellm_remaining_api_key_tokens_for_model, labels
    )
    assert int(req) == 17, f"expected 17, got {req} (sys.maxsize fallthrough?)"
    assert int(tok) == 1234, f"expected 1234, got {tok} (sys.maxsize fallthrough?)"
    # Make sure we did NOT emit sys.maxsize for either gauge.
    assert int(req) < sys.maxsize
    assert int(tok) < sys.maxsize


def test_v1_limiter_metadata_path_still_works(prometheus_logger):
    """v1 limiter path: values written to metadata."""
    kwargs = {
        "model": "gpt-4o-mini",
        "litellm_params": {"metadata": {"model_group": "gpt-4o-mini"}},
    }
    metadata = {
        "litellm-key-remaining-requests-gpt-4o-mini": 7,
        "litellm-key-remaining-tokens-gpt-4o-mini": 9999,
        "model_group": "gpt-4o-mini",
    }
    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="hashed-v1",
        user_api_key_alias="alias-v1",
        kwargs=kwargs,
        metadata=metadata,
        model_id="deployment-v1",
    )
    labels = ("hashed-v1", "alias-v1", "gpt-4o-mini", "deployment-v1")
    assert (
        int(
            _gauge_value(
                prometheus_logger.litellm_remaining_api_key_requests_for_model, labels
            )
        )
        == 7
    )
    assert (
        int(
            _gauge_value(
                prometheus_logger.litellm_remaining_api_key_tokens_for_model, labels
            )
        )
        == 9999
    )


def test_no_data_falls_back_to_sys_maxsize(prometheus_logger):
    """When neither metadata nor additional_headers carry the values, gauges fall back to sys.maxsize."""
    kwargs = {"litellm_params": {"metadata": {"model_group": "no-data-mg"}}}
    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="hashed-empty",
        user_api_key_alias="alias-empty",
        kwargs=kwargs,
        metadata={},
        model_id="deployment-empty",
    )
    labels = ("hashed-empty", "alias-empty", "no-data-mg", "deployment-empty")
    req = _gauge_value(
        prometheus_logger.litellm_remaining_api_key_requests_for_model, labels
    )
    # float(sys.maxsize) rounds up to sys.maxsize+1 in IEEE-754; allow either.
    assert int(req) >= sys.maxsize
