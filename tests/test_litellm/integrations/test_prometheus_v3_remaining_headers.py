"""
Regression test for: Prometheus per-key remaining gauges show 9e18 (sys.maxsize)
when parallel_request_limiter_v3 is in use.

The v3 limiter publishes remaining counters on the response under
hidden_params.additional_headers as
    "x-ratelimit-model_per_key-remaining-{requests,tokens}"
rather than into request metadata under
    "litellm-key-remaining-{requests,tokens}-{model_group}".

prometheus.py must read both, otherwise DataDog / Prometheus display
sys.maxsize (~9.22e18) instead of the real remaining values.
"""
import os
import sys

import pytest
from prometheus_client import REGISTRY

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.integrations.prometheus import PrometheusLogger


@pytest.fixture(scope="function")
def prometheus_logger():
    for c in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(c)
        except Exception:
            pass
    return PrometheusLogger()


class TestVirtualKeyRateLimitMetricsV3Headers:
    def _call(self, logger, *, metadata, additional_headers):
        kwargs = {
            "litellm_params": {
                "metadata": {"model_group": "gpt-4o"},
                "model_group": "gpt-4o",
            },
            "model": "gpt-4o",
        }
        slp = {
            "hidden_params": {"additional_headers": additional_headers}
        }
        logger._set_virtual_key_rate_limit_metrics(
            user_api_key="hashed-key",
            user_api_key_alias="alias",
            kwargs=kwargs,
            metadata=metadata,
            model_id="m1",
            standard_logging_payload=slp,
        )
        labels = ("hashed-key", "alias", "gpt-4o", "m1")
        req = logger.litellm_remaining_api_key_requests_for_model.labels(*labels)._value.get()
        tok = logger.litellm_remaining_api_key_tokens_for_model.labels(*labels)._value.get()
        return req, tok

    def test_falls_back_to_v3_additional_headers(self, prometheus_logger):
        """When metadata lacks legacy keys but v3 headers are set, use them."""
        metadata = {"model_group": "gpt-4o"}  # no litellm-key-remaining-*
        headers = {
            "x-ratelimit-model_per_key-remaining-tokens": 1234,
            "x-ratelimit-model_per_key-remaining-requests": 7,
        }
        req, tok = self._call(
            prometheus_logger, metadata=metadata, additional_headers=headers
        )
        assert req == 7
        assert tok == 1234
        assert req < 1e18 and tok < 1e18, "must not be sys.maxsize"

    def test_metadata_keys_take_precedence(self, prometheus_logger):
        """If legacy metadata keys are present, prefer them (back-compat)."""
        metadata = {
            "model_group": "gpt-4o",
            "litellm-key-remaining-requests-gpt-4o": 11,
            "litellm-key-remaining-tokens-gpt-4o": 2222,
        }
        headers = {
            "x-ratelimit-model_per_key-remaining-tokens": 99,
            "x-ratelimit-model_per_key-remaining-requests": 99,
        }
        req, tok = self._call(
            prometheus_logger, metadata=metadata, additional_headers=headers
        )
        assert req == 11
        assert tok == 2222

    def test_no_signals_falls_back_to_maxsize(self, prometheus_logger):
        """Preserve old behaviour: when nothing is set, default to sys.maxsize."""
        req, tok = self._call(
            prometheus_logger, metadata={"model_group": "gpt-4o"}, additional_headers={}
        )
        assert req == float(sys.maxsize)
        assert tok == float(sys.maxsize)
