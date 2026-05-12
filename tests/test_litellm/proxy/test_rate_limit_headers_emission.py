"""Tests for x-ratelimit-* header emission across response shapes.

Covers the seed sites added in `litellm/proxy/common_request_processing.py`
so that streaming responses, plain-dict responses, and pydantic responses
all surface the rate-limit headers stored on
`self.data["litellm_proxy_rate_limit_response"]`.
"""

import pytest

from litellm.proxy.hooks._rate_limit_headers import (
    apply_rate_limit_statuses_to_headers,
)


def _rate_limit_response_sample() -> dict:
    return {
        "overall_code": "OK",
        "statuses": [
            {
                "descriptor_key": "api_key",
                "rate_limit_type": "requests",
                "current_limit": 6000,
                "limit_remaining": 5997,
            },
            {
                "descriptor_key": "api_key",
                "rate_limit_type": "tokens",
                "current_limit": 100000,
                "limit_remaining": 99955,
            },
        ],
    }


class TestApplyRateLimitStatusesToHeaders:
    def test_populates_descriptor_remaining_and_limit_entries(self):
        headers: dict = {}
        apply_rate_limit_statuses_to_headers(headers, _rate_limit_response_sample())
        assert headers["x-ratelimit-api_key-remaining-requests"] == 5997
        assert headers["x-ratelimit-api_key-limit-requests"] == 6000
        assert headers["x-ratelimit-api_key-remaining-tokens"] == 99955
        assert headers["x-ratelimit-api_key-limit-tokens"] == 100000

    def test_setdefault_does_not_overwrite_existing_keys(self):
        headers = {
            "x-ratelimit-api_key-remaining-requests": "preserved",
        }
        apply_rate_limit_statuses_to_headers(headers, _rate_limit_response_sample())
        # Existing key preserved
        assert headers["x-ratelimit-api_key-remaining-requests"] == "preserved"
        # Other keys still filled in
        assert headers["x-ratelimit-api_key-limit-requests"] == 6000

    def test_noop_for_none(self):
        headers: dict = {}
        apply_rate_limit_statuses_to_headers(headers, None)
        assert headers == {}

    def test_noop_for_empty_dict(self):
        headers: dict = {}
        apply_rate_limit_statuses_to_headers(headers, {})
        assert headers == {}

    def test_noop_for_response_with_empty_statuses(self):
        headers: dict = {}
        apply_rate_limit_statuses_to_headers(
            headers, {"overall_code": "OK", "statuses": []}
        )
        assert headers == {}

    def test_skips_status_entries_missing_required_fields(self):
        headers: dict = {}
        apply_rate_limit_statuses_to_headers(
            headers,
            {
                "statuses": [
                    # missing descriptor_key
                    {
                        "rate_limit_type": "requests",
                        "current_limit": 1,
                        "limit_remaining": 1,
                    },
                    # missing rate_limit_type
                    {
                        "descriptor_key": "api_key",
                        "current_limit": 1,
                        "limit_remaining": 1,
                    },
                    # valid one to confirm iteration continues
                    {
                        "descriptor_key": "team",
                        "rate_limit_type": "requests",
                        "current_limit": 10,
                        "limit_remaining": 9,
                    },
                ]
            },
        )
        assert headers == {
            "x-ratelimit-team-remaining-requests": 9,
            "x-ratelimit-team-limit-requests": 10,
        }
