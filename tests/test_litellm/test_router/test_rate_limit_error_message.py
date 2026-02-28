"""
Tests for #20867: Rate limit errors should report a clear rate-limit
message instead of the misleading "No deployments available for selected
model", and should carry status_code=429.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.types.router import (
    RouterErrors,
    RouterRateLimitError,
    RouterRateLimitErrorBasic,
)


class TestRouterRateLimitErrorBasic:
    """Tests for RouterRateLimitErrorBasic error class."""

    def test_message_indicates_rate_limit(self):
        """The error message should clearly mention rate limiting, not
        'No deployments available'."""
        exc = RouterRateLimitErrorBasic(model="mine")
        msg = str(exc)
        assert RouterErrors.user_defined_ratelimit_error.value in msg
        assert "RPM limit" in msg
        assert "mine" in msg

    def test_message_does_not_say_no_deployments(self):
        """Regression: the old message 'No deployments available for
        selected model' was misleading for rate-limit errors (#20867)."""
        exc = RouterRateLimitErrorBasic(model="mine")
        assert RouterErrors.no_deployments_available.value not in str(exc)

    def test_has_status_code_429(self):
        """The exception must carry status_code=429 so the proxy surfaces
        the correct HTTP status without string-matching heuristics."""
        exc = RouterRateLimitErrorBasic(model="gpt-4")
        assert exc.status_code == 429

    def test_model_attribute_preserved(self):
        """The model name should be stored for debugging."""
        exc = RouterRateLimitErrorBasic(model="openai/gpt-5.2")
        assert exc.model == "openai/gpt-5.2"


class TestRouterRateLimitError:
    """Ensure the full RouterRateLimitError still works as before."""

    def test_message_contains_cooldown_info(self):
        exc = RouterRateLimitError(
            model="gpt-4",
            cooldown_time=30.0,
            enable_pre_call_checks=True,
            cooldown_list=["deployment-1"],
        )
        msg = str(exc)
        assert "30" in msg
        assert "gpt-4" in msg


class TestProxyExceptionRateLimitCode:
    """The ProxyException constructor should map rate-limit messages to
    code '429'."""

    def test_user_defined_ratelimit_message_maps_to_429(self):
        from litellm.proxy._types import ProxyException

        exc = ProxyException(
            message=(
                f"{RouterErrors.user_defined_ratelimit_error.value} "
                "Passed model=mine. All deployments for this model are at "
                "their RPM limit."
            ),
            type="None",
            param="None",
            code=500,  # default before message check
        )
        assert exc.code == "429"

    def test_no_deployments_message_still_maps_to_429(self):
        """Backward compatibility: old-style messages should still resolve
        to 429."""
        from litellm.proxy._types import ProxyException

        exc = ProxyException(
            message="No deployments available for selected model.",
            type="None",
            param="None",
            code=500,
        )
        assert exc.code == "429"
