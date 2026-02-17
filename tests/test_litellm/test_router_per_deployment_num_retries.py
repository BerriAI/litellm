"""
Unit tests for per-deployment num_retries in litellm_params
GitHub Issue: #18968 - Per-deployment max_retries/num_retries in litellm_params is not used in retry logic
"""

import pytest
from unittest.mock import MagicMock, patch

from litellm import Router


class TestPerDeploymentNumRetries:
    """Test that per-deployment num_retries in litellm_params is correctly used."""

    def test_set_deployment_num_retries_on_exception(self):
        """
        Test that _set_deployment_num_retries_on_exception sets num_retries
        on the exception from the deployment's litellm_params.
        """
        router = Router(
            model_list=[
                {
                    "model_name": "test-model",
                    "litellm_params": {
                        "model": "openai/gpt-4",
                        "api_key": "test-key",
                        "num_retries": 5,  # Per-deployment setting
                    },
                },
            ],
            num_retries=1,  # Global setting
        )

        deployment = router.model_list[0]

        # Create a mock exception without num_retries
        class MockException(Exception):
            pass

        exc = MockException("test error")
        assert not hasattr(exc, "num_retries") or exc.num_retries is None

        # Call the helper
        router._set_deployment_num_retries_on_exception(exc, deployment)

        # Verify num_retries was set from deployment
        assert exc.num_retries == 5

    def test_set_deployment_num_retries_does_not_override_existing(self):
        """
        Test that _set_deployment_num_retries_on_exception does NOT override
        if exception already has num_retries set.
        """
        router = Router(
            model_list=[
                {
                    "model_name": "test-model",
                    "litellm_params": {
                        "model": "openai/gpt-4",
                        "api_key": "test-key",
                        "num_retries": 5,
                    },
                },
            ],
            num_retries=1,
        )

        deployment = router.model_list[0]

        # Create an exception that already has num_retries
        class MockException(Exception):
            num_retries = 10  # Already set

        exc = MockException("test error")

        # Call the helper
        router._set_deployment_num_retries_on_exception(exc, deployment)

        # Verify num_retries was NOT overridden
        assert exc.num_retries == 10

    def test_deployment_without_num_retries(self):
        """
        Test that _set_deployment_num_retries_on_exception does nothing
        if deployment has no num_retries set.
        """
        router = Router(
            model_list=[
                {
                    "model_name": "test-model",
                    "litellm_params": {
                        "model": "openai/gpt-4",
                        "api_key": "test-key",
                        # No num_retries set
                    },
                },
            ],
            num_retries=3,
        )

        deployment = router.model_list[0]

        class MockException(Exception):
            pass

        exc = MockException("test error")

        # Call the helper
        router._set_deployment_num_retries_on_exception(exc, deployment)

        # Verify num_retries was not set (deployment has no num_retries)
        assert not hasattr(exc, "num_retries") or exc.num_retries is None

    def test_request_level_num_retries_takes_precedence(self):
        """
        Test that request-level num_retries (passed in kwargs) is still respected.
        """
        router = Router(
            model_list=[
                {
                    "model_name": "test-model",
                    "litellm_params": {
                        "model": "openai/gpt-4",
                        "api_key": "test-key",
                        "num_retries": 5,
                    },
                },
            ],
            num_retries=1,
        )

        # Pass num_retries in request kwargs - this should take precedence
        kwargs = {"num_retries": 10}
        router._update_kwargs_before_fallbacks(model="test-model", kwargs=kwargs)
        assert kwargs["num_retries"] == 10  # Request-level takes precedence

    def test_global_num_retries_used_when_no_deployment_setting(self):
        """
        Test that global num_retries is used when deployment has no num_retries.
        """
        router = Router(
            model_list=[
                {
                    "model_name": "test-model",
                    "litellm_params": {
                        "model": "openai/gpt-4",
                        "api_key": "test-key",
                        # No num_retries set
                    },
                },
            ],
            num_retries=7,  # Global setting
        )

        kwargs = {}
        router._update_kwargs_before_fallbacks(model="test-model", kwargs=kwargs)
        assert kwargs["num_retries"] == 7  # Uses global

    def test_set_deployment_num_retries_with_string_value(self):
        """
        Test that _set_deployment_num_retries_on_exception handles string values
        from environment variables correctly.
        GitHub Issue: #19481
        """
        router = Router(
            model_list=[
                {
                    "model_name": "test-model",
                    "litellm_params": {
                        "model": "openai/gpt-4",
                        "api_key": "test-key",
                        "num_retries": "6",  # String value (as from env var)
                    },
                },
            ],
            num_retries=0,  # Global setting
        )

        deployment = router.model_list[0]

        class MockException(Exception):
            pass

        exc = MockException("test error")

        # Call the helper
        router._set_deployment_num_retries_on_exception(exc, deployment)

        # Verify num_retries was converted from string to int
        assert exc.num_retries == 6
