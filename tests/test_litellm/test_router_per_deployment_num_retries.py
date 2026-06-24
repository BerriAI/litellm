"""
Unit tests for per-deployment num_retries in litellm_params
GitHub Issue: #18968 - Per-deployment max_retries/num_retries in litellm_params is not used in retry logic
"""

import pytest
from unittest.mock import patch

import litellm
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


class TestNumRetriesNoneGuard:
    """
    Regression tests for the num_retries=None TypeError in async_function_with_retries.

    When num_retries reaches async_function_with_retries as None - e.g. a caller passes
    num_retries=None explicitly (dict.get() does not fall back on an existing None value),
    an auto_router/complexity_router path does not propagate it, or
    Router.update_settings(num_retries=None) is used - AND the underlying call fails with a
    retryable error, the comparison `if num_retries > 0:` raised:

        TypeError: '>' not supported between instances of 'NoneType' and 'int'

    This masked the real upstream error (rate limit / connection / 5xx) behind a TypeError.
    Related issues: #23316, #25889, #23699, #28126.
    """

    @staticmethod
    def _mock_router(num_retries=2):
        return Router(
            model_list=[
                {
                    "model_name": "mock-model",
                    "litellm_params": {
                        "model": "gpt-4o-mini",
                        "mock_response": "ok",
                    },
                }
            ],
            num_retries=num_retries,
        )

    def test_update_kwargs_normalises_explicit_none_to_router_default(self):
        """
        _update_kwargs_before_fallbacks must normalise an explicit num_retries=None to
        the router default (not leave it as None), while preserving an explicit 0.
        """
        router = self._mock_router(num_retries=4)

        # explicit None -> router default
        kwargs = {"num_retries": None}
        router._update_kwargs_before_fallbacks(model="mock-model", kwargs=kwargs)
        assert kwargs["num_retries"] == 4

        # explicit 0 is preserved (retries stay disabled)
        kwargs = {"num_retries": 0}
        router._update_kwargs_before_fallbacks(model="mock-model", kwargs=kwargs)
        assert kwargs["num_retries"] == 0

        # absent -> router default (unchanged behaviour)
        kwargs = {}
        router._update_kwargs_before_fallbacks(model="mock-model", kwargs=kwargs)
        assert kwargs["num_retries"] == 4

        # explicit None with router default also None -> 0 (mirrors the downstream guard)
        router.num_retries = None  # simulate update_settings(num_retries=None) (#28126)
        kwargs = {"num_retries": None}
        router._update_kwargs_before_fallbacks(model="mock-model", kwargs=kwargs)
        assert kwargs["num_retries"] == 0

    @pytest.mark.asyncio
    async def test_acompletion_num_retries_none_does_not_raise_typeerror(self):
        """
        Per-request num_retries=None + a retryable error must NOT raise TypeError.
        The router falls back to its configured num_retries and retries the (transient)
        error, so the request succeeds.
        """
        router = self._mock_router(num_retries=2)
        with patch("asyncio.sleep", return_value=None):
            response = await router.acompletion(
                model="mock-model",
                messages=[{"role": "user", "content": "hi"}],
                num_retries=None,  # the trigger
                mock_testing_rate_limit_error=True,  # retryable error path
            )
        assert response.choices[0].message.content == "ok"

    @pytest.mark.asyncio
    async def test_async_function_with_retries_none_falls_back_to_zero(self):
        """
        When both the per-request value AND the router-level setting are None
        (e.g. after Router.update_settings(num_retries=None), #28126), num_retries must
        fall back to 0 and the real retryable error must surface - not a TypeError.
        """
        router = self._mock_router(num_retries=0)
        router.num_retries = None  # simulate update_settings(num_retries=None)

        async def failing_fn(*args, **kwargs):
            raise litellm.RateLimitError(
                message="boom", model="mock-model", llm_provider="openai"
            )

        with patch("asyncio.sleep", return_value=None):
            with pytest.raises(litellm.RateLimitError):
                await router.async_function_with_retries(
                    original_function=failing_fn,
                    model="mock-model",
                    messages=[{"role": "user", "content": "hi"}],
                    num_retries=None,
                )

    @pytest.mark.asyncio
    async def test_async_function_with_retries_none_falls_back_to_router_default(self):
        """
        A None per-request num_retries falls back to the router-level setting, so retries
        still happen (original_function is invoked more than once) before the real error
        is raised - proving None did not silently disable retries or crash.
        """
        router = self._mock_router(num_retries=3)
        calls = {"n": 0}

        async def failing_fn(*args, **kwargs):
            calls["n"] += 1
            raise litellm.InternalServerError(
                message="boom", model="mock-model", llm_provider="openai"
            )

        with patch("asyncio.sleep", return_value=None):
            with pytest.raises(litellm.InternalServerError):
                await router.async_function_with_retries(
                    original_function=failing_fn,
                    model="mock-model",
                    messages=[{"role": "user", "content": "hi"}],
                    metadata={},  # populated by acompletion in the real path; log_retry needs it
                    num_retries=None,
                )

        # 1 initial attempt + at least 1 retry -> proves None fell back to a positive int
        assert calls["n"] >= 2
