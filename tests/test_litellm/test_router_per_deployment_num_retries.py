"""
Unit tests for per-deployment num_retries in litellm_params
GitHub Issue: #18968 - Per-deployment max_retries/num_retries in litellm_params is not used in retry logic
"""

import httpx
import pytest
from unittest.mock import patch

import litellm
from litellm import Router
from litellm.types.router import RetryPolicy


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


class TestNoProviderRetryAmplification:
    """
    A routed request must reach the upstream provider exactly ``1 + <router retries>``
    times. The Router is the sole retry owner for routed calls, so the provider SDK
    must never retry on top of it. Otherwise a per-deployment ``num_retries`` set in
    ``litellm_params`` is applied twice - once by the Router loop and once as the
    provider client's ``max_retries`` - turning one request into ``(1 + num_retries) ** 2``
    upstream requests.

    These tests count actual upstream HTTP requests through the full Router completion
    path by injecting a counting transport via ``litellm.aclient_session`` (the
    documented seam the OpenAI client builder reads), so both Router-level and any
    provider-SDK-level retries are observed.
    """

    @staticmethod
    def _install_counting_upstream() -> dict:
        """Route every upstream POST to a 500 and count it. ``retry-after: 0`` keeps
        provider-SDK backoff at zero so a mutated (double-retrying) build stays fast."""
        counter = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            counter["n"] += 1
            return httpx.Response(
                500,
                headers={"retry-after": "0"},
                json={"error": {"message": "boom", "type": "server_error"}},
            )

        litellm.aclient_session = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return counter

    @pytest.fixture(autouse=True)
    def _isolate_clients(self):
        litellm.in_memory_llm_clients_cache.flush_cache()
        yield
        litellm.aclient_session = None
        litellm.in_memory_llm_clients_cache.flush_cache()

    @staticmethod
    def _router(api_base: str, litellm_params: dict, **router_kwargs) -> Router:
        params = {"model": "openai/gpt-4o-mini", "api_base": api_base, "api_key": "sk-fake"}
        params.update(litellm_params)
        return Router(model_list=[{"model_name": "mock", "litellm_params": params}], **router_kwargs)

    async def _call_and_count(self, router: Router, **call_kwargs) -> int:
        counter = self._install_counting_upstream()
        with patch("asyncio.sleep", return_value=None):
            with pytest.raises(litellm.InternalServerError):
                await router.acompletion(
                    model="mock", messages=[{"role": "user", "content": "hi"}], **call_kwargs
                )
        return counter["n"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("num_retries", [2, 5])
    async def test_deployment_num_retries_sends_no_extra_provider_requests(self, num_retries):
        """
        Deployment ``num_retries=N`` (every attempt failing) must send exactly ``N + 1``
        upstream requests, not ``(N + 1) ** 2``. This is the amplification regression:
        an unfixed build sends 9 (N=2) or 36 (N=5).
        """
        counter = self._install_counting_upstream()
        router = self._router(
            f"https://amp-{num_retries}.local/v1", {"num_retries": num_retries}, num_retries=1
        )
        with patch("asyncio.sleep", return_value=None):
            with pytest.raises(litellm.InternalServerError):
                await router.acompletion(model="mock", messages=[{"role": "user", "content": "hi"}])
        assert counter["n"] == num_retries + 1

    @pytest.mark.asyncio
    async def test_request_max_retries_does_not_nest_with_router_retries(self):
        """
        A request-body ``max_retries`` must not make the provider SDK retry on top of the
        Router. With deployment ``num_retries=5`` and request ``max_retries=3`` the count
        stays ``6``; a build that lets either value reach the provider SDK sends 24 or 36.
        """
        router = self._router("https://nest-req.local/v1", {"num_retries": 5}, num_retries=1)
        assert await self._call_and_count(router, max_retries=3) == 6

    @pytest.mark.asyncio
    async def test_deployment_max_retries_does_not_nest_with_router_retries(self):
        """
        A deployment-level ``max_retries`` is likewise never applied on top of the Router's
        retries for a routed call: deployment ``num_retries=5`` plus ``max_retries=3`` still
        sends exactly ``6`` upstream requests.
        """
        router = self._router(
            "https://nest-dep.local/v1", {"num_retries": 5, "max_retries": 3}, num_retries=1
        )
        assert await self._call_and_count(router) == 6

    @pytest.mark.asyncio
    async def test_retry_policy_configured_does_not_reintroduce_amplification(self):
        """
        With a retry policy configured alongside a per-deployment ``num_retries=5``, the
        provider SDK still must not retry: exactly ``6`` upstream requests, not 36.
        """
        router = self._router(
            "https://policy.local/v1",
            {"num_retries": 5},
            num_retries=1,
            retry_policy=RetryPolicy(InternalServerErrorRetries=2),
        )
        assert await self._call_and_count(router) == 6

    @pytest.mark.asyncio
    async def test_global_num_retries_not_amplified(self):
        """
        Global ``num_retries`` (no per-deployment setting) already behaves correctly and
        must stay that way: ``num_retries=3`` sends ``4`` upstream requests.
        """
        router = self._router("https://global.local/v1", {}, num_retries=3)
        assert await self._call_and_count(router) == 4

    @pytest.mark.asyncio
    async def test_direct_completion_still_forwards_num_retries_to_provider(self):
        """
        For a NON-routed direct ``litellm.acompletion`` call, ``num_retries`` remains an
        alias for the provider client's ``max_retries`` (the instructor use case). The
        provider SDK therefore retries in addition to litellm's own retry wrapper, so the
        upstream count exceeds ``num_retries + 1`` - proving the routed-call fix did not
        change direct-call behaviour.
        """
        counter = self._install_counting_upstream()
        num_retries = 2
        with patch("asyncio.sleep", return_value=None):
            with pytest.raises(litellm.InternalServerError):
                await litellm.acompletion(
                    model="openai/gpt-4o-mini",
                    api_base="https://direct.local/v1",
                    api_key="sk-fake",
                    messages=[{"role": "user", "content": "hi"}],
                    num_retries=num_retries,
                )
        assert counter["n"] > num_retries + 1
