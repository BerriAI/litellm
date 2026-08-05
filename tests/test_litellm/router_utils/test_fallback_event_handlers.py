import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.router_utils.cooldown_handlers import mark_advisor_orchestration_failure
from litellm.router_utils.fallback_event_handlers import (
    _trigger_cooldown_for_failed_deployment,
    get_fallback_model_group,
    run_async_fallback,
)


class StreamingWrapper:
    def __init__(self):
        self._hidden_params = {"additional_headers": {}}


class FakeRouter:
    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        return StreamingWrapper()


class AlwaysFailRouter:
    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        raise RuntimeError("fallback model also failed")


@pytest.mark.asyncio
async def test_run_async_fallback_adds_errors_when_opted_in():
    response = await run_async_fallback(
        litellm_router=FakeRouter(),
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
        include_fallback_errors=True,
    )

    additional_headers = response._hidden_params["additional_headers"]
    assert additional_headers["x-litellm-attempted-fallbacks"] == 1
    assert json.loads(additional_headers["x-litellm-fallback-errors"]) == [
        {
            "message": "upstream limited request",
            "type": "RuntimeError",
            "param": None,
            "code": None,
        }
    ]


@pytest.mark.asyncio
async def test_run_async_fallback_omits_errors_without_opt_in():
    response = await run_async_fallback(
        litellm_router=FakeRouter(),
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    additional_headers = response._hidden_params["additional_headers"]
    assert additional_headers["x-litellm-attempted-fallbacks"] == 1
    assert "x-litellm-fallback-errors" not in additional_headers


@pytest.mark.asyncio
async def test_run_async_fallback_raises_when_all_fallbacks_fail():
    with pytest.raises(RuntimeError, match="fallback model also failed"):
        await run_async_fallback(
            litellm_router=AlwaysFailRouter(),
            fallback_model_group=["fallback-model"],
            original_model_group="primary-model",
            original_exception=RuntimeError("original request failed"),
            max_fallbacks=3,
            fallback_depth=0,
            include_fallback_errors=True,
        )


class RecordingRouter:
    def __init__(self):
        self.received_kwargs = None

    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        self.received_kwargs = kwargs
        return StreamingWrapper()


@pytest.mark.asyncio
async def test_run_async_fallback_forwards_include_fallback_errors_to_nested_call():
    """A nested fallback (multi-hop) must keep collecting errors, so the opt-in
    flag has to reach the nested async_function_with_fallbacks call."""
    router = RecordingRouter()
    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
        include_fallback_errors=True,
    )

    assert router.received_kwargs.get("include_fallback_errors") is True


@pytest.mark.asyncio
async def test_run_async_fallback_does_not_forward_flag_without_opt_in():
    router = RecordingRouter()
    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    assert "include_fallback_errors" not in router.received_kwargs


@pytest.mark.asyncio
async def test_run_async_fallback_skips_original_model_group():
    response = await run_async_fallback(
        litellm_router=FakeRouter(),
        fallback_model_group=["primary-model", "fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("original failed"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    assert response._hidden_params["additional_headers"]["x-litellm-attempted-fallbacks"] == 1


def test_get_fallback_model_group_does_not_mutate_fallbacks():
    """A string fallback must be resolved without mutating the caller's
    fallbacks list, which is the live router config shared across requests."""
    fallbacks = [{"gpt-3.5-turbo": ["claude-3-haiku"]}, "gpt-4o-mini"]

    fallback_model_group, _ = get_fallback_model_group(fallbacks=fallbacks, model_group="unmatched-model")

    assert fallback_model_group == ["gpt-4o-mini"]
    assert fallbacks == [{"gpt-3.5-turbo": ["claude-3-haiku"]}, "gpt-4o-mini"]


class TestTriggerCooldownForFailedDeployment:
    def test_calls_set_cooldown_deployments_with_stamped_deployment_id(self):
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        exc.failed_deployment_id = "fallback-deployment"

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(litellm_router=mock_router, kwargs={}, exception=exc)

            mock_set_cooldown.assert_called_once()
            call_kwargs = mock_set_cooldown.call_args[1]
            assert call_kwargs["deployment"] == "fallback-deployment"
            assert call_kwargs["original_exception"] is exc

    def test_does_not_trust_caller_supplied_metadata_bucket(self):
        """A metadata bucket can't reliably be told apart from a caller-supplied
        one without knowing this call's function_name, so a client with
        permission to set metadata must not be able to get an arbitrary
        deployment cooled down by forging a deployment_model_name marker."""
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        kwargs = {
            "metadata": {
                "model_info": {"id": "attacker-chosen-deployment"},
                "deployment_model_name": "gpt-4",
            }
        }

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(litellm_router=mock_router, kwargs=kwargs, exception=exc)

            mock_set_cooldown.assert_not_called()

    def test_increments_failure_counter_before_cooldown_check(self):
        """The fallback path must feed the same per-minute failure counter the
        primary path uses, or repeated fallback failures never accumulate
        toward the default percent-fail-rate cooldown threshold."""
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        exc.failed_deployment_id = "fallback-deployment"

        with (
            patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown,
            patch(
                "litellm.router_utils.fallback_event_handlers.increment_deployment_failures_for_current_minute"
            ) as mock_increment,
        ):
            _trigger_cooldown_for_failed_deployment(litellm_router=mock_router, kwargs={}, exception=exc)

            mock_increment.assert_called_once_with(
                litellm_router_instance=mock_router, deployment_id="fallback-deployment"
            )
            mock_set_cooldown.assert_called_once()

    def test_no_op_when_deployment_id_missing(self):
        mock_router = MagicMock()

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(
                litellm_router=mock_router, kwargs={}, exception=RuntimeError("no metadata")
            )

            mock_set_cooldown.assert_not_called()

    def test_skipped_for_advisor_orchestration_failure(self):
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        exc.failed_deployment_id = "fallback-deployment"
        mark_advisor_orchestration_failure(exc)

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(litellm_router=mock_router, kwargs={}, exception=exc)

            mock_set_cooldown.assert_not_called()

    def test_uses_deployment_litellm_params_cooldown_time_override(self):
        mock_router = MagicMock()
        mock_router.cooldown_time = 300.0
        mock_router.get_model_info.return_value = {"litellm_params": {"cooldown_time": 30.0}}

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        exc.failed_deployment_id = "fallback-deployment"

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(litellm_router=mock_router, kwargs={}, exception=exc)

            call_kwargs = mock_set_cooldown.call_args[1]
            assert call_kwargs["time_to_cooldown"] == 30.0

    def test_uses_response_header_when_no_deployment_config(self):
        """Precedence must match Router.deployment_callback_on_failure's primary
        path: deployment config, then the response's Retry-After header, then the
        router default."""
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = {"litellm_params": {}}

        exc = RuntimeError("upstream error")
        exc.failed_deployment_id = "fallback-deployment"
        exc.litellm_response_headers = httpx.Headers({"retry-after": "45"})

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(litellm_router=mock_router, kwargs={}, exception=exc)

            call_kwargs = mock_set_cooldown.call_args[1]
            assert call_kwargs["time_to_cooldown"] == 45

    def test_silently_catches_exceptions(self):
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = RuntimeError("upstream error")
        exc.failed_deployment_id = "fallback-deployment"

        with patch(
            "litellm.router_utils.fallback_event_handlers._set_cooldown_deployments",
            side_effect=RuntimeError("cooldown error"),
        ):
            _trigger_cooldown_for_failed_deployment(litellm_router=mock_router, kwargs={}, exception=exc)

    def test_skips_request_scoped_404_on_generic_api_call(self):
        """A generic API call (files/batches/threads/rerank/...) forwards a caller-supplied
        resource id, so a 404 there means "that id doesn't exist", not "this deployment is
        unhealthy". Without this guard, a single bad id would 404 every deployment in the
        fallback chain and cool all of them down from one request."""
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.NotFoundError("not found", "openai", "gpt-4")
        exc.failed_deployment_id = "fallback-deployment"

        with (
            patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown,
            patch(
                "litellm.router_utils.fallback_event_handlers.increment_deployment_failures_for_current_minute"
            ) as mock_increment,
        ):
            _trigger_cooldown_for_failed_deployment(
                litellm_router=mock_router,
                kwargs={"original_generic_function": MagicMock()},
                exception=exc,
            )

            mock_set_cooldown.assert_not_called()
            mock_increment.assert_not_called()

    def test_still_cools_down_404_outside_generic_api_call(self):
        """The request-scoped-404 guard is scoped to generic API calls only: a 404 on a
        regular completion fallback (no original_generic_function in kwargs) must still
        cool down the deployment as before."""
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.NotFoundError("not found", "openai", "gpt-4")
        exc.failed_deployment_id = "fallback-deployment"

        with patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(litellm_router=mock_router, kwargs={}, exception=exc)

            mock_set_cooldown.assert_called_once()


class TestRunAsyncFallbackTriggersCooldown:
    class RouterWithLoggingKwarg:
        def __init__(self):
            self.cooldown_time = 60.0

        def log_retry(self, kwargs, e):
            return kwargs

        def get_model_info(self, id):
            return None

        async def async_function_with_fallbacks(self, *args, **kwargs):
            raise RuntimeError("fallback model also failed")

    def _logging_obj(self, has_logged_async_failure: bool) -> MagicMock:
        logging_obj = MagicMock()
        logging_obj.model_call_details = {"has_logged_async_failure": has_logged_async_failure}
        return logging_obj

    @pytest.mark.asyncio
    async def test_triggers_cooldown_when_has_logged_async_failure_is_true(self):
        with patch(
            "litellm.router_utils.fallback_event_handlers._trigger_cooldown_for_failed_deployment"
        ) as mock_trigger:
            with pytest.raises(RuntimeError, match="fallback model also failed"):
                await run_async_fallback(
                    litellm_router=self.RouterWithLoggingKwarg(),
                    fallback_model_group=["fallback-model"],
                    original_model_group="primary-model",
                    original_exception=RuntimeError("original request failed"),
                    max_fallbacks=3,
                    fallback_depth=0,
                    litellm_logging_obj=self._logging_obj(has_logged_async_failure=True),
                )

            mock_trigger.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_trigger_cooldown_when_has_logged_async_failure_is_false(self):
        """This is the exact dead-code scenario the bug fix addresses: before it,
        the normal failure callback runs for the first attempt in a fallback chain
        (has_logged_async_failure is still False at that point), so no explicit
        trigger is needed there."""
        with patch(
            "litellm.router_utils.fallback_event_handlers._trigger_cooldown_for_failed_deployment"
        ) as mock_trigger:
            with pytest.raises(RuntimeError, match="fallback model also failed"):
                await run_async_fallback(
                    litellm_router=self.RouterWithLoggingKwarg(),
                    fallback_model_group=["fallback-model"],
                    original_model_group="primary-model",
                    original_exception=RuntimeError("original request failed"),
                    max_fallbacks=3,
                    fallback_depth=0,
                    litellm_logging_obj=self._logging_obj(has_logged_async_failure=False),
                )

            mock_trigger.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_trigger_cooldown_when_no_logging_obj_present(self):
        with patch(
            "litellm.router_utils.fallback_event_handlers._trigger_cooldown_for_failed_deployment"
        ) as mock_trigger:
            with pytest.raises(RuntimeError, match="fallback model also failed"):
                await run_async_fallback(
                    litellm_router=self.RouterWithLoggingKwarg(),
                    fallback_model_group=["fallback-model"],
                    original_model_group="primary-model",
                    original_exception=RuntimeError("original request failed"),
                    max_fallbacks=3,
                    fallback_depth=0,
                )

            mock_trigger.assert_not_called()
