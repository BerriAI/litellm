import json
from unittest.mock import MagicMock, patch

import pytest

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


def test_trigger_cooldown_calls_set_cooldown_when_deployment_id_present():
    router = MagicMock()
    router.cooldown_time = 60
    router.get_model_info.return_value = None

    exc = RuntimeError("upstream error")
    exc.status_code = 429

    kwargs = {"litellm_metadata": {"model_info": {"id": "deployment-abc"}}}

    with patch(
        "litellm.router_utils.fallback_event_handlers._set_cooldown_deployments"
    ) as mock_set:
        _trigger_cooldown_for_failed_deployment(
            litellm_router=router, kwargs=kwargs, exception=exc
        )

    mock_set.assert_called_once()
    _, call_kwargs = mock_set.call_args
    assert call_kwargs["deployment"] == "deployment-abc"
    assert call_kwargs["exception_status"] == 429


def test_trigger_cooldown_skips_when_no_deployment_id():
    router = MagicMock()
    kwargs = {"litellm_metadata": {}}

    with patch(
        "litellm.router_utils.fallback_event_handlers._set_cooldown_deployments"
    ) as mock_set:
        _trigger_cooldown_for_failed_deployment(
            litellm_router=router, kwargs=kwargs, exception=RuntimeError("err")
        )

    mock_set.assert_not_called()


def test_trigger_cooldown_uses_deployment_cooldown_time_when_present():
    router = MagicMock()
    router.cooldown_time = 60
    router.get_model_info.return_value = {"litellm_params": {"cooldown_time": 30}}

    exc = RuntimeError("upstream error")
    exc.status_code = 429

    kwargs = {"litellm_metadata": {"model_info": {"id": "deployment-abc"}}}

    with patch(
        "litellm.router_utils.fallback_event_handlers._set_cooldown_deployments"
    ) as mock_set:
        _trigger_cooldown_for_failed_deployment(
            litellm_router=router, kwargs=kwargs, exception=exc
        )

    _, call_kwargs = mock_set.call_args
    assert call_kwargs["time_to_cooldown"] == 30


def test_trigger_cooldown_silently_catches_exceptions():
    router = MagicMock()
    router.cooldown_time = 60
    router.get_model_info.return_value = None

    exc = RuntimeError("upstream error")
    kwargs = {"litellm_metadata": {"model_info": {"id": "deployment-abc"}}}

    with patch(
        "litellm.router_utils.fallback_event_handlers._set_cooldown_deployments",
        side_effect=RuntimeError("cooldown error"),
    ):
        _trigger_cooldown_for_failed_deployment(
            litellm_router=router, kwargs=kwargs, exception=exc
        )


def test_trigger_cooldown_uses_litellm_metadata_only_not_metadata():
    router = MagicMock()
    router.cooldown_time = 60
    router.get_model_info.return_value = None

    exc = RuntimeError("err")
    # metadata (user-supplied) should be ignored; litellm_metadata is absent -> no cooldown
    kwargs = {"metadata": {"model_info": {"id": "injected-id"}}}

    with patch(
        "litellm.router_utils.fallback_event_handlers._set_cooldown_deployments"
    ) as mock_set:
        _trigger_cooldown_for_failed_deployment(
            litellm_router=router, kwargs=kwargs, exception=exc
        )

    mock_set.assert_not_called()


@pytest.mark.asyncio
async def test_run_async_fallback_triggers_cooldown_when_logging_obj_has_logged():
    router = MagicMock()
    router.cooldown_time = 60
    router.get_model_info.return_value = None
    router.log_retry = MagicMock(side_effect=lambda kwargs, e: kwargs)

    exc = RuntimeError("fallback failed")

    async def _always_fail(*args, **kwargs):
        raise exc

    router.async_function_with_fallbacks = _always_fail

    logging_obj = MagicMock()
    logging_obj.has_logged_async_failure = True

    kwargs = {
        "litellm_metadata": {"model_info": {"id": "dep-xyz"}},
        "litellm_logging_obj": logging_obj,
    }

    with patch(
        "litellm.router_utils.fallback_event_handlers._set_cooldown_deployments"
    ) as mock_set:
        with pytest.raises(RuntimeError):
            await run_async_fallback(
                litellm_router=router,
                fallback_model_group=["fallback-model"],
                original_model_group="primary-model",
                original_exception=RuntimeError("original"),
                max_fallbacks=3,
                fallback_depth=0,
                **kwargs,
            )

    mock_set.assert_called_once()


@pytest.mark.asyncio
async def test_run_async_fallback_skips_cooldown_when_logging_obj_not_logged():
    router = MagicMock()
    router.log_retry = MagicMock(side_effect=lambda kwargs, e: kwargs)

    exc = RuntimeError("fallback failed")

    async def _always_fail(*args, **kwargs):
        raise exc

    router.async_function_with_fallbacks = _always_fail

    logging_obj = MagicMock()
    logging_obj.has_logged_async_failure = False

    kwargs = {
        "litellm_metadata": {"model_info": {"id": "dep-xyz"}},
        "litellm_logging_obj": logging_obj,
    }

    with patch(
        "litellm.router_utils.fallback_event_handlers._set_cooldown_deployments"
    ) as mock_set:
        with pytest.raises(RuntimeError):
            await run_async_fallback(
                litellm_router=router,
                fallback_model_group=["fallback-model"],
                original_model_group="primary-model",
                original_exception=RuntimeError("original"),
                max_fallbacks=3,
                fallback_depth=0,
                **kwargs,
            )

    mock_set.assert_not_called()


def test_get_fallback_model_group_does_not_mutate_fallbacks():
    """A string fallback must be resolved without mutating the caller's
    fallbacks list, which is the live router config shared across requests."""
    fallbacks = [{"gpt-3.5-turbo": ["claude-3-haiku"]}, "gpt-4o-mini"]

    fallback_model_group, _ = get_fallback_model_group(
        fallbacks=fallbacks, model_group="unmatched-model"
    )

    assert fallback_model_group == ["gpt-4o-mini"]
    assert fallbacks == [{"gpt-3.5-turbo": ["claude-3-haiku"]}, "gpt-4o-mini"]
