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

    kwargs = {
        "litellm_metadata": {"model_info": {"id": "deployment-abc"}, "deployment_model_name": "gpt-4"}
    }

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

    kwargs = {
        "litellm_metadata": {"model_info": {"id": "deployment-abc"}, "deployment_model_name": "gpt-4"}
    }

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
    kwargs = {
        "litellm_metadata": {"model_info": {"id": "deployment-abc"}, "deployment_model_name": "gpt-4"}
    }

    with patch(
        "litellm.router_utils.fallback_event_handlers._set_cooldown_deployments",
        side_effect=RuntimeError("cooldown error"),
    ):
        _trigger_cooldown_for_failed_deployment(
            litellm_router=router, kwargs=kwargs, exception=exc
        )


def test_trigger_cooldown_uses_metadata_when_litellm_metadata_absent():
    """Router._update_kwargs_with_deployment() overwrites "model_info" (plus the
    sibling "deployment_model_name" key) on whichever of "metadata"/"litellm_metadata"
    the current call uses (regular completions use plain "metadata"; batch/thread/file
    endpoints use "litellm_metadata"), so either bucket is trusted as long as it
    carries that sibling key."""
    router = MagicMock()
    router.cooldown_time = 60
    router.get_model_info.return_value = None

    exc = RuntimeError("err")
    kwargs = {"metadata": {"model_info": {"id": "deployment-abc"}, "deployment_model_name": "gpt-4"}}

    with patch(
        "litellm.router_utils.fallback_event_handlers._set_cooldown_deployments"
    ) as mock_set:
        _trigger_cooldown_for_failed_deployment(
            litellm_router=router, kwargs=kwargs, exception=exc
        )

    mock_set.assert_called_once()
    _, call_kwargs = mock_set.call_args
    assert call_kwargs["deployment"] == "deployment-abc"


def test_trigger_cooldown_ignores_metadata_bucket_missing_router_marker():
    """A bucket with "model_info" but no "deployment_model_name" was never written by
    Router._update_kwargs_with_deployment(), so it must not be trusted even if it's
    the only bucket present. This is the caller-supplied-metadata case a request-body
    field could reach without the router ever touching it."""
    router = MagicMock()
    kwargs = {"metadata": {"model_info": {"id": "untrusted-id"}}}

    with patch(
        "litellm.router_utils.fallback_event_handlers._set_cooldown_deployments"
    ) as mock_set:
        _trigger_cooldown_for_failed_deployment(
            litellm_router=router, kwargs=kwargs, exception=RuntimeError("err")
        )

    mock_set.assert_not_called()


def test_trigger_cooldown_ignores_poisoned_litellm_metadata_when_metadata_is_authoritative():
    """Adversarial case: a caller with allow_client_pricing_override permission
    preserves a poisoned litellm_metadata.model_info.id naming another deployment,
    while the router (using plain "metadata" for this regular completion call, as it
    always does) correctly overwrote "metadata" with the real failed deployment's
    info. The poisoned litellm_metadata bucket must be ignored because it lacks the
    router's own "deployment_model_name" marker."""
    router = MagicMock()
    router.cooldown_time = 60
    router.get_model_info.return_value = None

    exc = RuntimeError("err")
    kwargs = {
        "litellm_metadata": {"model_info": {"id": "victim-deployment"}},
        "metadata": {"model_info": {"id": "real-failed-deployment"}, "deployment_model_name": "gpt-4"},
    }

    with patch(
        "litellm.router_utils.fallback_event_handlers._set_cooldown_deployments"
    ) as mock_set:
        _trigger_cooldown_for_failed_deployment(
            litellm_router=router, kwargs=kwargs, exception=exc
        )

    _, call_kwargs = mock_set.call_args
    assert call_kwargs["deployment"] == "real-failed-deployment"


def test_trigger_cooldown_uses_litellm_metadata_when_it_is_the_authoritative_bucket():
    """For batch/thread/file-style calls the router writes into litellm_metadata
    instead of metadata; that bucket must be used even if a stale/caller-supplied
    "metadata" bucket (lacking the router's marker) also happens to be present."""
    router = MagicMock()
    router.cooldown_time = 60
    router.get_model_info.return_value = None

    exc = RuntimeError("err")
    kwargs = {
        "litellm_metadata": {"model_info": {"id": "batch-deployment"}, "deployment_model_name": "gpt-4"},
        "metadata": {"model_info": {"id": "stale-id"}},
    }

    with patch(
        "litellm.router_utils.fallback_event_handlers._set_cooldown_deployments"
    ) as mock_set:
        _trigger_cooldown_for_failed_deployment(
            litellm_router=router, kwargs=kwargs, exception=exc
        )

    _, call_kwargs = mock_set.call_args
    assert call_kwargs["deployment"] == "batch-deployment"


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
    logging_obj.model_call_details = {"has_logged_async_failure": True}

    kwargs = {
        "litellm_metadata": {"model_info": {"id": "dep-xyz"}, "deployment_model_name": "gpt-4"},
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
    logging_obj.model_call_details = {"has_logged_async_failure": False}

    kwargs = {
        "litellm_metadata": {"model_info": {"id": "dep-xyz"}, "deployment_model_name": "gpt-4"},
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
