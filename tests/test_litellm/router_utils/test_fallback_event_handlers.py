import json

import pytest

from litellm.router_utils.fallback_event_handlers import (
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


class AttemptRecordingRouter:
    def __init__(self):
        self.attempted_model_groups = []
        self.received_kwargs = None

    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        self.attempted_model_groups.append(kwargs.get("model"))
        self.received_kwargs = kwargs
        return StreamingWrapper()


async def _acreate_batch(*args, **kwargs):
    raise AssertionError("only used for its __name__")


@pytest.mark.asyncio
async def test_run_async_fallback_keeps_uploaded_file_requests_in_their_model_group():
    """An input_file_id only exists under the credentials of the group it was uploaded
    to, so a cross-group fallback can only fail with the wrong provider's error."""
    router = AttemptRecordingRouter()
    owning_provider_error = RuntimeError("openai connection error")

    with pytest.raises(RuntimeError, match="openai connection error"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=["azure-group"],
            original_model_group="openai-group",
            original_exception=owning_provider_error,
            max_fallbacks=3,
            fallback_depth=0,
            model="openai-group",
            input_file_id="file-owned-by-openai",
            original_function=_acreate_batch,
        )

    assert router.attempted_model_groups == []


@pytest.mark.asyncio
async def test_run_async_fallback_keeps_fine_tuning_requests_in_their_model_group():
    router = AttemptRecordingRouter()

    with pytest.raises(RuntimeError, match="openai connection error"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=["azure-group"],
            original_model_group="openai-group",
            original_exception=RuntimeError("openai connection error"),
            max_fallbacks=3,
            fallback_depth=0,
            model="openai-group",
            training_file="file-owned-by-openai",
        )

    assert router.attempted_model_groups == []


@pytest.mark.asyncio
async def test_run_async_fallback_allows_same_model_group_retry_for_uploaded_file_requests():
    """Order-based fallbacks stay inside the owning group, so they must still run."""
    router = AttemptRecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=[{"model": "openai-group", "_target_order": 2}],
        original_model_group="openai-group",
        original_exception=RuntimeError("first deployment failed"),
        max_fallbacks=3,
        fallback_depth=0,
        model="openai-group",
        input_file_id="file-owned-by-openai",
        original_function=_acreate_batch,
    )

    assert router.attempted_model_groups == ["openai-group"]


@pytest.mark.asyncio
async def test_run_async_fallback_still_crosses_model_groups_without_an_uploaded_file():
    router = AttemptRecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["azure-group"],
        original_model_group="openai-group",
        original_exception=RuntimeError("openai connection error"),
        max_fallbacks=3,
        fallback_depth=0,
        model="openai-group",
    )

    assert router.attempted_model_groups == ["azure-group"]


@pytest.mark.asyncio
async def test_run_async_fallback_handles_explicitly_none_metadata():
    """/v1/batches always sets `metadata`, and sets it to None when the caller sent
    none, so setdefault() on it hands back None instead of a dict."""
    router = AttemptRecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["azure-group"],
        original_model_group="openai-group",
        original_exception=RuntimeError("openai connection error"),
        max_fallbacks=3,
        fallback_depth=0,
        model="openai-group",
        metadata=None,
    )

    assert router.received_kwargs["metadata"] == {"model_group": "azure-group"}


@pytest.mark.asyncio
async def test_run_async_fallback_records_batch_model_group_outside_provider_metadata():
    """`metadata` on a batch request is forwarded to the provider and stored on the
    batch, so the router's own model_group belongs in litellm_metadata."""
    router = AttemptRecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=[{"model": "openai-group", "_target_order": 2}],
        original_model_group="openai-group",
        original_exception=RuntimeError("first deployment failed"),
        max_fallbacks=3,
        fallback_depth=0,
        model="openai-group",
        input_file_id="file-owned-by-openai",
        metadata={"caller": "nightly-job"},
        litellm_metadata={"model_group": "openai-group"},
        original_function=_acreate_batch,
    )

    assert router.received_kwargs["metadata"] == {"caller": "nightly-job"}
    assert router.received_kwargs["litellm_metadata"]["model_group"] == "openai-group"


def test_get_fallback_model_group_does_not_mutate_fallbacks():
    """A string fallback must be resolved without mutating the caller's
    fallbacks list, which is the live router config shared across requests."""
    fallbacks = [{"gpt-3.5-turbo": ["claude-3-haiku"]}, "gpt-4o-mini"]

    fallback_model_group, _ = get_fallback_model_group(
        fallbacks=fallbacks, model_group="unmatched-model"
    )

    assert fallback_model_group == ["gpt-4o-mini"]
    assert fallbacks == [{"gpt-3.5-turbo": ["claude-3-haiku"]}, "gpt-4o-mini"]
