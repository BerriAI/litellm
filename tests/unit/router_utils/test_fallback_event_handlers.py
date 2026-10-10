import asyncio
import importlib
import json
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Any, Final, Literal, NoReturn
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils import get_llm_provider_logic
from litellm.router_utils.cooldown_handlers import mark_advisor_orchestration_failure
from litellm.router_utils.fallback_event_handlers import (
    MID_STREAM_FALLBACK_CONTROLS_KEY,
    PRE_ROUTING_SELECTED_MODEL_KEY,
    AttemptedFallbackTargets,
    MidStreamFallbackControls,
    _trigger_cooldown_for_failed_deployment,
    attempted_retries_for_request,
    carry_over_routed_deployment,
    clear_pre_routing_selection,
    committed_retry_budget_for_request,
    fallback_attempt_key,
    get_fallback_model_group,
    get_pre_routing_selection,
    log_failure_fallback_event,
    log_success_fallback_event,
    mid_stream_fallback_snapshot_kwargs,
    mid_stream_retry_kwargs,
    record_pre_routing_selection,
    record_retry_attempt,
    routed_deployment_id,
    run_async_fallback,
)
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome
from tests.fake_openai_endpoint import FAKE_OPENAI_API_BASE
from typing import Dict
import os


class StreamingWrapper:
    def __init__(self):
        self._hidden_params = {"additional_headers": {}}


class FakeRouter:
    fallback_access_check = None
    fallback_budget_check = None

    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        return StreamingWrapper()


class AlwaysFailRouter:
    fallback_access_check = None
    fallback_budget_check = None

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
    fallback_access_check = None
    fallback_budget_check = None

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
    fallback_access_check = None
    fallback_budget_check = None

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


async def _acreate_file(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("only used for its __name__")


async def _acancel_batch(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("only used for its __name__")


async def _acompletion(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("only used for its __name__")


async def _ageneric_api_call_with_fallbacks_helper(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("only used for its __name__")


async def acreate_fine_tuning_job(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("only used for its __name__")


async def aretrieve_fine_tuning_job(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("only used for its __name__")


async def afile_content(*args: object, **kwargs: object) -> NoReturn:
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
            original_function=_ageneric_api_call_with_fallbacks_helper,
            original_generic_function=acreate_fine_tuning_job,
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
async def test_run_async_fallback_keeps_file_creation_in_its_model_group():
    """A file created for batches lands in the account of the deployment that stored it,
    and its id is only usable against the model group the caller named. A cross-group
    fallback silently stores the file with the wrong provider."""
    router = AttemptRecordingRouter()

    with pytest.raises(RuntimeError, match="azure connection error"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=["openai-group"],
            original_model_group="azure-group",
            original_exception=RuntimeError("azure connection error"),
            max_fallbacks=3,
            fallback_depth=0,
            model="azure-group",
            original_function=_acreate_file,
        )

    assert router.attempted_model_groups == []


@pytest.mark.asyncio
async def test_run_async_fallback_allows_same_model_group_retry_for_file_creation():
    router = AttemptRecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=[{"model": "azure-group", "_target_order": 2}],
        original_model_group="azure-group",
        original_exception=RuntimeError("first deployment failed"),
        max_fallbacks=3,
        fallback_depth=0,
        model="azure-group",
        original_function=_acreate_file,
    )

    assert router.attempted_model_groups == ["azure-group"]


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
@pytest.mark.parametrize(
    ("resource_key", "handler_kwargs"),
    [
        ("batch_id", {"original_function": _acancel_batch}),
        (
            "file_id",
            {
                "original_function": _ageneric_api_call_with_fallbacks_helper,
                "original_generic_function": afile_content,
            },
        ),
        (
            "fine_tuning_job_id",
            {
                "original_function": _ageneric_api_call_with_fallbacks_helper,
                "original_generic_function": aretrieve_fine_tuning_job,
            },
        ),
    ],
)
async def test_run_async_fallback_keeps_provider_scoped_ids_in_their_model_group(
    resource_key: str, handler_kwargs: dict
):
    """A batch, file, or fine-tuning job id only exists under the credentials of the group
    that issued it, so a cross-group fallback asks a provider about an id it never saw.
    Generic API calls carry the real handler in original_generic_function, so the pin
    must recognize it there too."""
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
            **{resource_key: "owned-by-openai"},
            **handler_kwargs,
        )

    assert router.attempted_model_groups == []


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_key", ["batch_id", "file_id", "fine_tuning_job_id"])
async def test_run_async_fallback_ignores_stray_resource_ids_on_completion_calls(resource_key: str):
    """A caller-supplied top-level field like file_id on a chat completion is application
    data, never a provider resource reference, so it must not cost the request its
    cross-group fallbacks."""
    router = AttemptRecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["azure-group"],
        original_model_group="openai-group",
        original_exception=RuntimeError("openai connection error"),
        max_fallbacks=3,
        fallback_depth=0,
        model="openai-group",
        original_function=_acompletion,
        **{resource_key: "caller-app-data"},
    )

    assert router.attempted_model_groups == ["azure-group"]


@pytest.mark.asyncio
async def test_run_async_fallback_allows_same_model_group_retry_for_batch_cancel():
    router = AttemptRecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=[{"model": "openai-group", "_target_order": 2}],
        original_model_group="openai-group",
        original_exception=RuntimeError("first deployment failed"),
        max_fallbacks=3,
        fallback_depth=0,
        model="openai-group",
        batch_id="owned-by-openai",
        original_function=_acancel_batch,
    )

    assert router.attempted_model_groups == ["openai-group"]


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

    assert router.received_kwargs["metadata"] == {
        "model_group": "azure-group",
        "attempted_fallbacks": 1,
        "original_model_group": "openai-group",
    }


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


class AccessCheckedRouter(AttemptRecordingRouter):
    def __init__(self, allowed_models: frozenset[str]):
        super().__init__()
        self.allowed_models = allowed_models
        self.access_checks = []

    fallback_budget_check = None

    async def fallback_access_check(self, *, model, request_kwargs, llm_router):
        self.access_checks.append((model, request_kwargs["metadata"]["user_api_key"], llm_router is self))
        return model in self.allowed_models


@pytest.mark.asyncio
async def test_run_async_fallback_skips_targets_the_access_check_rejects():
    router = AccessCheckedRouter(allowed_models=frozenset({"allowed-model"}))

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=[
            {"model": "secret-model", "messages": [{"role": "user", "content": "hi"}]},
            "allowed-model",
        ],
        original_model_group="primary-model",
        original_exception=RuntimeError("primary failed"),
        max_fallbacks=3,
        fallback_depth=0,
        model="primary-model",
        metadata={"user_api_key": "hashed"},
    )

    assert router.attempted_model_groups == ["allowed-model"]
    assert router.access_checks == [
        ("secret-model", "hashed", True),
        ("allowed-model", "hashed", True),
    ]


@pytest.mark.asyncio
async def test_run_async_fallback_raises_original_error_when_no_target_is_authorized():
    router = AccessCheckedRouter(allowed_models=frozenset())

    with pytest.raises(RuntimeError, match="primary failed"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=["secret-model", "other-secret-model"],
            original_model_group="primary-model",
            original_exception=RuntimeError("primary failed"),
            max_fallbacks=3,
            fallback_depth=0,
            model="primary-model",
            metadata={"user_api_key": "hashed"},
        )

    assert router.attempted_model_groups == []
    assert [model for model, _, _ in router.access_checks] == ["secret-model", "other-secret-model"]


@pytest.mark.asyncio
async def test_run_async_fallback_does_not_consult_access_check_for_same_model_group_retries():
    router = AccessCheckedRouter(allowed_models=frozenset())

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=[{"model": "primary-model", "_target_order": 2}],
        original_model_group="primary-model",
        original_exception=RuntimeError("first order level failed"),
        max_fallbacks=3,
        fallback_depth=0,
        model="primary-model",
        metadata={"user_api_key": "hashed"},
    )

    assert router.attempted_model_groups == ["primary-model"]
    assert router.access_checks == []


class RecordingFailRouter:
    fallback_access_check = None
    fallback_budget_check = None

    def __init__(self):
        self.attempted_models = []

    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        self.attempted_models.append(kwargs.get("model"))
        raise RuntimeError("fallback model also failed")


@pytest.mark.asyncio
async def test_run_async_fallback_skips_model_group_already_attempted():
    """A fallback graph that loops back on itself must not re-attempt a model group that
    already failed for this request. Every group in a cycle fails identically, so
    revisiting one multiplies the work and the error output without any chance of
    succeeding."""
    router = RecordingFailRouter()

    with pytest.raises(RuntimeError, match="original failed"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=["already-attempted"],
            original_model_group="primary-model",
            original_exception=RuntimeError("original failed"),
            max_fallbacks=3,
            fallback_depth=0,
            attempted_targets=AttemptedFallbackTargets(frozenset({"already-attempted"})),
        )

    assert router.attempted_models == []


@pytest.mark.asyncio
async def test_run_async_fallback_attempts_a_repeated_target_once():
    router = RecordingFailRouter()

    with pytest.raises(RuntimeError, match="fallback model also failed"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=["fallback-model", "fallback-model", "other-model"],
            original_model_group="primary-model",
            original_exception=RuntimeError("original failed"),
            max_fallbacks=5,
            fallback_depth=0,
        )

    assert router.attempted_models == ["fallback-model", "other-model"]


@pytest.mark.asyncio
async def test_run_async_fallback_forwards_attempted_model_groups_to_nested_call():
    """The nested call is where the next hop of the walk decides what to skip, so the
    accumulated set has to reach it, carrying both the group that just failed and the
    target being attempted."""
    router = RecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("original failed"),
        max_fallbacks=3,
        fallback_depth=0,
        attempted_targets=AttemptedFallbackTargets(frozenset({"earlier-model"})),
    )

    assert router.received_kwargs["attempted_targets"].keys == frozenset(
        {"earlier-model", "primary-model", "fallback-model"}
    )


@pytest.mark.asyncio
async def test_run_async_fallback_can_target_the_requested_group_when_a_pre_router_replaced_it():
    """The requested group was never called when a pre-router selected a tier, so a
    tier fallback may legitimately target that originally requested group."""
    router = RecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["requested-model"],
        original_model_group="requested-model",
        original_exception=RuntimeError("selected tier failed"),
        max_fallbacks=3,
        fallback_depth=0,
        model="requested-model",
        metadata={"pre_routing_selected_model": "selected-tier"},
    )

    assert router.received_kwargs["model"] == "requested-model"
    assert router.received_kwargs["attempted_targets"].keys == frozenset({"selected-tier", "requested-model"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entry",
    [
        {"model": "primary-model", "_target_order": 2},
        {"model": "primary-model", "_excluded_deployment_ids": ["dep-1"]},
    ],
)
async def test_run_async_fallback_still_retargets_the_same_group_via_dict_entry(entry):
    """Order-based fallback and weighted intra-group failover both re-target the group that
    just failed, selecting a different set of deployments inside it. Those entries are dicts
    rather than plain names and must survive a guard that skips already-attempted names."""
    router = RecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=[entry],
        original_model_group="primary-model",
        original_exception=RuntimeError("original failed"),
        max_fallbacks=3,
        fallback_depth=0,
        attempted_targets=AttemptedFallbackTargets(frozenset({"primary-model"})),
    )

    assert router.received_kwargs["model"] == "primary-model"


@pytest.mark.asyncio
async def test_run_async_fallback_skips_a_repeated_dict_target():
    """A client-side fallback list names its targets with dicts, and that list is re-walked
    at every level of the recursion, so an entry that carries no request override has to be
    recognised as the same attempt as the bare name."""
    router = RecordingFailRouter()

    with pytest.raises(RuntimeError, match="original failed"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=[{"model": "already-attempted"}],
            original_model_group="primary-model",
            original_exception=RuntimeError("original failed"),
            max_fallbacks=3,
            fallback_depth=0,
            attempted_targets=AttemptedFallbackTargets(frozenset({"already-attempted"})),
        )

    assert router.attempted_models == []


@pytest.mark.asyncio
async def test_run_async_fallback_attempts_a_repeated_dict_target_once():
    router = RecordingFailRouter()
    entry = {"model": "fallback-model", "messages": [{"role": "user", "content": "shorter"}]}

    with pytest.raises(RuntimeError, match="fallback model also failed"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=[entry, entry, {"model": "other-model"}],
            original_model_group="primary-model",
            original_exception=RuntimeError("original failed"),
            max_fallbacks=5,
            fallback_depth=0,
        )

    assert router.attempted_models == ["fallback-model", "other-model"]


@pytest.mark.asyncio
async def test_run_async_fallback_keeps_a_request_override_distinct_from_the_bare_name():
    """The documented use of the client-side form is to retry a group with different request
    params, so an entry carrying an override must survive even when the bare name of that
    same group has already been attempted."""
    router = RecordingFailRouter()

    with pytest.raises(RuntimeError, match="fallback model also failed"):
        await run_async_fallback(
            litellm_router=router,
            fallback_model_group=[{"model": "already-attempted", "messages": [{"role": "user", "content": "shorter"}]}],
            original_model_group="primary-model",
            original_exception=RuntimeError("original failed"),
            max_fallbacks=3,
            fallback_depth=0,
            attempted_targets=AttemptedFallbackTargets(frozenset({"already-attempted"})),
        )

    assert router.attempted_models == ["already-attempted"]


@pytest.mark.parametrize(
    "target, expected",
    [
        ("group-a", "group-a"),
        ({"model": "group-a"}, "group-a"),
        (None, None),
        (["group-a"], None),
    ],
)
def test_fallback_attempt_key_identity(target, expected):
    """A bare name and a `{"model": name}` entry are the same attempt. A shape with no
    usable identity returns None and is never skipped, so an unrecognised entry keeps
    today's behaviour rather than being silently dropped."""
    assert fallback_attempt_key(target) == expected


def test_fallback_attempt_key_gives_a_param_only_entry_its_own_identity():
    """An entry with no `model` re-targets the group currently being attempted with
    different request params, so it is a distinct attempt and still needs an identity."""
    key = fallback_attempt_key({"messages": [{"role": "user", "content": "shorter"}]})

    assert key is not None
    assert key != fallback_attempt_key({"messages": [{"role": "user", "content": "other"}]})


def test_fallback_attempt_key_separates_overrides_from_the_bare_name():
    bare = fallback_attempt_key("group-a")
    override = fallback_attempt_key({"model": "group-a", "messages": [{"role": "user", "content": "x"}]})
    other_override = fallback_attempt_key({"model": "group-a", "messages": [{"role": "user", "content": "y"}]})
    order_retarget = fallback_attempt_key({"model": "group-a", "_target_order": 2})

    assert len({bare, override, other_override, order_retarget}) == 4


def test_fallback_attempt_key_is_stable_across_key_order():
    assert fallback_attempt_key({"model": "group-a", "_target_order": 2}) == fallback_attempt_key(
        {"_target_order": 2, "model": "group-a"}
    )


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

        with patch("litellm.router_utils.fallback_event_handlers.set_cooldown_deployments") as mock_set_cooldown:
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

        with patch("litellm.router_utils.fallback_event_handlers.set_cooldown_deployments") as mock_set_cooldown:
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
            patch("litellm.router_utils.fallback_event_handlers.set_cooldown_deployments") as mock_set_cooldown,
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

        with patch("litellm.router_utils.fallback_event_handlers.set_cooldown_deployments") as mock_set_cooldown:
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

        with patch("litellm.router_utils.fallback_event_handlers.set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(litellm_router=mock_router, kwargs={}, exception=exc)

            mock_set_cooldown.assert_not_called()

    def test_uses_deployment_litellm_params_cooldown_time_override(self):
        mock_router = MagicMock()
        mock_router.cooldown_time = 300.0
        mock_router.get_model_info.return_value = {"litellm_params": {"cooldown_time": 30.0}}

        exc = litellm.RateLimitError("Rate limit", "openai", "gpt-4")
        exc.failed_deployment_id = "fallback-deployment"

        with patch("litellm.router_utils.fallback_event_handlers.set_cooldown_deployments") as mock_set_cooldown:
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

        with patch("litellm.router_utils.fallback_event_handlers.set_cooldown_deployments") as mock_set_cooldown:
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
            "litellm.router_utils.fallback_event_handlers.set_cooldown_deployments",
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
            patch("litellm.router_utils.fallback_event_handlers.set_cooldown_deployments") as mock_set_cooldown,
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

        with patch("litellm.router_utils.fallback_event_handlers.set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(litellm_router=mock_router, kwargs={}, exception=exc)

            mock_set_cooldown.assert_called_once()

    def test_skips_client_side_timeout_408(self):
        """The proxy's x-litellm-timeout header lets a caller set an arbitrarily short
        timeout, which litellm.Timeout reports as status 408 regardless of the
        deployment's actual health. Without this guard, a caller could force a 408 on
        every deployment in the fallback chain from a single request.

        The failure logger never stamps end_time for a fallback hop (has_logged_async_failure
        is already set), so model_call_details still carries the previous hop's end_time, which
        predates this hop's api_call_start_time. The guard must not trust it."""
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.Timeout(message="timeout", model="gpt-4", llm_provider="openai")
        exc.failed_deployment_id = "fallback-deployment"

        with (
            patch("litellm.router_utils.fallback_event_handlers.set_cooldown_deployments") as mock_set_cooldown,
            patch(
                "litellm.router_utils.fallback_event_handlers.increment_deployment_failures_for_current_minute"
            ) as mock_increment,
        ):
            _trigger_cooldown_for_failed_deployment(
                litellm_router=mock_router,
                kwargs={"client_side_timeout": True},
                exception=exc,
                model_call_details={
                    "litellm_params": {"client_side_timeout": True, "timeout": 0.5},
                    "api_call_start_time": datetime.now() - timedelta(seconds=1),
                    "end_time": datetime.now() - timedelta(seconds=5),
                },
            )

            mock_set_cooldown.assert_not_called()
            mock_increment.assert_not_called()

    @pytest.mark.asyncio
    async def test_still_cools_down_provider_408_before_caller_deadline(self):
        """client_side_timeout only records that the caller configured a timeout. A 408
        that comes back before that deadline was raised by the provider itself, so it is
        a real health signal and must still cool the deployment down."""
        from litellm.router_utils.router_callbacks.track_deployment_metrics import (
            get_deployment_failures_for_current_minute,
        )

        router = litellm.Router(
            model_list=[
                {
                    "model_name": "fallback-model",
                    "litellm_params": {"model": "openai/gpt-5.6", "api_key": "sk-fake"},
                    "model_info": {"id": "fallback-deployment"},
                }
            ],
            allowed_fails=0,
            cooldown_time=60,
            num_retries=0,
        )
        exc = litellm.Timeout(message="timeout", model="gpt-5.6", llm_provider="openai")
        exc.failed_deployment_id = "fallback-deployment"
        started = datetime.now()

        _trigger_cooldown_for_failed_deployment(
            litellm_router=router,
            kwargs={"client_side_timeout": True},
            exception=exc,
            model_call_details={
                "litellm_params": {"client_side_timeout": True, "timeout": 30},
                "api_call_start_time": started,
                "end_time": started + timedelta(seconds=1),
            },
        )

        assert (
            get_deployment_failures_for_current_minute(
                litellm_router_instance=router, deployment_id="fallback-deployment"
            )
            == 1
        )
        active = router.cooldown_cache.get_active_cooldowns(model_ids=["fallback-deployment"], parent_otel_span=None)
        assert [entry[0] for entry in active] == ["fallback-deployment"]

    def test_still_cools_down_408_without_client_side_timeout_flag(self):
        """The client-side-timeout guard is scoped to caller-supplied timeouts only: a
        408 that did not come from x-litellm-timeout (no client_side_timeout in kwargs)
        must still cool down the deployment as before."""
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.Timeout(message="timeout", model="gpt-4", llm_provider="openai")
        exc.failed_deployment_id = "fallback-deployment"

        with patch("litellm.router_utils.fallback_event_handlers.set_cooldown_deployments") as mock_set_cooldown:
            _trigger_cooldown_for_failed_deployment(litellm_router=mock_router, kwargs={}, exception=exc)

            mock_set_cooldown.assert_called_once()


class TestRunAsyncFallbackTriggersCooldown:
    class RouterWithLoggingKwarg:
        fallback_access_check = None
        fallback_budget_check = None

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


@pytest.mark.asyncio
async def test_a_stored_fallback_target_cannot_carry_a_federation_field():
    """A dict fallback target is merged into kwargs, and kwargs beat the deployment's own params,
    so a stored key/team/global fallback could otherwise set the workspace a federation token is
    minted for. The request itself is already forbidden to carry these, and a stored setting is
    not a more trusted source than the request."""
    with pytest.raises(
        ValueError,
        match="server-owned workload identity federation or OAuth token exchange parameter",
    ):
        await run_async_fallback(
            litellm_router=FakeRouter(),
            fallback_model_group=[{"model": "anthropic-backup", "anthropic_federation_workspace_id": "wrkspc_other"}],
            original_model_group="primary-model",
            original_exception=RuntimeError("upstream limited request"),
            max_fallbacks=3,
            fallback_depth=0,
        )


@pytest.mark.asyncio
async def test_a_stored_fallback_target_cannot_carry_an_openai_federation_field():
    """The OpenAI identity trio is server-owned for the same reason: a stored fallback target
    naming a token file would pick which workload assertion is exchanged for the bearer."""
    with pytest.raises(ValueError, match="openai_identity_token_file"):
        await run_async_fallback(
            litellm_router=FakeRouter(),
            fallback_model_group=[
                {"model": "openai-backup", "openai_identity_token_file": "/var/run/secrets/tokens/other"}
            ],
            original_model_group="primary-model",
            original_exception=RuntimeError("upstream limited request"),
            max_fallbacks=3,
            fallback_depth=0,
        )


@pytest.mark.asyncio
async def test_the_refusal_is_not_swallowed_as_a_fallback_error():
    """Checked before the per-target loop on purpose: inside it, the refusal would be caught as
    that target's failure and the run would quietly continue to the next one."""
    with pytest.raises(ValueError, match="anthropic_issuer_signing_key_ref"):
        await run_async_fallback(
            litellm_router=FakeRouter(),
            fallback_model_group=[
                {"model": "anthropic-backup", "anthropic_issuer_signing_key_ref": "os.environ/ADMIN_KEY"},
                "a-perfectly-fine-model",
            ],
            original_model_group="primary-model",
            original_exception=RuntimeError("upstream limited request"),
            max_fallbacks=3,
            fallback_depth=0,
            include_fallback_errors=True,
        )


async def test_run_async_fallback_stamps_fallback_info_into_metadata():
    """Spend logs are built from the request metadata of the nested call, so the
    fallback signal has to be stamped there before recursing."""
    router = RecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("original failed"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    metadata = router.received_kwargs["metadata"]
    assert metadata["attempted_fallbacks"] == 1
    assert metadata["original_model_group"] == "primary-model"
    assert metadata["model_group"] == "fallback-model"


@pytest.mark.asyncio
async def test_run_async_fallback_preserves_original_model_group_on_nested_fallback():
    """A second-level fallback receives the first fallback target as its
    original_model_group argument, so the first-stamped value must survive the hop."""
    router = RecordingRouter()

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["second-fallback"],
        original_model_group="first-fallback",
        original_exception=RuntimeError("first fallback failed"),
        max_fallbacks=3,
        fallback_depth=1,
        metadata={"attempted_fallbacks": 1, "original_model_group": "primary-model"},
    )

    metadata = router.received_kwargs["metadata"]
    assert metadata["attempted_fallbacks"] == 2
    assert metadata["original_model_group"] == "primary-model"


class TestPreRoutingSelectionCarriesToFallbacks:
    """#38832: a complexity/auto router picks a tier behind the router name, but fallback
    lookup kept using the router name, so the tier's configured chain never ran."""

    def test_selection_is_recorded_in_the_metadata_bucket(self):
        kwargs = {"model": "smart-router", "metadata": {}}
        record_pre_routing_selection(kwargs, "tier1")
        assert kwargs["metadata"]["pre_routing_selected_model"] == "tier1"
        assert get_pre_routing_selection(kwargs) == "tier1"

    def test_selection_is_recorded_in_the_litellm_metadata_bucket(self):
        kwargs = {"model": "smart-router", "litellm_metadata": {}}
        record_pre_routing_selection(kwargs, "tier2")
        assert get_pre_routing_selection(kwargs) == "tier2"

    def test_a_bucket_survives_the_kwargs_copy_that_fallbacks_run_on(self):
        """The bucket is shared by reference, which is the whole reason this works."""
        outer = {"model": "smart-router", "metadata": {}}
        inner = {**outer}
        record_pre_routing_selection(inner, "tier1")
        assert get_pre_routing_selection(outer) == "tier1"

    def test_no_selection_reads_as_none(self):
        assert get_pre_routing_selection({"model": "smart-router", "metadata": {}}) is None
        assert get_pre_routing_selection({"model": "smart-router"}) is None

    def test_missing_kwargs_is_a_no_op(self):
        """A caller with no kwargs must not raise, and must not leak the selection anywhere."""
        record_pre_routing_selection(None, "tier1")

        assert get_pre_routing_selection({}) is None

    def test_a_non_dict_bucket_is_ignored(self):
        kwargs = {"model": "smart-router", "metadata": "not-a-dict"}
        record_pre_routing_selection(kwargs, "tier1")
        assert get_pre_routing_selection(kwargs) is None

    def test_fallbacks_resolve_against_the_selected_tier(self):
        """The lookup the router performs, keyed on the tier rather than the router name."""
        fallbacks = [{"tier1": ["backup-a", "backup-b"]}, {"tier2": ["backup-c"]}]
        assert get_fallback_model_group(fallbacks=fallbacks, model_group="tier1")[0] == ["backup-a", "backup-b"]
        assert get_fallback_model_group(fallbacks=fallbacks, model_group="smart-router")[0] is None


class TestPreRoutingSelectionIsPerHop:
    """#38832 review: the buckets also carry whatever the caller sent, and a fallback hop
    inherits the previous hop's tier, so a hop must start without a selection."""

    def test_a_caller_supplied_selection_is_dropped(self):
        kwargs = {"model": "plain", "metadata": {"pre_routing_selected_model": "tier1"}}

        clear_pre_routing_selection(kwargs)

        assert get_pre_routing_selection(kwargs) is None
        assert "pre_routing_selected_model" not in kwargs["metadata"]

    def test_both_buckets_are_cleared(self):
        kwargs = {
            "metadata": {"pre_routing_selected_model": "tier1"},
            "litellm_metadata": {"pre_routing_selected_model": "tier2"},
        }

        clear_pre_routing_selection(kwargs)

        assert get_pre_routing_selection(kwargs) is None

    def test_the_rest_of_the_bucket_is_left_alone(self):
        kwargs = {"metadata": {"pre_routing_selected_model": "tier1", "tags": ["a"]}}

        clear_pre_routing_selection(kwargs)

        assert kwargs["metadata"] == {"tags": ["a"]}

    def test_clearing_is_a_no_op_without_a_usable_bucket(self):
        kwargs = {"model": "plain", "metadata": "not-a-dict"}

        clear_pre_routing_selection(None)
        clear_pre_routing_selection(kwargs)

        assert kwargs == {"model": "plain", "metadata": "not-a-dict"}

    def test_a_selection_recorded_after_clearing_is_kept(self):
        """Clearing runs before routing, so the hook's own write must survive it."""
        kwargs = {"model": "smart-router", "metadata": {"pre_routing_selected_model": "stale"}}

        clear_pre_routing_selection(kwargs)
        record_pre_routing_selection(kwargs, "tier1")

        assert get_pre_routing_selection(kwargs) == "tier1"


class TestOrderedFallbackLookupGroups:
    def test_tier_first_then_requested_group_deduped(self):
        from litellm.router_utils.fallback_event_handlers import (
            PRE_ROUTING_SELECTED_MODEL_KEY,
            fallback_lookup_groups,
        )

        kwargs = {"litellm_metadata": {PRE_ROUTING_SELECTED_MODEL_KEY: "tier1"}}
        assert fallback_lookup_groups(kwargs, "smart-router") == ("tier1", "smart-router")
        assert fallback_lookup_groups(kwargs, "tier1") == ("tier1",)
        assert fallback_lookup_groups({}, "smart-router") == ("smart-router",)
        assert fallback_lookup_groups({}, None) == ()

    def test_session_remap_keeps_the_bound_router_between_tier_and_requested_group(self):
        from litellm.router_utils.fallback_event_handlers import (
            PRE_ROUTING_SELECTED_MODEL_KEY,
            fallback_lookup_groups,
        )

        kwargs = {
            "litellm_metadata": {
                PRE_ROUTING_SELECTED_MODEL_KEY: "tier1",
                "model_group": "smart-router",
            }
        }

        assert fallback_lookup_groups(kwargs, "requested-model") == (
            "tier1",
            "smart-router",
            "requested-model",
        )
        assert fallback_lookup_groups({"metadata": {"model_group": []}}, "requested-model") == (
            "requested-model",
        )

    def test_fallback_hop_resumes_the_original_groups_chain_last(self):
        from litellm.router_utils.fallback_event_handlers import fallback_lookup_groups

        kwargs = {"metadata": {"model_group": "fb1", "original_model_group": "primary"}}

        assert fallback_lookup_groups(kwargs, "fb1") == ("fb1", "primary")
        assert fallback_lookup_groups({"metadata": {"original_model_group": 42}}, "fb1") == ("fb1",)

    def test_first_resolving_group_wins_and_generic_idx_survives_a_miss(self):
        from litellm.router_utils.fallback_event_handlers import (
            get_fallback_model_group_for_lookup_groups,
        )

        fallbacks = [{"tier1": ["backup-a"]}, {"smart-router": ["backup-b"]}, {"*": ["backup-c"]}]
        assert get_fallback_model_group_for_lookup_groups(fallbacks, ("tier1", "smart-router")) == (["backup-a"], None)
        assert get_fallback_model_group_for_lookup_groups(fallbacks, ("tier9", "smart-router")) == (["backup-b"], None)
        assert get_fallback_model_group_for_lookup_groups(fallbacks, ("tier9", "no-such")) == (["backup-c"], 2)
        assert get_fallback_model_group_for_lookup_groups([{"tier1": ["backup-a"]}], ("no", "nope")) == (None, None)

    def test_a_string_valued_generic_rule_never_shadows_a_later_groups_own_chain(self):
        from litellm.router_utils.fallback_event_handlers import (
            get_fallback_model_group_for_lookup_groups,
        )

        fallbacks: Final = [{"smart-router": "backup-b"}, {"*": "backup-c"}]
        assert get_fallback_model_group_for_lookup_groups(fallbacks, ("tier9", "smart-router")) == (["backup-b"], None)
        assert get_fallback_model_group_for_lookup_groups(fallbacks, ("tier9", "no-such")) == (["backup-c"], 1)

    def test_two_catch_all_rules_never_shadow_a_later_groups_own_chain(self):
        from litellm.router_utils.fallback_event_handlers import (
            get_fallback_model_group_for_lookup_groups,
        )

        fallbacks: Final = [{"*": ["backup-a"]}, {"*": ["backup-b"]}, {"primary": ["backup-c"]}]
        assert get_fallback_model_group_for_lookup_groups(fallbacks, ("missing", "primary")) == (["backup-c"], 1)
        assert get_fallback_model_group_for_lookup_groups(fallbacks, ("missing", "no-such")) == (["backup-b"], 1)


class TestHasUnattemptedFallbackTarget:
    def test_exhausted_chain_is_not_recoverable_but_a_fresh_entry_is(self):
        from litellm.router_utils.fallback_event_handlers import (
            has_unattempted_fallback_target,
        )

        attempted: Final = AttemptedFallbackTargets()
        attempted.record("primary")
        attempted.record("fb1")
        attempted.record("fb2")

        assert has_unattempted_fallback_target(["fb1", "fb2"], {"attempted_targets": attempted}) is False
        assert has_unattempted_fallback_target(["fb1", "fb3"], {"attempted_targets": attempted}) is True
        assert has_unattempted_fallback_target(["fb1"], {}) is True
        assert has_unattempted_fallback_target(None, {}) is False


def test_get_fallback_model_group_matches_provider_prefixed_key():
    """A bare model group routed via a wildcard (e.g. "gpt-4o" through
    "openai/*") must match a fallback keyed on the provider-prefixed name,
    which is the form the Admin UI offers for wildcard routes."""
    fallbacks = [{"openai/gpt-4o": ["claude-3-haiku"]}]

    fallback_model_group, _ = get_fallback_model_group(fallbacks=fallbacks, model_group="gpt-4o")

    assert fallback_model_group == ["claude-3-haiku"]


def test_get_fallback_model_group_exact_match_beats_prefixed_match():
    fallbacks = [
        {"openai/gpt-4o": ["claude-3-haiku"]},
        {"gpt-4o": ["gemini-1.5-flash"]},
    ]

    fallback_model_group, _ = get_fallback_model_group(fallbacks=fallbacks, model_group="gpt-4o")

    assert fallback_model_group == ["gemini-1.5-flash"]


@pytest.mark.parametrize("rule_key", ["gpt-5.4-mini", "openai/gpt-5.4-mini", "*"])
def test_get_fallback_model_group_resolves_a_bare_string_rule_value_as_a_one_item_chain(rule_key: str):
    """Router.__init__ documents fallbacks=[{"primary": "backup"}]; the value is the chain to try, so a
    bare string is a chain of one and never iterated character by character."""
    fallback_model_group, _ = get_fallback_model_group(fallbacks=[{rule_key: "gpt-5.4-nano"}], model_group="gpt-5.4-mini")

    assert fallback_model_group == ["gpt-5.4-nano"]


def test_get_fallback_model_group_keeps_a_list_rule_value_as_is():
    fallbacks: Final = [{"gpt-5.4-mini": ["gpt-5.4-nano", "gpt-5.4"]}]

    fallback_model_group, _ = get_fallback_model_group(fallbacks=fallbacks, model_group="gpt-5.4-mini")

    assert fallback_model_group is fallbacks[0]["gpt-5.4-mini"]


def test_get_fallback_model_group_prefixed_match_ignores_unknown_models():
    """Provider inference fails for unknown bare names - the lookup must not
    raise and must fall through to the generic fallback."""
    fallbacks = [
        {"openai/some-model": ["claude-3-haiku"]},
        {"*": ["gemini-1.5-flash"]},
    ]

    fallback_model_group, _ = get_fallback_model_group(fallbacks=fallbacks, model_group="some-unknown-model-xyz")

    assert fallback_model_group == ["gemini-1.5-flash"]


def test_get_fallback_model_group_prefixed_match_skips_prefixed_model_group():
    """An already-prefixed model group must not double-prefix."""
    fallbacks = [{"openai/openai/gpt-4o": ["claude-3-haiku"]}]

    fallback_model_group, _ = get_fallback_model_group(fallbacks=fallbacks, model_group="openai/gpt-4o")

    assert fallback_model_group is None


def test_get_fallback_model_group_never_resolves_a_provider_without_a_prefixed_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An alias-style group name has no provider, and resolving it prints the SDK's provider-list banner,
    so the lookup only infers a provider when some key is spelled <provider>/<group>."""

    resolver: Final = MagicMock(return_value=("my-alias", "openai", None, None))
    monkeypatch.setattr(get_llm_provider_logic, "get_llm_provider", resolver)
    fallbacks: Final = [{"gpt-5.5-pro": ["claude-sonnet-4-6"]}, {"*": ["gpt-5.5-mini"]}]

    assert get_fallback_model_group(fallbacks=fallbacks, model_group="my-alias") == (["gpt-5.5-mini"], 1)
    resolver.assert_not_called()


def test_mid_stream_fallback_snapshot_kwargs_restores_the_popped_lists_and_shares_the_buckets():
    controls: Final = MidStreamFallbackControls(
        MappingProxyType({"fallbacks": [{"primary": ["backup"]}], "context_window_fallbacks": None})
    )
    metadata: Final = {"model_group": "primary"}
    kwargs: Final = {"messages": [{"role": "user", "content": "hi"}], "stream": True, "metadata": metadata}

    snapshot: Final = mid_stream_fallback_snapshot_kwargs(model="primary", controls=controls, kwargs=kwargs)

    assert snapshot == {
        **kwargs,
        "fallbacks": [{"primary": ["backup"]}],
        "context_window_fallbacks": None,
        MID_STREAM_FALLBACK_CONTROLS_KEY: controls,
        "model": "primary",
    }
    assert snapshot["metadata"] is metadata
    assert "fallbacks" not in kwargs

    bare: Final = mid_stream_fallback_snapshot_kwargs(model="primary", controls=None, kwargs=kwargs)
    assert "fallbacks" not in bare
    assert bare[MID_STREAM_FALLBACK_CONTROLS_KEY] == MidStreamFallbackControls(MappingProxyType({}))


def test_mid_stream_retry_kwargs_strips_what_the_retry_wrapper_pops_and_keeps_the_controls_carrier():
    def generic_function(**kwargs) -> None:
        return None

    def attempt(**kwargs) -> None:
        return None

    controls = MidStreamFallbackControls(MappingProxyType({"num_retries": 3}))
    litellm_metadata = {"model_group": "glm"}
    hop_kwargs = {
        "model": "glm",
        "original_generic_function": generic_function,
        "original_function": attempt,
        "fallbacks": [{"glm": ["fb"]}],
        "context_window_fallbacks": [],
        "content_policy_fallbacks": [],
        "num_retries": 3,
        "model_group_retry_policy": {},
        "stream": True,
        "litellm_metadata": litellm_metadata,
        MID_STREAM_FALLBACK_CONTROLS_KEY: controls,
    }

    retry_kwargs = mid_stream_retry_kwargs(hop_kwargs)

    assert retry_kwargs == {
        "model": "glm",
        "original_generic_function": generic_function,
        "stream": True,
        "litellm_metadata": litellm_metadata,
        MID_STREAM_FALLBACK_CONTROLS_KEY: controls,
    }
    assert retry_kwargs["litellm_metadata"] is litellm_metadata


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        pytest.param({"litellm_metadata": {"attempted_retries": 2}, "metadata": {"attempted_retries": 5}}, 2, id="litellm_metadata-wins"),
        pytest.param({"metadata": {"attempted_retries": 1}}, 1, id="metadata-bucket"),
        pytest.param({"litellm_metadata": {"attempted_retries": "2"}}, 0, id="string-is-not-a-count"),
        pytest.param({"litellm_metadata": {"attempted_retries": -1}}, 0, id="negative-is-not-a-count"),
        pytest.param({"litellm_metadata": {}}, 0, id="unstamped"),
        pytest.param({}, 0, id="no-bucket"),
    ],
)
def test_attempted_retries_for_request_reads_the_request_bucket(kwargs, expected):
    assert attempted_retries_for_request(kwargs) == expected


def test_record_retry_attempt_stamps_the_bucket_the_retry_wrapper_reads():
    kwargs = {"litellm_metadata": {"attempted_retries": 0, "max_retries": 2}, "metadata": {}}

    record_retry_attempt(kwargs, attempted_retries=1, max_retries=2)

    assert kwargs["litellm_metadata"] == {"attempted_retries": 1, "max_retries": 2}
    assert kwargs["metadata"] == {}
    assert attempted_retries_for_request(kwargs) == 1
    assert committed_retry_budget_for_request(kwargs) == 2


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        pytest.param({"litellm_metadata": {"attempted_retries": 1, "max_retries": 3}}, 3, id="committed-by-a-retry"),
        pytest.param({"litellm_metadata": {"attempted_retries": 0, "max_retries": 3}}, None, id="stamped-before-any-retry"),
        pytest.param({"litellm_metadata": {"attempted_retries": 1, "max_retries": "3"}}, None, id="string-is-not-a-budget"),
        pytest.param({"litellm_metadata": {"attempted_retries": 1}}, None, id="no-budget"),
        pytest.param({}, None, id="no-bucket"),
    ],
)
def test_committed_retry_budget_for_request_is_the_budget_a_retry_stamped(kwargs, expected):
    assert committed_retry_budget_for_request(kwargs) == expected


def test_carry_over_routed_deployment_copies_model_info_into_the_snapshot():
    live_kwargs = {"litellm_metadata": {"model_info": {"id": "dep-1"}, "deployment": "anthropic/glm-a"}}
    snapshot = {"litellm_metadata": {"model_group": "glm"}}

    carry_over_routed_deployment(live_kwargs=live_kwargs, snapshot=snapshot)

    assert snapshot["litellm_metadata"] == {"model_group": "glm", "model_info": {"id": "dep-1"}}
    assert snapshot["litellm_metadata"]["model_info"] is not live_kwargs["litellm_metadata"]["model_info"]
    assert routed_deployment_id(snapshot) == "dep-1"


def test_carry_over_routed_deployment_leaves_a_snapshot_without_a_bucket_alone():
    snapshot = {"model": "glm"}

    carry_over_routed_deployment(live_kwargs={"litellm_metadata": {"model_info": {"id": "dep-1"}}}, snapshot=snapshot)

    assert snapshot == {"model": "glm"}
    assert routed_deployment_id(snapshot) is None


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)

@pytest.fixture(scope="function")
def setup_and_teardown():
    """
    This fixture reloads litellm before every function. To speed up testing by removing callbacks being chained.
    """
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    importlib.reload(litellm)
    try:
        if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
            importlib.reload(litellm.proxy.proxy_server)
    except Exception:
        pass
    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    loop.close()
    asyncio.set_event_loop(None)

REFUSAL_RESPONSE: dict[str, Any] = {
    "id": "msg_refusal",
    "type": "message",
    "role": "assistant",
    "model": "claude-fable-5",
    "content": [],
    "stop_reason": "refusal",
    "stop_sequence": None,
    "stop_details": {"category": "cyber", "explanation": "flagged"},
    "usage": {"input_tokens": 25, "output_tokens": 1},
}

PLAIN_REFUSAL_RESPONSE: dict[str, Any] = {k: v for k, v in REFUSAL_RESPONSE.items() if k != "stop_details"}

OK_RESPONSE: dict[str, Any] = {
    "id": "msg_ok",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5",
    "content": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 25, "output_tokens": 2},
}

def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

REFUSAL_STREAM_FRAMES: tuple[bytes, ...] = (
    _sse("message_start", {"type": "message_start", "message": {**REFUSAL_RESPONSE, "stop_reason": None}}),
    _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "refusal", "stop_details": {"category": "cyber"}},
            "usage": {"output_tokens": 1},
        },
    ),
    _sse("message_stop", {"type": "message_stop"}),
)

OK_STREAM_FRAMES: tuple[bytes, ...] = (
    _sse("message_start", {"type": "message_start", "message": {**OK_RESPONSE, "stop_reason": None}}),
    _sse(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}},
    ),
    _sse("message_stop", {"type": "message_stop"}),
)

def _split_frames_mid_data_line(frames: tuple[bytes, ...]) -> tuple[bytes, ...]:
    """Split each frame's data line in half, modeling a transport chunk boundary."""
    return tuple(part for frame in frames for part in (frame[: len(frame) // 2], frame[len(frame) // 2 :]))

class _FrameStream(httpx.AsyncByteStream):
    def __init__(self, frames: tuple[bytes, ...]) -> None:
        self._frames = frames

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for frame in self._frames:
            yield frame

    async def aclose(self) -> None:
        return None

class FakeAnthropicUpstream:
    """Intercepts the third-party transport (httpx.AsyncClient.send): refuses on fable
    models, answers on others. The router deliberately does not forward caller-injected
    clients, so the transport is the seam that exercises the real litellm pipeline."""

    def __init__(
        self,
        refusal_body: dict[str, Any] = REFUSAL_RESPONSE,
        refusal_frames: tuple[bytes, ...] = REFUSAL_STREAM_FRAMES,
    ) -> None:
        self.refusal_body = refusal_body
        self.refusal_frames = refusal_frames
        self.calls: list[str] = []
        self.bodies: list[dict[str, Any]] = []

    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        model = body.get("model", "")
        self.calls.append(model)
        self.bodies.append(body)
        refuses = "fable" in model
        if body.get("stream"):
            frames = self.refusal_frames if refuses else OK_STREAM_FRAMES
            return httpx.Response(
                200,
                stream=_FrameStream(frames),
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        return httpx.Response(200, json=self.refusal_body if refuses else OK_RESPONSE, request=request)

    def install(self):
        async def _send(_client: httpx.AsyncClient, request: httpx.Request, **kwargs: Any) -> httpx.Response:
            return await self.send(request, **kwargs)

        return patch("httpx.AsyncClient.send", new=_send)

FABLE_TIER = {
    "model_name": "fable-tier",
    "litellm_params": {"model": "anthropic/claude-fable-5", "api_key": "sk-test"},
}

OPUS_TARGET = {
    "model_name": "opus-target",
    "litellm_params": {"model": "anthropic/claude-opus-5", "api_key": "sk-test"},
}

def _router(content_policy_fallbacks: list | None) -> Router:
    return Router(model_list=[FABLE_TIER, OPUS_TARGET], content_policy_fallbacks=content_policy_fallbacks)

async def _collect(stream: AsyncIterator[bytes]) -> bytes:
    return b"".join([chunk async for chunk in stream])

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
async def test_non_streaming_refusal_with_fallback_row_returns_fallback_response():
    fake = FakeAnthropicUpstream()
    router = _router(content_policy_fallbacks=[{"fable-tier": ["opus-target"]}])

    with fake.install():
        response = await router.aanthropic_messages(
            model="fable-tier", max_tokens=16, messages=[{"role": "user", "content": "hi"}]
        )

    assert response["stop_reason"] == "end_turn"
    assert response["id"] == "msg_ok"
    assert len(fake.calls) == 2
    assert "claude-opus-5" in fake.calls[1]

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_policy_fallbacks, upstream_body",
    [
        (None, REFUSAL_RESPONSE),
        ([{"unrelated-group": ["opus-target"]}], REFUSAL_RESPONSE),
        ([{"fable-tier": ["opus-target"]}], PLAIN_REFUSAL_RESPONSE),
    ],
    ids=["nothing-configured", "row-for-other-group", "refusal-without-stop-details"],
)
async def test_non_streaming_refusal_passes_through_untouched(content_policy_fallbacks, upstream_body):
    fake = FakeAnthropicUpstream(refusal_body=upstream_body)
    router = _router(content_policy_fallbacks=content_policy_fallbacks)

    with fake.install():
        response = await router.aanthropic_messages(
            model="fable-tier", max_tokens=16, messages=[{"role": "user", "content": "hi"}]
        )

    assert response["stop_reason"] == "refusal"
    assert response.get("stop_details") == upstream_body.get("stop_details")
    assert len(fake.calls) == 1

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
async def test_streaming_refusal_with_fallback_row_streams_fallback_frames():
    fake = FakeAnthropicUpstream()
    router = _router(content_policy_fallbacks=[{"fable-tier": ["opus-target"]}])

    with fake.install():
        stream = await router.aanthropic_messages(
            model="fable-tier", max_tokens=16, stream=True, messages=[{"role": "user", "content": "hi"}]
        )
        body = await _collect(stream)

    assert b'"refusal"' not in body
    assert b"text_delta" in body
    assert len(fake.calls) == 2

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
async def test_streaming_refusal_split_across_chunks_still_falls_back():
    fake = FakeAnthropicUpstream(refusal_frames=_split_frames_mid_data_line(REFUSAL_STREAM_FRAMES))
    router = _router(content_policy_fallbacks=[{"fable-tier": ["opus-target"]}])

    with fake.install():
        stream = await router.aanthropic_messages(
            model="fable-tier", max_tokens=16, stream=True, messages=[{"role": "user", "content": "hi"}]
        )
        body = await _collect(stream)

    assert b'"refusal"' not in body
    assert b"text_delta" in body
    assert len(fake.calls) == 2

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
async def test_streaming_refusal_without_fallback_row_passes_frames_through():
    fake = FakeAnthropicUpstream()
    router = _router(content_policy_fallbacks=None)

    with fake.install():
        stream = await router.aanthropic_messages(
            model="fable-tier", max_tokens=16, stream=True, messages=[{"role": "user", "content": "hi"}]
        )
        body = await _collect(stream)

    assert b'"stop_reason": "refusal"' in body
    assert b"stop_details" in body
    assert len(fake.calls) == 1

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
async def test_streaming_refusal_on_routed_tier_matches_tier_keyed_row_without_inbound_metadata():
    """The pre-routing hook's tier stamp must reach the mid-stream fallback lookup even when the
    request carries no metadata bucket at all (the snapshot is taken before the request runs)."""
    fake = FakeAnthropicUpstream()
    smart_router = {
        "model_name": "smart-router",
        "litellm_params": {
            "model": "auto_router/complexity_router",
            "complexity_router_config": {
                "tiers": {"SIMPLE": "fable-tier", "MEDIUM": "fable-tier", "COMPLEX": "fable-tier"}
            },
            "complexity_router_default_model": "fable-tier",
        },
        "model_info": {"id": "router-1", "db_model": True},
    }
    router = Router(
        model_list=[FABLE_TIER, OPUS_TARGET, smart_router],
        content_policy_fallbacks=[{"fable-tier": ["opus-target"]}],
        ignore_invalid_deployments=True,
    )

    with fake.install():
        stream = await router.aanthropic_messages(
            model="smart-router", max_tokens=16, stream=True, messages=[{"role": "user", "content": "hi"}]
        )
        body = await _collect(stream)

    assert b'"refusal"' not in body
    assert b"text_delta" in body
    assert len(fake.calls) == 2

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
async def test_caller_forged_tier_stamp_cannot_pick_the_streaming_fallback_chain():
    fake = FakeAnthropicUpstream()
    router = _router(content_policy_fallbacks=[{"forged-tier": ["opus-target"]}])

    with fake.install():
        stream = await router.aanthropic_messages(
            model="fable-tier",
            max_tokens=16,
            stream=True,
            messages=[{"role": "user", "content": "hi"}],
            litellm_metadata={PRE_ROUTING_SELECTED_MODEL_KEY: "forged-tier"},
        )
        body = await _collect(stream)

    assert b'"stop_reason": "refusal"' in body
    assert len(fake.calls) == 1

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
async def test_tier_stamp_never_reaches_provider_bound_metadata():
    """On /v1/messages the top-level metadata dict is Anthropic's own request field, so the
    routed-tier stamp must never appear in any upstream body even when the client sends one."""
    fake = FakeAnthropicUpstream()
    smart_router = {
        "model_name": "smart-router",
        "litellm_params": {
            "model": "auto_router/complexity_router",
            "complexity_router_config": {
                "tiers": {"SIMPLE": "fable-tier", "MEDIUM": "fable-tier", "COMPLEX": "fable-tier"}
            },
            "complexity_router_default_model": "fable-tier",
        },
        "model_info": {"id": "router-1", "db_model": True},
    }
    router = Router(
        model_list=[FABLE_TIER, OPUS_TARGET, smart_router],
        content_policy_fallbacks=[{"fable-tier": ["opus-target"]}],
        ignore_invalid_deployments=True,
    )

    with fake.install():
        response = await router.aanthropic_messages(
            model="smart-router",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            metadata={"user_id": "u1"},
        )

    assert response["stop_reason"] == "end_turn"
    assert len(fake.bodies) == 2
    for body in fake.bodies:
        assert body.get("metadata") == {"user_id": "u1"}

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_record_pre_routing_selection_writes_only_the_internal_bucket():
    """The Anthropic request's own metadata field must never carry the tier stamp."""
    kwargs = {"metadata": {"user_id": "u1"}, "litellm_metadata": {}}

    record_pre_routing_selection(kwargs, "tier-x")

    assert kwargs["litellm_metadata"] == {PRE_ROUTING_SELECTED_MODEL_KEY: "tier-x"}
    assert kwargs["metadata"] == {"user_id": "u1"}

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
async def test_generic_only_row_recovers_safeguard_refusal(stream):
    """With no content-policy list configured, a generic fallback row covers safeguard refusals,
    so the dashboard's generic fallbacks work without config-only content_policy rows."""
    fake = FakeAnthropicUpstream()
    router = Router(model_list=[FABLE_TIER, OPUS_TARGET], fallbacks=[{"fable-tier": ["opus-target"]}])

    with fake.install():
        response = await router.aanthropic_messages(
            model="fable-tier", max_tokens=16, stream=stream, messages=[{"role": "user", "content": "hi"}]
        )
        body = await _collect(response) if stream else response

    if stream:
        assert b'"refusal"' not in body
        assert b"text_delta" in body
    else:
        assert body["stop_reason"] == "end_turn"
    assert len(fake.calls) == 2
    assert "claude-opus-5" in fake.calls[1]

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
async def test_configured_content_policy_list_stays_authoritative_over_generic_rows():
    fake = FakeAnthropicUpstream()
    router = Router(
        model_list=[FABLE_TIER, OPUS_TARGET],
        fallbacks=[{"fable-tier": ["opus-target"]}],
        content_policy_fallbacks=[{"unrelated-group": ["opus-target"]}],
    )

    with fake.install():
        response = await router.aanthropic_messages(
            model="fable-tier", max_tokens=16, messages=[{"role": "user", "content": "hi"}]
        )

    assert response["stop_reason"] == "refusal"
    assert len(fake.calls) == 1

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_refusal_fallback_available_arms_on_generic_rows_only_without_content_policy():
    router = Router(model_list=[FABLE_TIER, OPUS_TARGET], fallbacks=[{"tier-group": ["opus-target"]}])
    stamped = {"litellm_metadata": {PRE_ROUTING_SELECTED_MODEL_KEY: "tier-group"}}

    assert router._refusal_fallback_available("router-group", stamped) is True
    assert router._refusal_fallback_available("router-group", {}) is False
    assert router._refusal_fallback_available("router-group", {"content_policy_fallbacks": [{"other": ["x"]}]}) is False

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_chat_content_filter_gate_unchanged_by_generic_rows():
    """The generic-row arming is scoped to /v1/messages safeguard refusals; the chat surface's
    content_filter gate keeps its long-standing content-policy-only semantics."""
    from litellm.types.utils import Choices, ModelResponse

    router = Router(model_list=[FABLE_TIER, OPUS_TARGET], fallbacks=[{"fable-tier": ["opus-target"]}])
    response = ModelResponse(choices=[Choices(finish_reason="content_filter")])

    assert router._should_raise_content_policy_error(model="fable-tier", response=response, kwargs={}) is False

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
async def test_disable_fallbacks_returns_the_refusal_instead_of_raising(stream):
    """A request that opted out of fallbacks must receive the provider's refusal response,
    never a ContentPolicyViolationError the dispatcher refuses to recover."""
    fake = FakeAnthropicUpstream()
    router = Router(model_list=[FABLE_TIER, OPUS_TARGET], fallbacks=[{"fable-tier": ["opus-target"]}])

    with fake.install():
        response = await router.aanthropic_messages(
            model="fable-tier",
            max_tokens=16,
            stream=stream,
            disable_fallbacks=True,
            messages=[{"role": "user", "content": "hi"}],
        )
        body = await _collect(response) if stream else response

    if stream:
        assert b'"stop_reason": "refusal"' in body
    else:
        assert body["stop_reason"] == "refusal"
    assert len(fake.calls) == 1

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
@pytest.mark.asyncio
async def test_disable_fallbacks_beats_a_content_policy_row_too():
    fake = FakeAnthropicUpstream()
    router = Router(
        model_list=[FABLE_TIER, OPUS_TARGET],
        content_policy_fallbacks=[{"fable-tier": ["opus-target"]}],
    )

    with fake.install():
        response = await router.aanthropic_messages(
            model="fable-tier",
            max_tokens=16,
            disable_fallbacks=True,
            messages=[{"role": "user", "content": "hi"}],
        )

    assert response["stop_reason"] == "refusal"
    assert len(fake.calls) == 1

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_refusal_gate_keys_on_pre_routing_tier_stamp():
    router = _router(content_policy_fallbacks=[{"tier-group": ["opus-target"]}])

    def anthropic_messages(**kwargs: Any) -> None:
        return None

    refusal_kwargs = {"litellm_metadata": {PRE_ROUTING_SELECTED_MODEL_KEY: "tier-group"}}
    assert (
        router._should_raise_anthropic_refusal_error(
            model="router-group",
            original_generic_function=anthropic_messages,
            response=dict(REFUSAL_RESPONSE),
            kwargs=refusal_kwargs,
        )
        is True
    )
    assert (
        router._should_raise_anthropic_refusal_error(
            model="router-group",
            original_generic_function=anthropic_messages,
            response=dict(REFUSAL_RESPONSE),
            kwargs={},
        )
        is False
    )

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_has_content_policy_fallback_default_fallbacks_arm():
    router = Router(model_list=[OPUS_TARGET], fallbacks=[{"*": ["opus-target"]}])

    assert router._has_content_policy_fallback("any-group", {}) is True
    assert router._has_content_policy_fallback("any-group", {"content_policy_fallbacks": [{"other": ["x"]}]}) is False

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_get_fallback_model_group_for_lookup_groups_orders_tier_before_requested():
    router = _router(content_policy_fallbacks=None)
    fallbacks = [{"tier1": ["backup-a"]}, {"smart-router": ["backup-b"]}]

    assert router._get_fallback_model_group_for_lookup_groups(
        fallbacks=fallbacks, lookup_groups=("tier1", "smart-router")
    ) == ["backup-a"]
    assert router._get_fallback_model_group_for_lookup_groups(
        fallbacks=fallbacks, lookup_groups=("tier9", "smart-router")
    ) == ["backup-b"]
    assert router._get_fallback_model_group_for_lookup_groups(fallbacks=fallbacks, lookup_groups=()) is None

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_refusal_gate_ignores_other_generic_call_types():
    router = _router(content_policy_fallbacks=[{"fable-tier": ["opus-target"]}])

    def aresponses(**kwargs: Any) -> None:
        return None

    assert (
        router._should_raise_anthropic_refusal_error(
            model="fable-tier",
            original_generic_function=aresponses,
            response=dict(REFUSAL_RESPONSE),
            kwargs={},
        )
        is False
    )


class FallbackEventLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.success_fallback_events: list[tuple[str, dict[str, object], Exception]] = []
        self.failure_fallback_events: list[tuple[str, dict[str, object], Exception]] = []

    async def log_success_fallback_event(
        self,
        original_model_group: str,
        kwargs: dict[str, object],
        original_exception: Exception,
    ) -> None:
        self.success_fallback_events.append((original_model_group, kwargs, original_exception))

    async def log_failure_fallback_event(
        self,
        original_model_group: str,
        kwargs: dict[str, object],
        original_exception: Exception,
    ) -> None:
        self.failure_fallback_events.append((original_model_group, kwargs, original_exception))


def _create_fallback_event_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "test-key"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "test-key"},
            },
        ],
        fallbacks=[{"gpt-3.5-turbo": ["gpt-4"]}],
    )


@pytest.mark.parametrize(
    "function_name",
    ["_acompletion", "_atext_completion", "_aembedding"],
)
@pytest.mark.asyncio
async def test_run_async_fallback(function_name):
    """
    Basic test - given a list of fallback models, run the original function with the fallback models
    """
    router = create_test_router()
    original_function = getattr(router, function_name)

    litellm.set_verbose = True
    fallback_model_group = ["gpt-4"]
    original_model_group = "gpt-3.5-turbo"
    original_exception = litellm.exceptions.InternalServerError(
        message="Simulated error",
        llm_provider="openai",
        model="gpt-3.5-turbo",
    )

    request_kwargs = {
        "mock_response": "hello this is a test for run_async_fallback",
        "metadata": {"previous_models": ["gpt-3.5-turbo"]},
    }

    if function_name == "_aembedding":
        request_kwargs["input"] = "hello this is a test for run_async_fallback"
    elif function_name == "_atext_completion":
        request_kwargs["prompt"] = "hello this is a test for run_async_fallback"
    elif function_name == "_acompletion":
        request_kwargs["messages"] = [{"role": "user", "content": "Hello, world!"}]

    result = await run_async_fallback(
        litellm_router=router,
        original_function=original_function,
        num_retries=1,
        fallback_model_group=fallback_model_group,
        original_model_group=original_model_group,
        original_exception=original_exception,
        max_fallbacks=5,
        fallback_depth=0,
        **request_kwargs,
    )

    assert result is not None

    if function_name == "_acompletion":
        assert isinstance(result, litellm.ModelResponse)
    elif function_name == "_atext_completion":
        assert isinstance(result, litellm.TextCompletionResponse)
    elif function_name == "_aembedding":
        assert isinstance(result, litellm.EmbeddingResponse)


@pytest.mark.asyncio
async def test_log_success_fallback_event():
    """
    Tests that successful fallback events are logged correctly
    """
    original_model_group = "gpt-3.5-turbo"
    kwargs = {"messages": [{"role": "user", "content": "Hello, world!"}]}
    original_exception = litellm.exceptions.InternalServerError(
        message="Simulated error",
        llm_provider="openai",
        model="gpt-3.5-turbo",
    )

    logger = CustomTestLogger()
    litellm.callbacks = [logger]

    # This test mainly checks if the function runs without errors
    await log_success_fallback_event(original_model_group, kwargs, original_exception)

    await asyncio.sleep(0.5)
    assert len(logger.success_fallback_events) == 1
    assert len(logger.failure_fallback_events) == 0
    assert logger.success_fallback_events[0] == (
        original_model_group,
        kwargs,
        original_exception,
    )


@pytest.mark.asyncio
async def test_log_failure_fallback_event():
    """
    Tests that failed fallback events are logged correctly
    """
    original_model_group = "gpt-3.5-turbo"
    kwargs = {"messages": [{"role": "user", "content": "Hello, world!"}]}
    original_exception = litellm.exceptions.InternalServerError(
        message="Simulated error",
        llm_provider="openai",
        model="gpt-3.5-turbo",
    )

    logger = CustomTestLogger()
    litellm.callbacks = [logger]

    # This test mainly checks if the function runs without errors
    await log_failure_fallback_event(original_model_group, kwargs, original_exception)

    await asyncio.sleep(0.5)

    assert len(logger.failure_fallback_events) == 1
    assert len(logger.success_fallback_events) == 0
    assert logger.failure_fallback_events[0] == (
        original_model_group,
        kwargs,
        original_exception,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("function_name", ["_acompletion", "_atext_completion"])
async def test_failed_fallbacks_raise_most_recent_exception(function_name):
    """
    Tests that if all fallbacks fail, the most recent occuring exception is raised

    meaning the exception from the last fallback model is raised
    """
    router = create_test_router()
    original_function = getattr(router, function_name)

    fallback_model_group = ["gpt-4"]
    original_model_group = "gpt-3.5-turbo"
    original_exception = litellm.exceptions.InternalServerError(
        message="Simulated error",
        llm_provider="openai",
        model="gpt-3.5-turbo",
    )

    request_kwargs: Dict[str, Any] = {
        "metadata": {"previous_models": ["gpt-3.5-turbo"]}
    }

    if function_name == "_aembedding":
        request_kwargs["input"] = "hello this is a test for run_async_fallback"
    elif function_name == "_atext_completion":
        request_kwargs["prompt"] = "hello this is a test for run_async_fallback"
    elif function_name == "_acompletion":
        request_kwargs["messages"] = [{"role": "user", "content": "Hello, world!"}]

    with pytest.raises(litellm.exceptions.RateLimitError):
        await run_async_fallback(
            litellm_router=router,
            original_function=original_function,
            num_retries=1,
            fallback_model_group=fallback_model_group,
            original_model_group=original_model_group,
            original_exception=original_exception,
            mock_response="litellm.RateLimitError",
            max_fallbacks=5,
            fallback_depth=0,
            **request_kwargs,
        )


def create_test_router():
    return Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ],
        fallbacks=[{"gpt-3.5-turbo": ["gpt-4"]}],
    )


class CustomTestLogger(CustomLogger):
    def __init__(self):
        super().__init__()
        self.success_fallback_events = []
        self.failure_fallback_events = []

    async def log_success_fallback_event(
        self, original_model_group, kwargs, original_exception
    ):
        print(
            "in log_success_fallback_event for original_model_group: ",
            original_model_group,
        )
        self.success_fallback_events.append(
            (original_model_group, kwargs, original_exception)
        )

    async def log_failure_fallback_event(
        self, original_model_group, kwargs, original_exception
    ):
        print(
            "in log_failure_fallback_event for original_model_group: ",
            original_model_group,
        )
        self.failure_fallback_events.append(
            (original_model_group, kwargs, original_exception)
        )


def create_test_router_2():
    return Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                    "api_key": "very-fake-key",
                },
            },
            {
                "model_name": "fake-openai-endpoint-2",
                "litellm_params": {
                    "model": "openai/fake-openai-endpoint-2",
                    "api_key": "working-key-since-this-is-fake-endpoint",
                    "api_base": FAKE_OPENAI_API_BASE,
                },
            },
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("function_name", ["_acompletion", "_atext_completion"])
async def test_multiple_fallbacks(function_name, respx_mock: respx.MockRouter, monkeypatch):
    """
    Tests that if multiple fallbacks passed:
    - fallback 1 = bad configured deployment / failing endpoint
    - fallback 2 = working deployment / working endpoint

    Assert that:
    - a success response is received from the working endpoint (fallback 2)
    """
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    invalid_key = httpx.Response(
        401,
        json={
            "error": {
                "message": "Incorrect API key provided",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            }
        },
    )
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(return_value=invalid_key)
    respx_mock.post("https://api.openai.com/v1/completions").mock(return_value=invalid_key)
    respx_mock.post(f"{FAKE_OPENAI_API_BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "fake-openai-endpoint-2",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )
    )
    respx_mock.post(f"{FAKE_OPENAI_API_BASE}/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "cmpl-fake",
                "object": "text_completion",
                "created": 1700000000,
                "model": "fake-openai-endpoint-2",
                "choices": [{"index": 0, "text": "hi", "logprobs": None, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )
    )
    router_2 = create_test_router_2()
    original_function = getattr(router_2, function_name)

    fallback_model_group = ["gpt-4", "fake-openai-endpoint-2"]
    original_model_group = "gpt-3.5-turbo"
    original_exception = Exception("Simulated error")

    request_kwargs: dict[str, object] = {"metadata": {"previous_models": ["gpt-3.5-turbo"]}}

    if function_name == "_aembedding":
        request_kwargs["input"] = "hello this is a test for run_async_fallback"
    elif function_name == "_atext_completion":
        request_kwargs["prompt"] = "hello this is a test for run_async_fallback"
    elif function_name == "_acompletion":
        request_kwargs["messages"] = [{"role": "user", "content": "Hello, world!"}]

    result = await run_async_fallback(
        litellm_router=router_2,
        original_function=original_function,
        num_retries=1,
        fallback_model_group=fallback_model_group,
        original_model_group=original_model_group,
        original_exception=original_exception,
        max_fallbacks=5,
        fallback_depth=0,
        **request_kwargs,
    )

    print(result)

    print(result._hidden_params)

    assert result._hidden_params["api_base"] == FAKE_OPENAI_API_BASE
