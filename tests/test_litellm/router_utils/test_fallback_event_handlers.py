import json
from typing import NoReturn
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.router_utils.cooldown_handlers import mark_advisor_orchestration_failure
from litellm.router_utils.fallback_event_handlers import (
    AttemptedFallbackTargets,
    _trigger_cooldown_for_failed_deployment,
    fallback_attempt_key,
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


async def _acreate_file(*args: object, **kwargs: object) -> NoReturn:
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


class RecordingFailRouter:
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
            fallback_model_group=[
                {"model": "already-attempted", "messages": [{"role": "user", "content": "shorter"}]}
            ],
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

    def test_skips_client_side_timeout_408(self):
        """The proxy's x-litellm-timeout header lets a caller set an arbitrarily short
        timeout, which litellm.Timeout reports as status 408 regardless of the
        deployment's actual health. Without this guard, a caller could force a 408 on
        every deployment in the fallback chain from a single request."""
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.Timeout(message="timeout", model="gpt-4", llm_provider="openai")
        exc.failed_deployment_id = "fallback-deployment"

        with (
            patch("litellm.router_utils.fallback_event_handlers._set_cooldown_deployments") as mock_set_cooldown,
            patch(
                "litellm.router_utils.fallback_event_handlers.increment_deployment_failures_for_current_minute"
            ) as mock_increment,
        ):
            _trigger_cooldown_for_failed_deployment(
                litellm_router=mock_router,
                kwargs={"client_side_timeout": True},
                exception=exc,
            )

            mock_set_cooldown.assert_not_called()
            mock_increment.assert_not_called()

    def test_still_cools_down_408_without_client_side_timeout_flag(self):
        """The client-side-timeout guard is scoped to caller-supplied timeouts only: a
        408 that did not come from x-litellm-timeout (no client_side_timeout in kwargs)
        must still cool down the deployment as before."""
        mock_router = MagicMock()
        mock_router.cooldown_time = 60.0
        mock_router.get_model_info.return_value = None

        exc = litellm.Timeout(message="timeout", model="gpt-4", llm_provider="openai")
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


@pytest.mark.asyncio
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
