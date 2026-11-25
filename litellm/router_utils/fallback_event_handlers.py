from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import partial, reduce
from types import MappingProxyType
from typing import Final, cast  # noqa: TID251  # the router passes the request kwargs as an untyped dict

from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.legacy import Admit, hook_subscribers, kwargs_after, run_subscribers
from litellm.integrations.custom_logger_v2.logger import CustomLoggerV2
from litellm.litellm_core_utils.core_helpers import get_metadata_variable_name_from_kwargs
from litellm.litellm_core_utils.get_llm_provider_logic import inferred_provider
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.openai.common_utils import create_may_already_exist
from litellm.types.integrations.custom_logger_v2 import Outcome
from litellm.types.router import (
    FallbackSkipReason,
    LiteLLMParamsTypedDict,
    SkippedFallbackTarget,
)

# Status codes a generic API call's caller-supplied resource id can trigger on its own
# (e.g. a nonexistent file/batch/thread id), independent of the selected deployment's health.
_REQUEST_SCOPED_STATUS_CODES: Final = frozenset((404,))
_NO_MODEL_CALL_DETAILS: Final[Mapping[str, object]] = MappingProxyType({})
_FALLBACK_KWARGS_ADAPTER: Final = TypeAdapter(dict[str, object])


@dataclass(slots=True)
class AttemptedFallbackTargets:
    """
    The fallback attempts a single request has already made.

    One instance is created on the first fallback hop and shared by reference for the rest
    of the walk, so an attempt made in one branch is not repeated in a sibling branch.
    Without it the walk enumerates paths rather than attempts: a fallback graph containing
    a cycle retries one deterministic failure once per path through the cycle, and a
    client-side fallback list is re-walked at every level of the recursion.
    """

    keys: frozenset[str] = frozenset()
    _skip_reasons: dict[str, FallbackSkipReason] = field(default_factory=dict)  # mutable-ok: O(1) per caller-sized list

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    def record(self, key: str) -> None:
        self.keys = self.keys | frozenset((key,))

    @property
    def skipped(self) -> tuple[SkippedFallbackTarget, ...]:
        return tuple(
            SkippedFallbackTarget(target=target, reason=reason) for target, reason in self._skip_reasons.items()
        )

    def record_skip(self, target: str, reason: FallbackSkipReason) -> None:
        self._skip_reasons.setdefault(target, reason)


def _check_stripped_model_group(model_group: str, fallback_key: str) -> bool:
    """
    Handles wildcard routing scenario

    where fallbacks set like:
    [{"gpt-3.5-turbo": ["claude-3-haiku"]}]

    but model_group is like:
    "openai/gpt-3.5-turbo"

    Returns:
    - True if the stripped model group == fallback_key
    """
    for provider in litellm.provider_list:
        if isinstance(provider, Enum):
            _provider = provider.value
        else:
            _provider = provider
        if model_group.startswith(f"{_provider}/"):
            stripped_model_group = model_group.replace(f"{_provider}/", "")
            if stripped_model_group == fallback_key:
                return True
    return False


def _provider_prefixed_model_group(model_group: str, fallback_keys: Sequence[str]) -> str | None:
    if "/" in model_group or not any(key.endswith(f"/{model_group}") for key in fallback_keys):
        return None
    provider: Final = inferred_provider(model_group)
    return f"{provider}/{model_group}" if provider else None


PRE_ROUTING_SELECTED_MODEL_KEY: Final = "pre_routing_selected_model"
_ROUTER_METADATA_BUCKETS: Final = ("metadata", "litellm_metadata")


def record_pre_routing_selection(request_kwargs: Mapping[str, object] | None, selected_model: str) -> None:
    """
    Remember which model a pre-routing hook picked, so fallback lookup can key off it.

    Fallback resolution runs on an outer kwargs dict that ``**kwargs`` already copied, so
    writing the model there is invisible by the time routing picks a tier. The metadata
    buckets are nested dicts shared by reference across those copies, which is how the
    router already carries values back up.

    The write goes through the proxy-internal bucket resolver, never into both buckets:
    on /v1/messages the top-level ``metadata`` dict is the provider's own request field,
    so a blanket write would forward the tier stamp upstream.
    """
    if request_kwargs is None:
        return
    bucket: Final = request_kwargs.get(get_metadata_variable_name_from_kwargs(request_kwargs))
    if isinstance(bucket, dict):
        bucket[PRE_ROUTING_SELECTED_MODEL_KEY] = selected_model


def clear_pre_routing_selection(request_kwargs: Mapping[str, object] | None) -> None:
    """
    Drop any selection the router did not make itself on this hop.

    The buckets carry whatever the caller sent, so an inbound value is the caller
    choosing a fallback chain rather than the router choosing a tier. A fallback hop
    also inherits the previous hop's selection, which would key its own failure off
    the tier that already failed. Clearing at the start of every hop leaves only a
    value the pre-routing hook wrote while routing that hop.
    """
    if request_kwargs is None:
        return
    for bucket in (request_kwargs.get(name) for name in _ROUTER_METADATA_BUCKETS):
        if isinstance(bucket, dict) and PRE_ROUTING_SELECTED_MODEL_KEY in bucket:
            del bucket[PRE_ROUTING_SELECTED_MODEL_KEY]


def get_pre_routing_selection(kwargs: Mapping[str, object]) -> str | None:
    """The model a pre-routing hook selected for this request, if one did."""
    buckets: Final = (kwargs.get(name) for name in _ROUTER_METADATA_BUCKETS)
    selections: Final = (bucket.get(PRE_ROUTING_SELECTED_MODEL_KEY) for bucket in buckets if isinstance(bucket, dict))
    return next((selected for selected in selections if isinstance(selected, str) and selected), None)


MID_STREAM_FALLBACK_CONTROLS_KEY: Final = "_mid_stream_fallback_controls"
_PER_REQUEST_FALLBACK_CONTROL_KEYS: Final = (
    "fallbacks",
    "context_window_fallbacks",
    "content_policy_fallbacks",
    "num_retries",
    "model_group_retry_policy",
)


@dataclass(frozen=True, slots=True)
class MidStreamFallbackControls:
    """
    The per-request fallback and retry overrides every streaming attempt must see again.

    async_function_with_retries pops them before the attempt function runs, so without this
    carrier a fallback hop's own mid-stream re-entry would fall back to the router-level settings.
    """

    overrides: Mapping[str, object]


_NO_FALLBACK_CONTROLS: Final = MidStreamFallbackControls(MappingProxyType({}))


_MID_STREAM_RETRY_STRIPPED_KEYS: Final = (*_PER_REQUEST_FALLBACK_CONTROL_KEYS, "original_function")
_MID_STREAM_RETRY_ATTEMPTED_KEY: Final = "attempted_retries"
_MID_STREAM_RETRY_BUDGET_KEY: Final = "max_retries"


def carry_superseded_attempt_usage(kwargs: Mapping[str, object]) -> None:
    """Carry the superseded stream's partial usage into billing if the request ultimately fails."""
    logging_obj: Final = kwargs.get("litellm_logging_obj")
    if isinstance(logging_obj, Logging):
        logging_obj.carry_failed_attempt_usage()


def _failure_callbacks_blocked_logging_object(  # pyright: ignore[reportUnusedFunction]  # Rust health callers are excluded from type checking
    kwargs: Mapping[str, object],
) -> Logging | None:
    logging_obj: Final = kwargs.get("litellm_logging_obj")
    if not isinstance(logging_obj, Logging):
        return None
    return logging_obj if logging_obj.failure_callbacks_blocked_for_attempt() else None


def _request_metadata_bucket(kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
    bucket: Final = kwargs.get(get_metadata_variable_name_from_kwargs(kwargs))
    return bucket if isinstance(bucket, Mapping) else None


def _routed_model_info(kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
    bucket: Final = _request_metadata_bucket(kwargs)
    model_info: Final = bucket.get("model_info") if bucket is not None else None
    return model_info if isinstance(model_info, Mapping) else None


def routed_deployment_id(kwargs: Mapping[str, object]) -> str | None:
    model_info: Final = _routed_model_info(kwargs)
    deployment_id: Final = model_info.get("id") if model_info is not None else None
    return deployment_id if isinstance(deployment_id, str) else None


DISABLE_FALLBACKS_METADATA_KEY: Final = "_disable_fallbacks"


def fallbacks_disabled_for_request(kwargs: Mapping[str, object]) -> bool:
    """True when this request opted out of fallbacks, read from the raw kwarg (pre-pop
    snapshots keep it) or the router-internal bucket the wrapper stamps after popping it."""
    if kwargs.get("disable_fallbacks") is True:
        return True
    buckets: Final = (kwargs.get(name) for name in _ROUTER_METADATA_BUCKETS)
    return any(isinstance(bucket, dict) and bucket.get(DISABLE_FALLBACKS_METADATA_KEY) is True for bucket in buckets)


def fallback_lookup_groups(kwargs: Mapping[str, object], model_group: str | None) -> tuple[str, ...]:
    """
    Ordered keys for resolving a fallback chain: the tier a pre-routing hook selected wins,
    then the routed group, then the requested group, then the group the request was
    originally for. The routed group differs when Claude Code session affinity remaps a
    subagent's concrete model to its bound router. The original group differs on a fallback
    hop that fails after `run_async_fallback` already returned its stream: the hop has no
    chain of its own, so it resumes the original group's chain, and `attempted_targets` keeps
    the entries already tried from being repeated.
    """
    raw_metadata: Final = kwargs.get(get_metadata_variable_name_from_kwargs(kwargs))
    metadata: Final[Mapping[object, object] | None] = raw_metadata if isinstance(raw_metadata, Mapping) else None
    routed_group_value: Final = metadata.get("model_group") if metadata is not None else None
    routed_group: Final = routed_group_value if isinstance(routed_group_value, str) else None
    original_group_value: Final = metadata.get("original_model_group") if metadata is not None else None
    original_group: Final = original_group_value if isinstance(original_group_value, str) else None
    ordered: Final = (get_pre_routing_selection(kwargs), routed_group, model_group, original_group)
    return tuple(dict.fromkeys(group for group in ordered if group))


def _resolved_a_specific_chain(
    fallbacks: Sequence[Mapping[str, list[str]]],  # mutable-ok: mirrors the fallbacks config shape
    result: tuple[list[str] | None, int | None],  # mutable-ok: mirrors get_fallback_model_group's contract
) -> bool:
    resolved, generic_idx = result
    if resolved is None:
        return False
    return generic_idx is None or resolved is not fallbacks[generic_idx]["*"]


def get_fallback_model_group_for_lookup_groups(
    fallbacks: Sequence[Mapping[str, list[str]]],  # mutable-ok: mirrors the fallbacks config shape
    lookup_groups: tuple[str, ...],
) -> tuple[list[str] | None, int | None]:  # mutable-ok: mirrors get_fallback_model_group's contract
    """
    First lookup group with a specifically-keyed chain wins; the generic "*" chain applies
    only after every group missed, so a catch-all cannot shadow a later group's own chain.
    """
    results: Final = tuple(get_fallback_model_group(fallbacks=fallbacks, model_group=group) for group in lookup_groups)
    specific: Final = next((result for result in results if _resolved_a_specific_chain(fallbacks, result)), None)
    if specific is not None:
        return specific
    return next((result for result in results if result[0] is not None), (None, None))


def get_fallback_model_group(
    fallbacks: Sequence[Mapping[str, list[str]]],  # mutable-ok: mirrors the fallbacks config shape
    model_group: str,
) -> tuple[list[str] | None, int | None]:  # mutable-ok: mirrors get_fallback_model_group's contract
    """
    Returns:
    - fallback_model_group: List[str] of fallback model groups. example: ["gpt-4", "gpt-3.5-turbo"]
    - generic_fallback_idx: int of the index of the generic fallback in the fallbacks list.

    Checks:
    - exact match
    - stripped model group match
    - provider-prefixed model group match
    - generic fallback
    """
    generic_fallback_idx: int | None = None
    stripped_model_fallback: list[str] | None = None
    fallback_model_group: list[str] | None = None
    fallback_keys: Final = tuple(next(iter(item)) for item in fallbacks if isinstance(item, dict) and item)
    prefixed_model_group: Final = _provider_prefixed_model_group(model_group, fallback_keys)
    ## check for specific model group-specific fallbacks
    for idx, item in enumerate(fallbacks):
        if isinstance(item, dict):
            if not item:
                continue  # an empty entry names no group: the next entries still apply
            fallback_key = next(iter(item))
            if fallback_key == model_group:  # check exact match
                fallback_model_group = item[model_group]
                break
            elif fallback_key == prefixed_model_group or _check_stripped_model_group(
                model_group=model_group, fallback_key=fallback_key
            ):  # check generic fallback
                stripped_model_fallback = item[fallback_key]
            elif fallback_key == "*":  # check generic fallback
                generic_fallback_idx = idx
        elif isinstance(item, str):
            fallback_model_group = [item]
    ## if none, check for generic fallback
    if fallback_model_group is None:
        if stripped_model_fallback is not None:
            fallback_model_group = stripped_model_fallback
        elif generic_fallback_idx is not None:
            fallback_model_group = fallbacks[generic_fallback_idx]["*"]

    return fallback_model_group, generic_fallback_idx


def _resolved_fallback_chain(entries: Sequence[object], lookup_groups: tuple[str, ...]) -> Sequence[object]:
    group_entries: Final[Sequence[Mapping[str, list[str]]]] = tuple(  # mutable-ok: mirrors the fallbacks config shape
        entry for entry in entries if isinstance(entry, Mapping) and entry
    )
    chain, _ = get_fallback_model_group_for_lookup_groups(fallbacks=group_entries, lookup_groups=lookup_groups)
    return chain if isinstance(chain, list) else ()


def continuation_fallback_targets(fallbacks: object, lookup_groups: tuple[str, ...]) -> Sequence[object]:
    entries: Final[Sequence[object]] = fallbacks if isinstance(fallbacks, list) else ()
    return (
        entries
        if check_non_standard_fallback_format(fallbacks=entries)
        else _resolved_fallback_chain(entries, lookup_groups)
    )


def fallback_target_overrides_request_params(target: object) -> bool:
    return isinstance(target, Mapping) and "model" in target and len(target) > 1


def fallback_target_name(target: object) -> str | None:
    name: Final = target.get("model") if isinstance(target, Mapping) else target
    return name if isinstance(name, str) else None


PROVIDER_SCOPED_RESOURCE_KEYS: Final = ("input_file_id", "training_file", "batch_id", "file_id", "fine_tuning_job_id")
PROVIDER_SCOPED_RESOURCE_FUNCTION_NAMES: Final = frozenset(
    {
        "_acreate_batch",
        "_acancel_batch",
        "acreate_fine_tuning_job",
        "acancel_fine_tuning_job",
        "aretrieve_fine_tuning_job",
        "afile_content",
        "afile_delete",
    }
)
PROVIDER_SCOPED_CREATION_FUNCTION_NAMES: Final = frozenset({"_acreate_file"})
NON_IDEMPOTENT_CREATE_FUNCTION_NAMES: Final = frozenset({"_acreate_file", "_acreate_batch"})


def is_create_that_may_already_exist(function_name: str, error: BaseException) -> bool:
    return function_name in NON_IDEMPOTENT_CREATE_FUNCTION_NAMES and create_may_already_exist(error)


def log_create_not_resent(model_group: str | None) -> None:
    verbose_router_logger.info(
        "Model group '%s' does not resend a failed create, because the provider may already have created it",
        model_group,
    )


def references_provider_scoped_resource(kwargs: Mapping[str, object]) -> bool:
    """
    True when a file, batch, or fine-tuning job operation names an id that only exists
    under one provider's credentials.

    Each of those ids lives in the account of the deployment that issued it. Handing it to
    a different model group asks a provider about an id it never issued, which costs an
    extra round trip that can only answer not-found. Generic calls dispatched through
    `Router._ageneric_api_call_with_fallbacks` carry the real handler in
    `original_generic_function`, so both slots are checked. Gating on the handler name
    keeps completion-style requests eligible for cross-group fallback even when a caller
    passes a stray extra body field that happens to share one of these key names.
    """
    handler_names: Final = tuple(
        getattr(kwargs.get(function_key), "__name__", None)
        for function_key in ("original_function", "original_generic_function")
    )
    if all(name not in PROVIDER_SCOPED_RESOURCE_FUNCTION_NAMES for name in handler_names):
        return False
    return any(kwargs.get(key) for key in PROVIDER_SCOPED_RESOURCE_KEYS)


def creates_provider_scoped_resource(kwargs: Mapping[str, object]) -> bool:
    """
    True when the request creates a resource that will live under one provider's credentials.

    A file uploaded for batches or fine-tuning is stored in the account of the deployment
    that handled it, and its id is only usable against the model group the caller named.
    Letting the upload fall back to a different model group silently stores the file with
    the wrong provider, and every later use of the returned id fails.
    """
    return getattr(kwargs.get("original_function"), "__name__", None) in PROVIDER_SCOPED_CREATION_FUNCTION_NAMES


async def log_success_fallback_event(
    original_model_group: str,
    kwargs: dict,  # mutable-ok: the request kwargs the custom loggers receive
    original_exception: Exception,
):
    await _log_fallback_event(
        True,
        original_model_group,
        cast(  # cast-ok: the request kwargs dict, keyed by parameter name
            dict[str, object], kwargs
        ),
        original_exception,
    )


async def log_failure_fallback_event(
    original_model_group: str,
    kwargs: dict,  # mutable-ok: the request kwargs the custom loggers receive
    original_exception: Exception,
):
    await _log_fallback_event(
        False,
        original_model_group,
        cast(  # cast-ok: the request kwargs dict, keyed by parameter name
            dict[str, object], kwargs
        ),
        original_exception,
    )


async def _log_fallback_event(
    succeeded: bool,
    original_model_group: str,
    kwargs: dict[str, object],  # mutable-ok: the request kwargs the custom loggers receive
    original_exception: Exception,
) -> None:
    callbacks: Final = reduce(_with_event_logger, litellm.logging_callback_manager.get_all_callbacks(), ())
    fields: Final = MappingProxyType(
        {
            Field.deprecated_kwargs: kwargs,
            Field.model: original_model_group,
            Field.error: original_exception,
            Field.outcome: Outcome.fallback_success if succeeded else Outcome.fallback_failure,
        }
    )
    subscribers: Final = hook_subscribers(Hook.router_event, callbacks, admit=Admit.every_logger)
    _ = kwargs_after(
        await run_subscribers(
            subscribers, fields, partial(_start_fallback_event, succeeded, original_model_group, original_exception)
        )
    )


def _with_event_logger(kept: tuple[object, ...], callback: object) -> tuple[object, ...]:
    if not isinstance(callback, CustomLogger | CustomLoggerV2) or callback in kept:
        return kept
    return (*kept, callback)


def _start_fallback_event(
    succeeded: bool, original_model_group: str, original_exception: Exception, callback: object, kwargs: object
) -> Awaitable[None] | None:
    if not isinstance(callback, CustomLogger):
        return None
    return _deliver_fallback_event(
        callback,
        succeeded,
        original_model_group,
        kwargs,  # pyright: ignore[reportArgumentType]  # the request kwargs as the router passed them
        original_exception,
    )


async def _deliver_fallback_event(
    callback: CustomLogger,
    succeeded: bool,
    original_model_group: str,
    kwargs: dict[str, object],  # mutable-ok: the request kwargs the custom loggers receive
    original_exception: Exception,
) -> None:
    try:
        if succeeded:
            await callback.log_success_fallback_event(
                original_model_group=original_model_group, kwargs=kwargs, original_exception=original_exception
            )
        else:
            await callback.log_failure_fallback_event(
                original_model_group=original_model_group, kwargs=kwargs, original_exception=original_exception
            )
    except Exception as e:
        verbose_router_logger.error(
            "Error in log_success_fallback_event: %s" if succeeded else "Error in log_failure_fallback_event: %s", e
        )


def check_non_standard_fallback_format(fallbacks: Sequence[object] | None) -> bool:
    """
    Checks if the fallbacks list is a list of strings or a list of dictionaries.

    If
    - List[str]: e.g. ["claude-3-haiku", "openai/o-1"]
    - List[Dict[<LiteLLMParamsTypedDict>, Any]]: e.g. [{"model": "claude-3-haiku", "messages": [{"role": "user", "content": "Hey, how's it going?"}]}]

    If [{"gpt-3.5-turbo": ["claude-3-haiku"]}] then standard format.
    """
    if fallbacks is None or not isinstance(fallbacks, list) or len(fallbacks) == 0:
        return False
    if all(isinstance(item, str) for item in fallbacks):
        return True
    dict_entries: Final = tuple(item for item in fallbacks if isinstance(item, dict))
    if len(dict_entries) == len(fallbacks):
        for item in dict_entries:
            for key in LiteLLMParamsTypedDict.__annotations__:
                if key in item:
                    # If the value is a list, it's likely a standard fallback model group mapping
                    # (e.g. {"model": ["backup"]}) rather than a parameter override.
                    if not isinstance(item[key], list):
                        return True

    return False


_check_non_standard_fallback_format = check_non_standard_fallback_format
