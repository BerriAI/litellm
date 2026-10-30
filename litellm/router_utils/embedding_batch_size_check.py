import traceback
from collections.abc import Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Final

from typing_extensions import TypeIs  # noqa: TID251  # TypeIs reaches typing only on 3.13

import litellm
from litellm.constants import PROXY_REJECTED_BEFORE_ROUTING_KEY
from litellm.types.utils import CallTypes

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_SELECTING_FOR_EMBEDDING: Final[ContextVar[bool]] = ContextVar("selecting_for_embedding", default=False)


def _is_str_keyed_mapping(
    value: object,
) -> TypeIs[Mapping[str, object]]:  # guard-ok: model_info, litellm_params and extra_body are str-keyed config objects
    return isinstance(value, Mapping)


def _is_embedding_input(
    value: object,
) -> TypeIs[str | Sequence[object]]:  # guard-ok: only the length and first item are read
    return isinstance(value, Sequence)


def count_embedding_inputs(input: str | Sequence[object]) -> int:
    if isinstance(input, str):
        return 1
    if input and isinstance(input[0], int):
        return 1
    return len(input)


def effective_embedding_input(
    input: object,
    extra_body: object,
) -> str | Sequence[object] | None:
    """Return the input shape used for batch-size checks, or None when neither is an embedding input.

    Some providers merge ``extra_body`` over the request body (OpenAI-compatible paths);
    others ignore ``extra_body.input`` and use the top-level input (e.g. Bedrock Titan).
    Use whichever of the two yields the larger input count so limits cannot be bypassed.
    """
    override: Final = extra_body.get("input") if _is_str_keyed_mapping(extra_body) else None
    candidates: Final = tuple(value for value in (input, override) if _is_embedding_input(value))
    return max(candidates, key=count_embedding_inputs, default=None)


def _litellm_param(deployment: Mapping[str, object], key: str) -> object:
    litellm_params: Final = deployment.get("litellm_params")
    return litellm_params.get(key) if _is_str_keyed_mapping(litellm_params) else None


def _embedding_batch_limit(deployment: Mapping[str, object]) -> int | None:
    model_info: Final = deployment.get("model_info")
    limit: Final = model_info.get("max_embedding_batch_size") if _is_str_keyed_mapping(model_info) else None
    return limit if isinstance(limit, int) else None


def _requested_embedding_count(
    deployment: Mapping[str, object],
    input: object,
    request_kwargs: Mapping[str, object],
) -> int:
    extra_body: Final = request_kwargs.get("extra_body", _litellm_param(deployment, "extra_body"))
    effective_input: Final = effective_embedding_input(input, extra_body)
    return 0 if effective_input is None else count_embedding_inputs(effective_input)


def _deployment_admits_embedding_batch(
    deployment: Mapping[str, object],
    input: object,
    request_kwargs: Mapping[str, object],
) -> bool:
    limit: Final = _embedding_batch_limit(deployment)
    return limit is None or _requested_embedding_count(deployment, input, request_kwargs) <= limit


def _str_keyed_or_empty(value: object) -> Mapping[str, object]:
    return value if _is_str_keyed_mapping(value) else {}


def _deployment_id(deployment: Mapping[str, object]) -> str | None:
    model_info: Final = deployment.get("model_info")
    deployment_id: Final = model_info.get("id") if _is_str_keyed_mapping(model_info) else None
    return None if deployment_id is None else str(deployment_id)


def _is_paused(deployment: Mapping[str, object]) -> bool:
    model_info: Final = deployment.get("model_info")
    return _is_str_keyed_mapping(model_info) and model_info.get("blocked") is True


@contextmanager
def embedding_deployment_selection() -> Generator[None]:
    """Apply the input limit to deployment selection inside this block, which only embedding calls enter."""
    token: Final = _SELECTING_FOR_EMBEDDING.set(True)
    try:
        yield
    finally:
        _SELECTING_FOR_EMBEDDING.reset(token)


def deployments_admitting_embedding_batch(
    deployments: list[dict[str, object]],  # mutable-ok: router deployment-list contract
    input: object,
    request_kwargs: object,
) -> list[dict[str, object]] | None:  # mutable-ok: returns a mutable deployment list
    """Keep the deployments whose input limit admits the request, all of them outside embedding selection, else None"""
    if not _SELECTING_FOR_EMBEDDING.get():
        return deployments
    kwargs: Final = _str_keyed_or_empty(request_kwargs)
    admitting: Final = [
        deployment for deployment in deployments if _deployment_admits_embedding_batch(deployment, input, kwargs)
    ]
    return admitting or None


def deployments_when_none_admits_embedding_batch(
    deployments: list[dict[str, object]],  # mutable-ok: router deployment-list contract
    before_cooldown: Iterable[Mapping[str, object]],
    routable_cooling_ids: frozenset[str],
    input: object,
    request_kwargs: object,
) -> list[dict[str, object]]:  # mutable-ok: returns a mutable deployment list
    """None while an unpaused cooling deployment the request routes to admits it, so it waits out the cooldown,
    else the deployments with the largest limit, so the 400 names the most the group accepts"""
    kwargs: Final = _str_keyed_or_empty(request_kwargs)
    routable_cooling: Final = (
        deployment
        for deployment in before_cooldown
        if _deployment_id(deployment) in routable_cooling_ids and not _is_paused(deployment)
    )
    if any(_deployment_admits_embedding_batch(deployment, input, kwargs) for deployment in routable_cooling):
        return []
    largest_limit: Final = max((_embedding_batch_limit(deployment) or 0 for deployment in deployments), default=0)
    return [deployment for deployment in deployments if _embedding_batch_limit(deployment) == largest_limit]


def _mark_litellm_params(logging_obj: "LiteLLMLoggingObj", *keys: str) -> None:
    litellm_params: Final = logging_obj.model_call_details.get("litellm_params")
    if not isinstance(litellm_params, dict):
        return
    for key in keys:
        litellm_params[key] = True


def log_rejection_before_provider_call(logging_obj: "LiteLLMLoggingObj", error: Exception) -> None:
    """Mark the request as never reaching a provider, so deployment health metrics skip its failure, then log it"""
    _mark_litellm_params(logging_obj, PROXY_REJECTED_BEFORE_ROUTING_KEY)
    logging_obj.failure_handler(error, "".join(traceback.format_exception(error)))


async def async_log_rejection_before_provider_call(logging_obj: "LiteLLMLoggingObj", error: Exception) -> None:
    """Log it like an async provider failure: sync handlers inline, before a fallback swaps the deployment they read,
    and the call marked async, so a logger with both hooks records it once"""
    _mark_litellm_params(logging_obj, PROXY_REJECTED_BEFORE_ROUTING_KEY, CallTypes.aembedding.value)
    traceback_text: Final = "".join(traceback.format_exception(error))
    logging_obj.failure_handler(error, traceback_text)
    await logging_obj.async_failure_handler(error, traceback_text)


def clear_rejected_before_provider_call(logging_obj: "LiteLLMLoggingObj") -> None:
    """Drop the mark an earlier rejected attempt left, since a fallback or retry shares the logging object"""
    litellm_params: Final = logging_obj.model_call_details.get("litellm_params")
    if isinstance(litellm_params, dict) and PROXY_REJECTED_BEFORE_ROUTING_KEY in litellm_params:
        del litellm_params[PROXY_REJECTED_BEFORE_ROUTING_KEY]


def embedding_batch_too_large_error(
    deployment: object,
    input: object,
    request_kwargs: object,
    model: str,
) -> litellm.BadRequestError | None:
    if not _is_str_keyed_mapping(deployment):
        return None
    limit: Final = _embedding_batch_limit(deployment)
    if limit is None:
        return None

    input_count: Final = _requested_embedding_count(deployment, input, _str_keyed_or_empty(request_kwargs))
    if input_count <= limit:
        return None

    return litellm.BadRequestError(
        message=(
            f"Embedding request has {input_count} inputs, which exceeds "
            f"max_embedding_batch_size={limit} for model={model}. "
            "Split the input into smaller batches."
        ),
        model=model,
        llm_provider="",
    )
