"""Validate the number of inputs in a single embeddings request."""

from collections.abc import Mapping, Sequence
from typing import Final

from typing_extensions import TypeIs  # noqa: TID251  # TypeIs reaches typing only on 3.13

import litellm


def _is_str_keyed_mapping(
    value: object,
) -> TypeIs[Mapping[str, object]]:  # guard-ok: model_info, litellm_params and extra_body are str-keyed config objects
    return isinstance(value, Mapping)


def _is_embedding_input(
    value: object,
) -> TypeIs[str | Sequence[object]]:  # guard-ok: only the length and first item are read
    return isinstance(value, Sequence)


def count_embedding_inputs(input: str | Sequence[object]) -> int:
    """Return the number of embeddings requested without tokenizing the input."""
    if isinstance(input, str):
        return 1
    if input and isinstance(input[0], int):
        return 1
    return len(input)


def effective_embedding_input(
    input: str | Sequence[object],
    extra_body: object,
) -> str | Sequence[object]:
    """Return the input shape used for batch-size checks.

    Some providers merge ``extra_body`` over the request body (OpenAI-compatible paths);
    others ignore ``extra_body.input`` and use the top-level input (e.g. Bedrock Titan).
    Use whichever of the two yields the larger input count so limits cannot be bypassed.
    """
    if _is_str_keyed_mapping(extra_body):
        override: Final = extra_body.get("input")
        if _is_embedding_input(override) and count_embedding_inputs(override) > count_embedding_inputs(input):
            return override
    return input


def _litellm_param(deployment: Mapping[str, object], key: str) -> object:
    litellm_params: Final = deployment.get("litellm_params")
    return litellm_params.get(key) if _is_str_keyed_mapping(litellm_params) else None


def _embedding_batch_limit(deployment: Mapping[str, object]) -> int | None:
    model_info: Final = deployment.get("model_info")
    limit: Final = model_info.get("max_embedding_batch_size") if _is_str_keyed_mapping(model_info) else None
    return limit if isinstance(limit, int) else None


def _requested_embedding_count(
    deployment: Mapping[str, object],
    input: str | Sequence[object],
    request_kwargs: Mapping[str, object],
) -> int:
    extra_body: Final = request_kwargs.get("extra_body", _litellm_param(deployment, "extra_body"))
    return count_embedding_inputs(effective_embedding_input(input, extra_body))


def _deployment_admits_embedding_batch(
    deployment: Mapping[str, object],
    input: str | Sequence[object],
    request_kwargs: Mapping[str, object],
) -> bool:
    limit: Final = _embedding_batch_limit(deployment)
    return limit is None or _requested_embedding_count(deployment, input, request_kwargs) <= limit


def _str_keyed_or_empty(value: object) -> Mapping[str, object]:
    return value if _is_str_keyed_mapping(value) else {}


def deployments_admitting_embedding_batch(
    deployments: list[dict[str, object]],  # mutable-ok: router deployment-list contract
    input: object,
    request_kwargs: object,
) -> list[dict[str, object]]:  # mutable-ok: returns a mutable deployment list
    """Keep the deployments whose input limit admits the request, or all of them when none does."""
    if not _is_embedding_input(input):
        return deployments
    kwargs: Final = _str_keyed_or_empty(request_kwargs)
    admitting: Final = [
        deployment for deployment in deployments if _deployment_admits_embedding_batch(deployment, input, kwargs)
    ]
    return admitting or deployments


def raise_if_embedding_batch_too_large(
    deployment: object,
    input: object,
    request_kwargs: object,
    model: str,
) -> None:
    """Reject an embeddings request that exceeds its deployment's input limit."""
    if not _is_str_keyed_mapping(deployment) or not _is_embedding_input(input):
        return
    limit: Final = _embedding_batch_limit(deployment)
    if limit is None:
        return

    input_count: Final = _requested_embedding_count(deployment, input, _str_keyed_or_empty(request_kwargs))
    if input_count <= limit:
        return

    provider: Final = _litellm_param(deployment, "custom_llm_provider")
    raise litellm.BadRequestError(
        message=(
            f"Embedding request has {input_count} inputs, which exceeds "
            f"max_embedding_batch_size={limit} for model={model}. "
            "Split the input into smaller batches."
        ),
        model=model,
        llm_provider=provider if isinstance(provider, str) else "",
    )
