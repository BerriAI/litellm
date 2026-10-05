from collections.abc import Sequence
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm.types.proxy.model_listing import ModelDiscoveryInfo
from litellm.types.proxy.model_metadata import GatewayModelMetadata

_ADAPTER: Final = TypeAdapter(ModelDiscoveryInfo)


def guaranteed_token_limit(
    sources: tuple[GatewayModelMetadata, ...], field: Literal["context_window", "max_input_tokens", "max_output_tokens"]
) -> int | None:
    values: Final = tuple(
        source.context_window
        if field == "context_window"
        else source.max_input_tokens
        if field == "max_input_tokens"
        else source.max_output_tokens
        for source in sources
    )
    if not values or any(value is None for value in values):
        return None
    return min(value for value in values if value is not None)


def _common_boolean(values: tuple[bool | None, ...]) -> bool | None:
    return all(values) if values and all(value is not None for value in values) else None


def _common_list(values: tuple[Sequence[str] | None, ...]) -> Sequence[str] | None:
    if not values or any(value is None for value in values):
        return None
    first: Final = values[0]
    if first is None:
        return None
    return tuple(item for item in first if all(value is not None and item in value for value in values))


def discover_model_metadata(sources: tuple[GatewayModelMetadata, ...]) -> ModelDiscoveryInfo:
    efforts: Final = _common_list(tuple(source.reasoning_effort_levels for source in sources))
    defaults: Final = tuple(source.default_reasoning_effort for source in sources)
    default: Final = defaults[0] if defaults and all(value == defaults[0] for value in defaults) else None
    valid_default: Final = default if efforts is None or default in efforts else None
    metadata: Final = GatewayModelMetadata(
        context_window=guaranteed_token_limit(sources, "context_window"),
        supports_function_calling=_common_boolean(tuple(source.supports_function_calling for source in sources)),
        supports_parallel_function_calling=_common_boolean(
            tuple(source.supports_parallel_function_calling for source in sources)
        ),
        supports_reasoning=_common_boolean(tuple(source.supports_reasoning for source in sources)),
        reasoning_effort_levels=efforts,
        default_reasoning_effort=valid_default,
        supported_endpoints=_common_list(tuple(source.supported_endpoints for source in sources)),
        supported_modalities=_common_list(tuple(source.supported_modalities for source in sources)),
        supported_output_modalities=_common_list(tuple(source.supported_output_modalities for source in sources)),
    )
    return _ADAPTER.validate_python(metadata.model_dump(mode="json", exclude_none=True))
