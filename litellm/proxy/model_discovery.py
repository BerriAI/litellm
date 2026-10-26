from typing import Final

from pydantic import TypeAdapter

from litellm.types.proxy.model_listing import ModelDiscoveryInfo
from litellm.types.router import ModelMetadata, common_model_metadata

_ADAPTER: Final = TypeAdapter(ModelDiscoveryInfo)


def discover_model_metadata(sources: tuple[ModelMetadata, ...]) -> ModelDiscoveryInfo:
    return _ADAPTER.validate_python(
        common_model_metadata(sources).model_dump(
            mode="json", exclude_none=True, exclude={"max_input_tokens", "max_output_tokens"}
        )
    )
