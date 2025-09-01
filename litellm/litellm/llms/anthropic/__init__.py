from typing import Type, Union

from .batches.transformation import AnthropicBatchesConfig
from .chat.transformation import AnthropicConfig

__all__ = ["AnthropicBatchesConfig", "AnthropicConfig"]


def get_anthropic_config(
    url_route: str,
) -> Union[Type[AnthropicBatchesConfig], Type[AnthropicConfig]]:
    if "messages/batches" in url_route and "results" in url_route:
        return AnthropicBatchesConfig
    else:
        return AnthropicConfig
