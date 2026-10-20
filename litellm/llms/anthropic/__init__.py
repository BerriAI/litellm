from .batches.transformation import AnthropicBatchesConfig
from .chat.transformation import AnthropicConfig

__all__ = ["AnthropicBatchesConfig", "AnthropicConfig"]


def get_anthropic_config(
    url_route: str,
) -> type[AnthropicBatchesConfig] | type[AnthropicConfig]:
    if "messages/batches" in url_route and "results" in url_route:
        return AnthropicBatchesConfig
    else:
        return AnthropicConfig
