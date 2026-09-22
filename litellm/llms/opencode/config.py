"""
Shared provider config for opencode surfaces.

Decides which wire format a model speaks and returns the matching config.
Both arms return a ``BaseConfig``, so callers that only know the provider (the
generic ``completion()`` preprocessing, ``get_supported_openai_params``, the
streaming wrappers) get a usable config either way.
"""

from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.opencode.chat.messages_transformation import is_messages_model

from .chat.anthropic_transformation import OpenCodeAnthropicConfig
from .chat.transformation import OpenCodeConfig


def get_opencode_config(surface: str, model: str) -> BaseConfig:
    """Return the right config for *surface* / *model*.

    Models the gateway serves over the Anthropic Messages wire format get the
    Anthropic-wire chat config; everything else falls through to
    chat-completions.
    """
    if is_messages_model(surface, model):
        return OpenCodeAnthropicConfig(surface=surface)
    return OpenCodeConfig(surface=surface)
