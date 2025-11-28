"""
Azure Anthropic provider - supports Claude models via Azure Foundry
"""
from .handler import AzureAnthropicChatCompletion
from .transformation import AzureAnthropicConfig

try:
    from .messages_transformation import AzureAnthropicMessagesConfig
    __all__ = ["AzureAnthropicChatCompletion", "AzureAnthropicConfig", "AzureAnthropicMessagesConfig"]
except ImportError:
    __all__ = ["AzureAnthropicChatCompletion", "AzureAnthropicConfig"]

