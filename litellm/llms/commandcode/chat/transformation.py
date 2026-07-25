"""
CommandCode Provider API

Supports both OpenAI and Anthropic compatible endpoints:
- OpenAI:    https://api.commandcode.ai/provider/v1/chat/completions
- Anthropic: https://api.commandcode.ai/provider/v1/messages

Docs: https://commandcode.ai/docs/provider
Auth: Authorization: Bearer <CMD_API_KEY>
"""

from typing import List, Optional, Tuple

from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues


COMMANDCODE_API_BASE = "https://api.commandcode.ai"



class CommandCodeError(Exception):
    """Exception class for CommandCode API errors."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def _get_commandcode_credentials(
    api_base: Optional[str],
    api_key: Optional[str],
) -> Tuple[str, str]:
    """
    Resolve API base and key from params or environment.
    Shared by both OpenAI and Anthropic configs.
    """
    resolved_base = (
        api_base
        or get_secret_str("COMMANDCODE_API_BASE")
        or get_secret_str("CMD_API_BASE")
        or COMMANDCODE_API_BASE
    ).rstrip("/")

    resolved_key = (
        api_key
        or get_secret_str("COMMANDCODE_API_KEY")
        or get_secret_str("CMD_API_KEY")
    )

    if resolved_key is None:
        raise CommandCodeError(
            "No CommandCode API key found. "
            "Set COMMANDCODE_API_KEY environment variable "
            "or pass api_key parameter."
        )

    return resolved_base, resolved_key

# ─────────────────────────────────────────────
# OpenAI-compatible config (non-Claude models)
# ─────────────────────────────────────────────
class CommandCodeOpenAIConfig(OpenAIGPTConfig):
    """
    Handles: GPT, Gemini, DeepSeek, and all other non-Claude models
    Endpoint: /provider/v1/chat/completions
    Format:   OpenAI Chat Completions
    """

    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = (
            api_base
            or get_secret_str("COMMANDCODE_API_BASE")
            or get_secret_str("CMD_API_BASE")
            or COMMANDCODE_API_BASE
        )
        dynamic_api_key = (
            api_key
            or get_secret_str("COMMANDCODE_API_KEY")
            or get_secret_str("CMD_API_KEY")
        )
        return api_base, dynamic_api_key

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base, _ = _get_commandcode_credentials(api_base, api_key)
        return f"{base}/provider/v1/chat/completions"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        _, key = _get_commandcode_credentials(api_base, api_key)
        headers["Authorization"] = f"Bearer {key}"
        headers["Content-Type"] = "application/json"
        return headers
    
# ─────────────────────────────────────────────
# Anthropic-compatible config (Claude models)
# ─────────────────────────────────────────────
class CommandCodeAnthropicConfig(AnthropicConfig):
    """
    Handles: All claude-* models
    Endpoint: /provider/v1/messages
    Format:   Anthropic Messages API
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "commandcode"

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base, _ = _get_commandcode_credentials(api_base, api_key)
        return f"{base}/provider/v1/messages"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        _, key = _get_commandcode_credentials(api_base, api_key)
        headers["Authorization"] = f"Bearer {key}"
        headers["Content-Type"] = "application/json"
        headers["anthropic-version"] = "2023-06-01"
        return headers
    
    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = (
            api_base
            or get_secret_str("COMMANDCODE_API_BASE")
            or get_secret_str("CMD_API_BASE")
            or COMMANDCODE_API_BASE
        )
        dynamic_api_key = (
            api_key
            or get_secret_str("COMMANDCODE_API_KEY")
            or get_secret_str("CMD_API_KEY")
        )
        return api_base, dynamic_api_key