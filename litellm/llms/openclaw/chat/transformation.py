"""
Translate from OpenAI's `/v1/chat/completions` to OpenClaw's `/v1/chat/completions`

OpenClaw is an AI agent framework that exposes an OpenAI-compatible HTTP endpoint.
https://docs.openclaw.ai

Key features:
- Target specific agents via model field: `openclaw/main`, `openclaw/research`
- Session persistence via `user` field
- Streaming support (SSE)
"""

from typing import List, Optional, Tuple

from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class OpenClawChatConfig(OpenAIGPTConfig):
    """
    OpenClaw configuration for chat completions.
    
    OpenClaw agents are targeted via the model field:
    - `openclaw/main` -> main agent
    - `openclaw/research` -> research agent
    - `openclaw/<agent-id>` -> any configured agent
    
    Environment variables:
    - OPENCLAW_API_BASE: Gateway URL (e.g., http://localhost:18789)
    - OPENCLAW_API_KEY: Gateway auth token
    """

    def get_supported_openai_params(self, model: str) -> List[str]:
        """OpenClaw supports standard OpenAI params plus user for session persistence."""
        return [
            "stream",
            "stop",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",  # Used for session key derivation
            "n",
            "tools",
            "tool_choice",
            "response_format",
        ]

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get OpenClaw API base and key from environment or parameters.
        
        OpenClaw requires an auth token when gateway.auth.mode is set.
        """
        api_base = api_base or get_secret_str("OPENCLAW_API_BASE")
        dynamic_api_key = api_key or get_secret_str("OPENCLAW_API_KEY") or ""
        return api_base, dynamic_api_key

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to OpenClaw format.
        
        OpenClaw is fully OpenAI-compatible, so minimal transformation needed.
        The model field can include agent targeting: openclaw/main -> agent:main
        """
        return super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )
