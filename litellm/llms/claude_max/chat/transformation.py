import os
from typing import Optional

from litellm.llms.anthropic.chat.transformation import AnthropicConfig

from ..authenticator import ClaudeMaxAuthenticator


class ClaudeMaxConfig(AnthropicConfig):
    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "claude_max"

    def get_access_token(self) -> str:
        return ClaudeMaxAuthenticator().get_access_token()

    def get_api_base(self, api_base: Optional[str] = None) -> str:
        return api_base or os.getenv("CLAUDE_MAX_API_BASE") or "https://api.anthropic.com/v1/messages"

    def get_default_headers(self) -> dict:
        return {
            "User-Agent": os.getenv("CLAUDE_MAX_USER_AGENT", "claude-cli/2.1.123 (external, sdk-cli)"),
            "x-app": "cli",
            "X-Stainless-Lang": "js",
            "X-Stainless-Runtime": "node",
        }
