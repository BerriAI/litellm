"""
This file contains the handler for xAI's Grok Voice Agent API `/v1/realtime` endpoint.

xAI's Realtime API is fully OpenAI-compatible, so we inherit from OpenAIRealtime
and only override the configuration differences.

This requires websockets, and is currently only supported on LiteLLM Proxy.
"""

from litellm.constants import XAI_API_BASE

from ...openai.realtime.handler import OpenAIRealtime


class XAIRealtime(OpenAIRealtime):
    """
    Handler for xAI Grok Voice Agent API.
    
    xAI's Realtime API uses the same WebSocket protocol as OpenAI but with:
    - Different endpoint: wss://api.x.ai/v1/realtime (via _get_default_api_base)
    - No OpenAI-Beta header required (via _get_additional_headers)
    - Model: grok-4-1-fast-non-reasoning
    
    All WebSocket logic is inherited from OpenAIRealtime.
    """
    
    def _get_default_api_base(self) -> str:
        """xAI uses a different API base URL."""
        return XAI_API_BASE
    
    def _get_additional_headers(self, api_key: str) -> dict:
        """
        xAI does NOT require the OpenAI-Beta header.
        Only send Authorization header.
        """
        return {
            "Authorization": f"Bearer {api_key}",
        }
