"""
Claude Code Endpoints

Provides endpoints for Claude Code plugin marketplace integration.
"""

from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_marketplace import (
    router as claude_code_marketplace_router,
)

__all__ = ["claude_code_marketplace_router"]
