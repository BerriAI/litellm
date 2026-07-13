"""
Claude Code Endpoints

Provides endpoints for Claude Code plugin marketplace integration.
"""

from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_marketplace import (
    router as claude_code_marketplace_router,
)
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_marketplace_sources import (
    router as claude_code_marketplace_sources_router,
)

claude_code_marketplace_router.include_router(claude_code_marketplace_sources_router)

__all__ = ["claude_code_marketplace_router"]
