"""Initialize semantic MCP tool filtering from proxy config."""

from __future__ import annotations

from typing import Any, Optional

from litellm._logging import verbose_proxy_logger

from .registry import semantic_mcp_filter_registry
from .settings import SemanticFilterConfig


async def configure_semantic_filter(
    config_section: Optional[dict],
    *,
    mcp_server_manager: Any,
) -> None:
    """Parse config and bootstrap the registry.

    Args:
        config_section: The `litellm_settings.semantic_mcp_filter` payload.
        mcp_server_manager: Global MCP server manager for listing tools.
    """
    if not config_section:
        semantic_mcp_filter_registry.reset()
        return

    try:
        parsed_config = SemanticFilterConfig.model_validate(config_section)
    except Exception:
        verbose_proxy_logger.exception(
            "semantic_mcp_filter config is invalid; feature disabled"
        )
        semantic_mcp_filter_registry.reset()
        return

    semantic_mcp_filter_registry.configure(parsed_config)

    if not parsed_config.enabled:
        return

    try:
        await semantic_mcp_filter_registry.rebuild_index(mcp_server_manager)
    except Exception:
        verbose_proxy_logger.exception("Failed to bootstrap semantic MCP filter index")
