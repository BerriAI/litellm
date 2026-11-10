import litellm.mcp_client as _mcp_client
from litellm.litellm_core_utils.deprecated_package import alias_deprecated_package
from litellm.mcp_client import call_openai_tool, load_mcp_tools

alias_deprecated_package(__name__, _mcp_client)

__all__ = ["call_openai_tool", "load_mcp_tools"]
