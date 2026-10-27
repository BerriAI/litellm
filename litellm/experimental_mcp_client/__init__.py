from litellm.litellm_core_utils.deprecated_package import alias_deprecated_package
from litellm.mcp_client import call_openai_tool, load_mcp_tools

__getattr__ = alias_deprecated_package(__name__, "litellm.mcp_client")

__all__ = ["call_openai_tool", "load_mcp_tools"]
