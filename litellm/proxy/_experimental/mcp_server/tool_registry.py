import json
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from litellm._logging import verbose_logger
from litellm.proxy.types_utils.utils import get_instance_fn
from litellm.types.mcp_server.tool_registry import MCPTool

if TYPE_CHECKING:
    from mcp.types import Tool as MCPToolSDKTool
else:
    try:
        from mcp.types import Tool as MCPToolSDKTool
    except ImportError:
        MCPToolSDKTool = None  # type: ignore


class MCPToolRegistry:
    """
    A registry for managing MCP tools
    """

    def __init__(self):
        # Registry to store all registered tools
        self.tools: Dict[str, MCPTool] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable,
    ) -> None:
        """
        Register a new tool in the registry
        """
        self.tools[name] = MCPTool(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
        )
        verbose_logger.debug(f"Registered tool: {name}")

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """
        Get a tool from the registry by name
        """
        return self.tools.get(name)

    def list_tools(self, tool_prefix: Optional[str] = None) -> List[MCPTool]:
        """
        List all registered tools
        """
        if tool_prefix:
            return [
                tool
                for tool in self.tools.values()
                if tool.name.startswith(tool_prefix)
            ]
        return list(self.tools.values())

    def convert_tools_to_mcp_sdk_tool_type(
        self, tools: List[MCPTool]
    ) -> List["MCPToolSDKTool"]:
        if MCPToolSDKTool is None:
            raise ImportError(
                "MCP SDK is not installed. Please install it with: pip install 'litellm[proxy]'"
            )
        return [
            MCPToolSDKTool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.input_schema,
            )
            for tool in tools
        ]

    def load_tools_from_config(
        self, mcp_tools_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Load and register tools from the proxy config

        Args:
            mcp_tools_config: The mcp_tools config from the proxy config
        """
        if mcp_tools_config is None:
            raise ValueError(
                "mcp_tools_config is required, please set `mcp_tools` in your proxy config"
            )

        for tool_config in mcp_tools_config:
            if not isinstance(tool_config, dict):
                raise ValueError("mcp_tools_config must be a list of dictionaries")

            name = tool_config.get("name")
            description = tool_config.get("description")
            input_schema = tool_config.get("input_schema", {})
            handler_name = tool_config.get("handler")

            if not all([name, description, handler_name]):
                continue

            # Try to resolve the handler
            # First check if it's a module path (e.g., "module.submodule.function")
            if handler_name is None:
                raise ValueError(f"handler is required for tool {name}")
            handler = get_instance_fn(handler_name)

            if handler is None:
                verbose_logger.warning(
                    f"Warning: Could not find handler {handler_name} for tool {name}"
                )
                continue

            # Register the tool
            if name is None:
                raise ValueError(f"name is required for tool {name}")
            if description is None:
                raise ValueError(f"description is required for tool {name}")

            self.register_tool(
                name=name,
                description=description,
                input_schema=input_schema,
                handler=handler,
            )
        verbose_logger.debug(
            "all registered tools: %s", json.dumps(self.tools, indent=4, default=str)
        )


global_mcp_tool_registry = MCPToolRegistry()
