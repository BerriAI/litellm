from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel

from litellm._logging import verbose_logger
from litellm.types.mcp_server.tool_registry import MCPTool


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

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """
        Get a tool from the registry by name
        """
        return self.tools.get(name)

    def list_tools(self) -> List[MCPTool]:
        """
        List all registered tools
        """
        return list(self.tools.values())

    def load_tools_from_config(self, config: Dict[str, Any]) -> None:
        """
        Load and register tools from the proxy config

        Args:
            config: The loaded proxy configuration
        """
        if "mcp_tools" not in config:
            return

        # Import handlers dynamically
        import importlib
        import sys

        for tool_config in config["mcp_tools"]:
            name = tool_config.get("name")
            description = tool_config.get("description")
            input_schema = tool_config.get("input_schema", {})
            handler_name = tool_config.get("handler")

            if not all([name, description, handler_name]):
                continue

            # Try to resolve the handler
            # First check if it's a module path (e.g., "module.submodule.function")
            if "." in handler_name:
                module_path, func_name = handler_name.rsplit(".", 1)
                try:
                    module = importlib.import_module(module_path)
                    handler = getattr(module, func_name)
                except (ImportError, AttributeError):
                    verbose_logger.warning(
                        f"Warning: Could not load handler {handler_name} for tool {name}"
                    )
                    continue
            else:
                # Check if it's in the global namespace
                handler = globals().get(handler_name)
                if handler is None:
                    # Check if it's in sys.modules
                    for module_name, module in sys.modules.items():
                        if hasattr(module, handler_name):
                            handler = getattr(module, handler_name)
                            break

                if handler is None:
                    verbose_logger.warning(
                        f"Warning: Could not find handler {handler_name} for tool {name}"
                    )
                    continue

            # Register the tool
            self.register_tool(
                name=name,
                description=description,
                input_schema=input_schema,
                handler=handler,
            )
