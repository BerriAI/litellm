"""
Built-in MCP Server Registry

This module provides a registry for pre-configured MCP servers that can be referenced
by simple names (e.g., "zapier", "jira") instead of requiring full server configuration.

The built-in servers are managed centrally at the proxy level, providing:
- Enhanced security (no credential exposure to clients)  
- Simplified client usage (just specify builtin name)
- Centralized authentication management
- Environment-based configuration
"""

import os
from typing import Dict, List, Optional, Union

from litellm._logging import verbose_logger
from litellm.types.mcp_server.mcp_server_manager import MCPServer


class BuiltinStdioMCPServerConfig:
    """Configuration for a built-in stdio MCP server"""
    
    def __init__(
        self,
        name: str,
        command: str,
        args: List[str] = None,
        env: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        spec_version: str = "2025-03-26"
    ):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.description = description
        self.spec_version = spec_version
        self.transport = "stdio"
        self.auth_type = None
        self.env_key = None
    
    def to_mcp_server(self, server_id: str) -> MCPServer:
        """Convert to MCPServer instance for stdio transport"""
        return MCPServer(
            server_id=server_id,
            name=self.name,
            url=None,  # No URL for stdio transport
            transport=self.transport,
            auth_type=self.auth_type,
            spec_version=self.spec_version,
            command=self.command,
            args=self.args,
            env=self.env
        )


class BuiltinMCPServerConfig:
    """Configuration for a built-in MCP server"""
    
    def __init__(
        self,
        name: str,
        url: str,
        transport: str = "sse",
        auth_type: Optional[str] = None,
        env_key: Optional[str] = None,
        description: Optional[str] = None,
        spec_version: str = "2025-03-26"
    ):
        self.name = name
        self.url = url
        self.transport = transport
        self.auth_type = auth_type
        self.env_key = env_key
        self.description = description
        self.spec_version = spec_version
    
    def to_mcp_server(self, server_id: str) -> MCPServer:
        """Convert to MCPServer instance with environment-based authentication"""
        auth_headers = {}
        
        # Set up authentication if configured
        if self.auth_type and self.env_key:
            token = os.getenv(self.env_key)
            if not token:
                verbose_logger.warning(
                    f"Environment variable {self.env_key} not set for builtin MCP server '{self.name}'"
                )
            else:
                if self.auth_type == "bearer_token":
                    auth_headers["Authorization"] = f"Bearer {token}"
                elif self.auth_type == "api_key":
                    auth_headers["Authorization"] = f"Bearer {token}"
                elif self.auth_type == "basic":
                    import base64
                    auth_headers["Authorization"] = f"Basic {base64.b64encode(token.encode()).decode()}"
        
        return MCPServer(
            server_id=server_id,
            name=self.name,
            url=self.url,
            transport=self.transport,  # Already string, let pydantic handle validation
            auth_type=self.auth_type,  # Already string, let pydantic handle validation  
            spec_version=self.spec_version,
            authentication_token=os.getenv(self.env_key) if self.env_key else None
        )


class BuiltinMCPRegistry:
    """Registry for built-in MCP server configurations"""
    
    def __init__(self):
        self._builtin_configs: Dict[str, Union[BuiltinMCPServerConfig, BuiltinStdioMCPServerConfig]] = {}
        self._initialize_default_servers()
    
    def _initialize_default_servers(self):
        """Initialize the default set of built-in MCP servers"""
        
        # Zapier MCP Server
        self.register_builtin(
            BuiltinMCPServerConfig(
                name="zapier",
                url="https://mcp.zapier.com/api/mcp",
                transport="sse",
                auth_type="bearer_token",
                env_key="ZAPIER_TOKEN",
                description="Zapier automation platform integration"
            )
        )
        
        # Jira MCP Server  
        self.register_builtin(
            BuiltinMCPServerConfig(
                name="jira",
                url="https://mcp.atlassian.com/v1/sse", 
                transport="sse",
                auth_type="bearer_token",
                env_key="JIRA_TOKEN",
                description="Atlassian Jira project management integration"
            )
        )
        
        # GitHub MCP Server
        self.register_builtin(
            BuiltinMCPServerConfig(
                name="github",
                url="https://mcp.github.com/mcp",
                transport="http", 
                auth_type="bearer_token",
                env_key="GITHUB_TOKEN",
                description="GitHub repository and issue management"
            )
        )
        
        # Slack MCP Server
        self.register_builtin(
            BuiltinMCPServerConfig(
                name="slack",
                url="https://mcp.slack.com/api/mcp",
                transport="http",
                auth_type="bearer_token", 
                env_key="SLACK_BOT_TOKEN",
                description="Slack team communication platform"
            )
        )
        
        # Sample Calculator MCP Server (Example)
        # This demonstrates how to add a custom stdio-based MCP server as a builtin
        import os
        calc_server_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "sample_calculator_mcp_server.py")
        self.register_builtin(
            BuiltinStdioMCPServerConfig(
                name="calculator",
                command="python",
                args=[calc_server_path],
                description="Sample calculator MCP server for basic math operations"
            )
        )
        
        verbose_logger.debug(f"Initialized {len(self._builtin_configs)} built-in MCP servers")
    
    def register_builtin(self, config: Union[BuiltinMCPServerConfig, BuiltinStdioMCPServerConfig]):
        """Register a new built-in MCP server configuration"""
        self._builtin_configs[config.name] = config
        verbose_logger.debug(f"Registered built-in MCP server: {config.name}")
    
    def get_builtin_config(self, name: str) -> Optional[Union[BuiltinMCPServerConfig, BuiltinStdioMCPServerConfig]]:
        """Get built-in server configuration by name"""
        return self._builtin_configs.get(name)
    
    def list_builtin_names(self) -> List[str]:
        """List all available built-in server names"""
        return list(self._builtin_configs.keys())
    
    def is_builtin_available(self, name: str) -> bool:
        """Check if a built-in server is available and properly configured"""
        config = self.get_builtin_config(name)
        if not config:
            return False
        
        # For stdio servers, check if the command exists
        if isinstance(config, BuiltinStdioMCPServerConfig):
            # For stdio servers, we assume they're available if the config exists
            # In a production system, you might want to check if the command/script exists
            return True
        
        # For HTTP/SSE servers, check if required environment variable is set
        if hasattr(config, 'env_key') and config.env_key:
            return bool(os.getenv(config.env_key))
        
        return True
    
    def get_builtin_server(self, name: str, server_id: Optional[str] = None) -> Optional[MCPServer]:
        """Get a built-in MCP server instance by name"""
        config = self.get_builtin_config(name)
        if not config:
            return None
        
        # Use name as server_id if not provided
        if not server_id:
            server_id = f"builtin_{name}"
        
        return config.to_mcp_server(server_id)
    
    def get_available_builtins(self) -> Dict[str, MCPServer]:
        """Get all available built-in servers as MCPServer instances"""
        available = {}
        
        for name in self.list_builtin_names():
            if self.is_builtin_available(name):
                server = self.get_builtin_server(name)
                if server:
                    available[f"builtin_{name}"] = server
            else:
                verbose_logger.debug(f"Built-in MCP server '{name}' not available (missing environment variable)")
        
        return available


# Global registry instance
global_builtin_registry = BuiltinMCPRegistry()