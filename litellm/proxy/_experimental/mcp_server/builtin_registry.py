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
from litellm.types.mcp import MCPTransport, MCPAuth, MCPSpecVersion
from litellm.types.mcp_server.mcp_server_manager import MCPServer


class BuiltinStdioMCPServerConfig:
    """Configuration for a built-in stdio MCP server"""
    
    def __init__(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
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
            transport=MCPTransport.stdio,
            auth_type=None,
            spec_version=MCPSpecVersion.mar_2025,
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
        auth_token = None
        
        # Set up authentication if configured
        if self.auth_type and self.env_key:
            token = os.getenv(self.env_key)
            if not token:
                verbose_logger.warning(
                    f"Environment variable {self.env_key} not set for builtin MCP server '{self.name}'"
                )
            else:
                auth_token = token
        
        # Convert string values to proper enum types
        transport_type = getattr(MCPTransport, self.transport, MCPTransport.sse)
        auth_type_enum = getattr(MCPAuth, self.auth_type, None) if self.auth_type else None
        spec_version_enum = getattr(MCPSpecVersion, self.spec_version.replace('-', '_').replace('.', '_'), MCPSpecVersion.mar_2025)
        
        return MCPServer(
            server_id=server_id,
            name=self.name,
            url=self.url,
            transport=transport_type,
            auth_type=auth_type_enum,
            spec_version=spec_version_enum,
            authentication_token=auth_token
        )


class BuiltinMCPRegistry:
    """Registry for built-in MCP server configurations"""
    
    def __init__(self, config_builtin_servers: Optional[Dict[str, Dict]] = None):
        self._builtin_configs: Dict[str, Union[BuiltinMCPServerConfig, BuiltinStdioMCPServerConfig]] = {}
        self._config_builtin_servers = config_builtin_servers or {}
        self._initialize_servers()
    
    def _initialize_servers(self):
        """Initialize built-in MCP servers from config and defaults"""
        
        # Load servers from config first
        if self._config_builtin_servers:
            for name, config in self._config_builtin_servers.items():
                # Only register if enabled (default: True)
                if config.get('enabled', True):
                    self._register_server_from_config(name, config)
        
        # If no config servers, load defaults
        if not self._builtin_configs:
            self._initialize_default_fallback_servers()
            
        verbose_logger.debug(f"Initialized {len(self._builtin_configs)} built-in MCP servers")
    
    def _register_server_from_config(self, name: str, config: Dict):
        """Register a server from config dictionary"""
        transport = config.get('transport', 'sse')
        
        if transport == 'stdio':
            # Stdio-based server
            server_config = BuiltinStdioMCPServerConfig(
                name=name,
                command=config['command'],
                args=config.get('args', []),
                env=config.get('env', {}),
                description=config.get('description', f'{name} MCP server'),
                spec_version=config.get('spec_version', '2025-03-26')
            )
        else:
            # HTTP/SSE-based server
            server_config = BuiltinMCPServerConfig(
                name=name,
                url=config['url'],
                transport=transport,
                auth_type=config.get('auth_type'),
                env_key=config.get('env_key'),
                description=config.get('description', f'{name} MCP server'),
                spec_version=config.get('spec_version', '2025-03-26')
            )
        
        self.register_builtin(server_config)
    
    def _initialize_default_fallback_servers(self):
        """Initialize the default set of built-in MCP servers (fallback only)"""
        
        # Only basic calculator as fallback
        import os
        calc_server_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "sample_calculator_mcp_server.py"))
        self.register_builtin(
            BuiltinStdioMCPServerConfig(
                name="calculator",
                command="python",
                args=[calc_server_path],
                description="Sample calculator MCP server for basic math operations"
            )
        )
        
        verbose_logger.debug("Using default builtin MCP servers")
    
    def register_builtin(self, config: Union[BuiltinMCPServerConfig, BuiltinStdioMCPServerConfig]):
        """Register a new built-in MCP server configuration"""
        self._builtin_configs[config.name] = config
    
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


# Global registry instance - will be initialized with config during proxy startup
global_builtin_registry: Optional[BuiltinMCPRegistry] = None

def initialize_builtin_registry(config_builtin_servers: Optional[Dict[str, Dict]] = None) -> BuiltinMCPRegistry:
    """Initialize the global builtin registry with config"""
    global global_builtin_registry
    global_builtin_registry = BuiltinMCPRegistry(config_builtin_servers)
    return global_builtin_registry

def get_builtin_registry() -> BuiltinMCPRegistry:
    """Get the global builtin registry, initializing with defaults if needed"""
    global global_builtin_registry
    if global_builtin_registry is None:
        global_builtin_registry = BuiltinMCPRegistry()  # Initialize with defaults
    return global_builtin_registry