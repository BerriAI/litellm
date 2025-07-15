"""
Unit tests for GetRoutes utility class.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

import pytest

from litellm.proxy.common_utils.get_routes import GetRoutes


class TestGetRoutes:
    
    def test_get_app_routes_regular_route(self):
        """Test getting routes for a regular route with endpoint."""
        # Mock a regular route
        mock_route = Mock()
        mock_route.path = "/test/endpoint"
        mock_route.methods = ["GET", "POST"]
        mock_route.name = "test_endpoint"
        mock_route.endpoint = Mock()
        
        # Mock endpoint function
        mock_endpoint = Mock()
        mock_endpoint.__name__ = "test_function"
        
        result = GetRoutes.get_app_routes(mock_route, mock_endpoint)
        
        assert len(result) == 1
        assert result[0]["path"] == "/test/endpoint"
        assert result[0]["methods"] == ["GET", "POST"]
        assert result[0]["name"] == "test_endpoint"
        assert result[0]["endpoint"] == "test_function"
    
    def test_get_routes_for_mounted_app_regular_routes(self):
        """Test getting routes for mounted app with regular API routes."""
        # Mock the main mount route
        mock_mount_route = Mock()
        mock_mount_route.path = "/mcp"
        
        # Mock sub-app with regular routes
        mock_sub_app = Mock()
        mock_sub_app.routes = []
        
        # Create a regular API route
        mock_api_route = Mock()
        mock_api_route.path = "/enabled"
        mock_api_route.methods = ["GET"]
        mock_api_route.name = "get_mcp_server_enabled"
        
        # Mock endpoint function
        mock_endpoint = Mock()
        mock_endpoint.__name__ = "get_mcp_server_enabled"
        mock_api_route.endpoint = mock_endpoint
        mock_api_route.app = None  # Regular route doesn't have app
        
        mock_sub_app.routes.append(mock_api_route)
        mock_mount_route.app = mock_sub_app
        
        result = GetRoutes.get_routes_for_mounted_app(mock_mount_route)
        
        assert len(result) == 1
        assert result[0]["path"] == "/mcp/enabled"
        assert result[0]["methods"] == ["GET"]
        assert result[0]["name"] == "get_mcp_server_enabled"
        assert result[0]["endpoint"] == "get_mcp_server_enabled"
        assert result[0]["mounted_app"] is True
    
    def test_get_routes_for_mounted_app_mount_objects(self):
        """Test getting routes for mounted app with Mount objects (the main fix)."""
        # Mock the main mount route
        mock_mount_route = Mock()
        mock_mount_route.path = "/mcp"
        
        # Mock sub-app
        mock_sub_app = Mock()
        mock_sub_app.routes = []
        
        # Create Mount object for base MCP route (path='')
        mock_mount_base = Mock(spec=['path', 'name', 'endpoint', 'app'])
        mock_mount_base.path = ""
        mock_mount_base.name = ""
        mock_mount_base.endpoint = None  # Mount objects don't have endpoint
        
        # Mock app function
        mock_app_function = Mock()
        mock_app_function.__name__ = "handle_streamable_http_mcp"
        mock_mount_base.app = mock_app_function
        
        # Create Mount object for SSE route (path='/sse')
        mock_mount_sse = Mock(spec=['path', 'name', 'endpoint', 'app'])
        mock_mount_sse.path = "/sse"
        mock_mount_sse.name = ""
        mock_mount_sse.endpoint = None  # Mount objects don't have endpoint
        
        # Mock app function for SSE
        mock_sse_function = Mock()
        mock_sse_function.__name__ = "handle_sse_mcp"
        mock_mount_sse.app = mock_sse_function
        
        mock_sub_app.routes.extend([mock_mount_base, mock_mount_sse])
        mock_mount_route.app = mock_sub_app
        
        result = GetRoutes.get_routes_for_mounted_app(mock_mount_route)
        
        # Should capture both /mcp and /mcp/sse routes
        assert len(result) == 2
        
        # Check base MCP route
        base_route = next(r for r in result if r["path"] == "/mcp")
        assert base_route["methods"] == ["GET", "POST"]  # Default methods
        assert base_route["endpoint"] == "handle_streamable_http_mcp"
        assert base_route["mounted_app"] is True
        
        # Check SSE route
        sse_route = next(r for r in result if r["path"] == "/mcp/sse")
        assert sse_route["methods"] == ["GET", "POST"]  # Default methods
        assert sse_route["endpoint"] == "handle_sse_mcp"
        assert sse_route["mounted_app"] is True
    
    def test_get_routes_for_mounted_app_mixed_routes(self):
        """Test getting routes for mounted app with both regular routes and Mount objects."""
        # Mock the main mount route
        mock_mount_route = Mock()
        mock_mount_route.path = "/mcp"
        
        # Mock sub-app
        mock_sub_app = Mock()
        mock_sub_app.routes = []
        
        # Create a regular API route
        mock_api_route = Mock()
        mock_api_route.path = "/enabled"
        mock_api_route.methods = ["GET"]
        mock_api_route.name = "get_mcp_server_enabled"
        mock_endpoint = Mock()
        mock_endpoint.__name__ = "get_mcp_server_enabled"
        mock_api_route.endpoint = mock_endpoint
        mock_api_route.app = None
        
        # Create Mount object
        mock_mount_base = Mock(spec=['path', 'name', 'endpoint', 'app'])
        mock_mount_base.path = ""
        mock_mount_base.name = ""
        mock_mount_base.endpoint = None
        mock_app_function = Mock()
        mock_app_function.__name__ = "handle_streamable_http_mcp"
        mock_mount_base.app = mock_app_function
        
        mock_sub_app.routes.extend([mock_api_route, mock_mount_base])
        mock_mount_route.app = mock_sub_app
        
        result = GetRoutes.get_routes_for_mounted_app(mock_mount_route)
        
        # Should capture both the API route and the Mount object
        assert len(result) == 2
        
        # Check API route
        api_route = next(r for r in result if r["path"] == "/mcp/enabled")
        assert api_route["methods"] == ["GET"]
        assert api_route["endpoint"] == "get_mcp_server_enabled"
        
        # Check Mount object route
        mount_route = next(r for r in result if r["path"] == "/mcp")
        assert mount_route["endpoint"] == "handle_streamable_http_mcp"
        assert mount_route["mounted_app"] is True

