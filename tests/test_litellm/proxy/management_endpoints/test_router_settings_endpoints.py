"""
Tests for router settings management endpoints.

Tests the GET endpoints for router settings and router fields.
"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)

from litellm.proxy.proxy_server import app

client = TestClient(app)


class TestRouterSettingsEndpoints:
    """Test suite for router settings endpoints"""

    @pytest.mark.asyncio
    async def test_get_router_fields_success(self):
        """
        Test GET /router/fields endpoint successfully returns field definitions without values.
        """
        # Make request to router fields endpoint
        response = client.get(
            "/router/fields",
            headers={"Authorization": "Bearer sk-1234"}
        )

        # Verify response
        assert response.status_code == 200
        
        response_data = response.json()
        
        # Verify response structure
        assert "fields" in response_data
        assert "routing_strategy_descriptions" in response_data
        
        # Verify fields is a list
        assert isinstance(response_data["fields"], list)
        assert len(response_data["fields"]) > 0
        
        # Verify each field has required properties and field_value is None
        for field in response_data["fields"]:
            assert "field_name" in field
            assert "field_type" in field
            assert "field_description" in field
            assert "field_default" in field
            assert "ui_field_name" in field
            assert "field_value" in field
            assert field["field_value"] is None  # Ensure field_value is None
        
        # Verify routing_strategy_descriptions is a dict
        assert isinstance(response_data["routing_strategy_descriptions"], dict)
        assert len(response_data["routing_strategy_descriptions"]) > 0
        
        # Verify routing_strategy field has options populated
        routing_strategy_field = next(
            (f for f in response_data["fields"] if f["field_name"] == "routing_strategy"),
            None
        )
        assert routing_strategy_field is not None
        assert "options" in routing_strategy_field
        assert isinstance(routing_strategy_field["options"], list)
        assert len(routing_strategy_field["options"]) > 0
