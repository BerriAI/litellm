from datetime import datetime
from typing import List, Optional
import pytest
import uuid
import os
import asyncio
from unittest import mock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from starlette import status

from litellm.constants import LITELLM_PROXY_ADMIN_NAME
from litellm.proxy._types import MCPSpecVersion, MCPSpecVersionType, MCPTransportType, MCPTransport, NewMCPServerRequest, LiteLLM_MCPServerTable, LitellmUserRoles, UserAPIKeyAuth
from litellm.types.mcp import MCPAuth
from litellm.proxy.management_endpoints.mcp_management_endpoints import does_mcp_server_exist

TEST_MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-1234")

def generate_mcpserver_record(url: Optional[str] = None, 
                    transport: Optional[MCPTransportType] = None,
                    spec_version: Optional[MCPSpecVersionType] = None) -> LiteLLM_MCPServerTable:
    """
    Generate a mock record for testing.
    """
    now = datetime.now()

    return LiteLLM_MCPServerTable(
        server_id=str(uuid.uuid4()),
        alias="Test Server",
        url=url or "http://localhost.com:8080/mcp",
        transport=transport or MCPTransport.sse,
        spec_version=spec_version or MCPSpecVersion.mar_2025,
        created_at=now,
        updated_at=now,
    )

# Cheers SO
def is_valid_uuid(val):
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False

def generate_mcpserver_create_request(
                    server_id: Optional[str] = None,
                    url: Optional[str] = None, 
                    transport: Optional[MCPTransportType] = None,
                    spec_version: Optional[MCPSpecVersionType] = None) -> NewMCPServerRequest:
    """
    Generate a mock create request for testing.
    """
    return NewMCPServerRequest(
        server_id=server_id,
        alias="Test Server",
        url=url or "http://localhost.com:8080/mcp",
        transport=transport or MCPTransport.sse,
        spec_version=spec_version or MCPSpecVersion.mar_2025,
    )

def assert_mcp_server_record_same(mcp_server: NewMCPServerRequest, resp: LiteLLM_MCPServerTable):
    """
    Assert that the mcp server record is created correctly.
    """
    if mcp_server.server_id is not None:
        assert resp.server_id == mcp_server.server_id
    else:
        assert is_valid_uuid(resp.server_id)
    assert resp.alias == mcp_server.alias
    assert resp.url == mcp_server.url
    assert resp.description == mcp_server.description
    assert resp.transport == mcp_server.transport
    assert resp.spec_version == mcp_server.spec_version
    assert resp.auth_type == mcp_server.auth_type
    assert resp.created_at is not None
    assert resp.updated_at is not None
    assert resp.created_by == LITELLM_PROXY_ADMIN_NAME
    assert resp.updated_by == LITELLM_PROXY_ADMIN_NAME


def test_does_mcp_server_exist():
    """
    Unit Test if the MCP server exists in the list.
    """
    mcp_server_records: List[LiteLLM_MCPServerTable] = [generate_mcpserver_record(), generate_mcpserver_record()]
    # test all records are found
    for record in mcp_server_records:
        assert does_mcp_server_exist(mcp_server_records, record.server_id)
    
    # test record not found
    not_found_record = str(uuid.uuid4())
    assert False == does_mcp_server_exist(mcp_server_records, not_found_record)

@pytest.mark.asyncio
async def test_create_mcp_server_direct():
    """
    Direct test of the MCP server creation logic without HTTP calls.
    """
    # Mock the database functions directly
    with mock.patch("litellm.proxy.management_endpoints.mcp_management_endpoints.MCP_AVAILABLE", True), \
         mock.patch("litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw") as mock_get_prisma, \
         mock.patch("litellm.proxy.management_endpoints.mcp_management_endpoints.create_mcp_server") as mock_create, \
         mock.patch("litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server") as mock_get_server, \
         mock.patch("litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager") as mock_manager:
        
        # Import after mocking
        from litellm.proxy.management_endpoints.mcp_management_endpoints import add_mcp_server
        
        # Mock database client
        mock_prisma = mock.Mock()
        mock_get_prisma.return_value = mock_prisma
        
        # Mock server manager
        mock_manager.add_update_server = mock.Mock()
        
        # Set up test data
        server_id = str(uuid.uuid4())
        mcp_server_request = generate_mcpserver_create_request(server_id=server_id)
        
        expected_response = LiteLLM_MCPServerTable(
            server_id=server_id,
            alias=mcp_server_request.alias,
            description=mcp_server_request.description,
            url=mcp_server_request.url,
            transport=mcp_server_request.transport,
            spec_version=mcp_server_request.spec_version,
            auth_type=mcp_server_request.auth_type,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by=LITELLM_PROXY_ADMIN_NAME,
            updated_by=LITELLM_PROXY_ADMIN_NAME,
            teams=[]
        )
        
        # Mock the database calls
        mock_get_server.return_value = None  # Server doesn't exist yet
        mock_create.return_value = expected_response
        
        # Create mock user auth
        user_auth = UserAPIKeyAuth(
            api_key=TEST_MASTER_KEY,
            user_id="test-user",
            user_role=LitellmUserRoles.PROXY_ADMIN
        )
        
        # Call the function directly
        result = await add_mcp_server(
            payload=mcp_server_request,
            user_api_key_dict=user_auth
        )
        
        # Verify the result
        assert result.server_id == server_id
        assert result.alias == mcp_server_request.alias
        assert result.url == mcp_server_request.url
        assert result.transport == mcp_server_request.transport
        assert result.spec_version == mcp_server_request.spec_version
        
        # Verify mocks were called
        mock_get_server.assert_called_once_with(mock_prisma, server_id)
        mock_create.assert_called_once()
        mock_manager.add_update_server.assert_called_once_with(expected_response)

@pytest.mark.asyncio 
async def test_create_duplicate_mcp_server():
    """
    Test that creating a duplicate MCP server fails appropriately.
    """
    # Mock the database functions directly
    with mock.patch("litellm.proxy.management_endpoints.mcp_management_endpoints.MCP_AVAILABLE", True), \
         mock.patch("litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw") as mock_get_prisma, \
         mock.patch("litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server") as mock_get_server:
        
        # Import after mocking
        from litellm.proxy.management_endpoints.mcp_management_endpoints import add_mcp_server
        from fastapi import HTTPException
        
        # Mock database client
        mock_prisma = mock.Mock()
        mock_get_prisma.return_value = mock_prisma
        
        # Set up test data
        server_id = str(uuid.uuid4())
        mcp_server_request = generate_mcpserver_create_request(server_id=server_id)
        
        existing_server = LiteLLM_MCPServerTable(
            server_id=server_id,
            alias="Existing Server",
            url="http://existing.com",
            transport=MCPTransport.sse,
            spec_version=MCPSpecVersion.mar_2025,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            teams=[]
        )
        
        # Mock that server already exists
        mock_get_server.return_value = existing_server
        
        # Create mock user auth
        user_auth = UserAPIKeyAuth(
            api_key=TEST_MASTER_KEY,
            user_id="test-user",
            user_role=LitellmUserRoles.PROXY_ADMIN
        )
        
        # Expect HTTPException to be raised
        with pytest.raises(HTTPException) as exc_info:
            await add_mcp_server(
                payload=mcp_server_request,
                user_api_key_dict=user_auth
            )
        
        # Verify the exception details
        assert exc_info.value.status_code == 400
        assert "already exists" in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_create_mcp_server_auth_failure():
    """
    Test that non-admin users cannot create MCP servers.
    """
    # Mock the database functions directly
    with mock.patch("litellm.proxy.management_endpoints.mcp_management_endpoints.MCP_AVAILABLE", True), \
         mock.patch("litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw") as mock_get_prisma:
        
        # Import after mocking
        from litellm.proxy.management_endpoints.mcp_management_endpoints import add_mcp_server
        from fastapi import HTTPException
        
        # Mock database client 
        mock_prisma = mock.Mock()
        mock_get_prisma.return_value = mock_prisma
        
        # Set up test data
        server_id = str(uuid.uuid4())
        mcp_server_request = generate_mcpserver_create_request(server_id=server_id)
        
        # Create mock user auth without admin role
        user_auth = UserAPIKeyAuth(
            api_key=TEST_MASTER_KEY,
            user_id="test-user",
            user_role=LitellmUserRoles.INTERNAL_USER  # Not an admin
        )
        
        # Expect HTTPException to be raised
        with pytest.raises(HTTPException) as exc_info:
            await add_mcp_server(
                payload=mcp_server_request,
                user_api_key_dict=user_auth
            )
        
        # Verify the exception details
        assert exc_info.value.status_code == 403
        assert "permission" in str(exc_info.value.detail)

@pytest.mark.asyncio 
async def test_create_mcp_server_invalid_alias():
    """
    Test that creating an MCP server with a '-' in the alias fails with the correct error.
    """
    with mock.patch("litellm.proxy.management_endpoints.mcp_management_endpoints.MCP_AVAILABLE", True), \
         mock.patch("litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw") as mock_get_prisma, \
         mock.patch("litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server") as mock_get_server:
        
        from litellm.proxy.management_endpoints.mcp_management_endpoints import add_mcp_server
        from fastapi import HTTPException
        
        mock_prisma = mock.Mock()
        mock_get_prisma.return_value = mock_prisma
        
        # Set up test data with invalid alias
        server_id = str(uuid.uuid4())
        mcp_server_request = generate_mcpserver_create_request(server_id=server_id)
        mcp_server_request.alias = "invalid-alias"  # This should trigger the validation error
        
        # Mock that server does not exist
        mock_get_server.return_value = None
        
        user_auth = UserAPIKeyAuth(
            api_key=TEST_MASTER_KEY,
            user_id="test-user",
            user_role=LitellmUserRoles.PROXY_ADMIN
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await add_mcp_server(
                payload=mcp_server_request,
                user_api_key_dict=user_auth
            )
        
        assert exc_info.value.status_code == 400
        assert "Server name cannot contain '-'. Use an alternative character instead Found: invalid-alias" in str(exc_info.value.detail)
