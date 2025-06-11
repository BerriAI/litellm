from datetime import datetime
from typing import List, Optional
import pytest
import uuid
from httpx import AsyncClient
import uuid
import os

from starlette import status

from litellm.constants import LITELLM_PROXY_ADMIN_NAME
from litellm.proxy._types import MCPAuth, MCPSpecVersion, MCPSpecVersionType, MCPTransportType, MCPTransport, NewMCPServerRequest, LiteLLM_MCPServerTable
from litellm.proxy.management_endpoints.mcp_management_endpoints import does_mcp_server_exist

TEST_MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-1234")
PROXY_BASE_URL = os.getenv("PROXY_BASE_URL", "http://localhost:4000")

def generate_mcpserver_record(url: Optional[str] = None, 
                    transport: Optional[MCPTransportType] = None,
                    spec_version: Optional[MCPSpecVersionType] = None) -> LiteLLM_MCPServerTable:
    """
    Generate a mock record for testing.
    """
    now = datetime.now()

    return LiteLLM_MCPServerTable(
        server_id=str(uuid.uuid4()),alias="Test Server",url=url or "http://localhost.com:8080/mcp",transport=transport or MCPTransport.sse,spec_version=spec_version or MCPSpecVersion.mar_2025,created_at=now,updated_at=now,
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
    now = datetime.now()

    return NewMCPServerRequest(server_id=server_id,
        alias="Test Server",url=url or "http://localhost.com:8080/mcp",transport=transport or MCPTransport.sse,spec_version=spec_version or MCPSpecVersion.mar_2025,
    )

def get_http_client():
    """
    Create an HTTP client for making requests to the proxy server.
    """
    headers = {"Authorization": f"Bearer {TEST_MASTER_KEY}"}
    # headers = {"Authorization": f"x-litellm-api-key {TEST_MASTER_KEY}"}
    return AsyncClient(base_url=PROXY_BASE_URL), headers

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
async def test_create_get_delete():
    """
    Integration Test mcp servers can be created and returned correctly.
    1. Create a new mcp server with server id 
    2. Create another mcp server without server id
    2.1 Verify duplicate mcp server (server id) creation fails
    3. Verify first server has matching server id and second server has a new server id
    4. Verify both servers are in the full mcp server list
    5. Verify first server can be retrieved by server id
    6. Delete both mcp servers
    7. Verify both servers are no longer in the full mcp server list
    8. Verify both servers cannot be retrieved by server id
    """
    # client, headers = AsyncClient(base_url=PROXY_BASE_URL), headers
    client, headers = get_http_client()
    
    first_server_id = str(uuid.uuid4())
    first_server = generate_mcpserver_create_request(server_id=first_server_id)

    # Add new mcp server with server id
    first_create_response = await client.post(
        "/v1/mcp/server",
        json=first_server.json(),
        headers=headers,
    )

    # Validate that the response is as expected and the server is created
    assert status.HTTP_201_CREATED == first_create_response.status_code
    first_resp = LiteLLM_MCPServerTable(**first_create_response.json())
    assert_mcp_server_record_same(first_server, first_resp)

    # Create second mcp server without server id
    second_server = generate_mcpserver_create_request()
    second_create_response = await client.post(
        "/v1/mcp/server",
        json=second_server.json(),
        headers=headers,
    )
    assert status.HTTP_201_CREATED == second_create_response.status_code
    second_resp = LiteLLM_MCPServerTable(**second_create_response.json())
    assert_mcp_server_record_same(second_server, second_resp)

    # Try to create a duplicate mcp server
    duplicate_create_response = await client.post(
        "/v1/mcp/server",
        json=first_server.json(),
        headers=headers,
    )
    assert status.HTTP_400_BAD_REQUEST == duplicate_create_response.status_code

    # Validate that the servers are in the full mcp server list
    get_all_mcp_servers_response = await client.get(
        "/v1/mcp/server",
        headers=headers,
    )
    assert status.HTTP_200_OK == get_all_mcp_servers_response.status_code
    mcp_servers = [
        LiteLLM_MCPServerTable(**record) for record in get_all_mcp_servers_response.json()
    ]
    assert len(mcp_servers) >= 2
    assert does_mcp_server_exist(mcp_servers, first_resp.server_id)
    assert does_mcp_server_exist(mcp_servers, second_resp.server_id)
    
    # Validate that the first server can be retrieved by server id
    get_mcp_server_response = await client.get(
        f"/v1/mcp/server/{first_resp.server_id}",
        headers=headers,
    )
    assert status.HTTP_200_OK == get_mcp_server_response.status_code
    resp = LiteLLM_MCPServerTable(**get_mcp_server_response.json())
    assert_mcp_server_record_same(first_server, resp)
    
    # Delete the mcp servers
    delete_response = await client.delete(
        f"/v1/mcp/server/{first_resp.server_id}",
        headers=headers,
    )
    assert status.HTTP_202_ACCEPTED == delete_response.status_code
    delete_response = await client.delete(
        f"/v1/mcp/server/{second_resp.server_id}",
        headers=headers,
    )
    assert status.HTTP_202_ACCEPTED == delete_response.status_code

    # Validate that the servers are no longer in the full list
    get_all_mcp_servers_response = await client.get(
        "/v1/mcp/server",
        headers=headers,
    )
    assert status.HTTP_200_OK == get_all_mcp_servers_response.status_code
    mcp_servers = [
        LiteLLM_MCPServerTable(**record) for record in get_all_mcp_servers_response.json()
    ]
    assert not does_mcp_server_exist(mcp_servers, first_resp.server_id)
    assert not does_mcp_server_exist(mcp_servers, second_resp.server_id)
    
    # Validate that both servers cannot be retrieved by server id
    for server_id in [first_resp.server_id, second_resp.server_id]:
        get_mcp_server_response = await client.get(
            f"/v1/mcp/server/{server_id}",
            headers=headers,
        )
        assert status.HTTP_404_NOT_FOUND == get_mcp_server_response.status_code

@pytest.mark.asyncio
async def test_edit():
    """
    Integration Test mcp servers can be created and edited correctly.
    1. Create a new mcp server 
    2. Edit the server id
    3. Verify the mcp server's data is updated
    """
    # client, headers = AsyncClient(base_url=PROXY_BASE_URL), headers
    client, headers = get_http_client()
    
    mcp_server_request = generate_mcpserver_create_request()

    # Add new mcp server with server id
    first_create_response = await client.post(
        "/v1/mcp/server",
        json=mcp_server_request.json(),
        headers=headers,
    )

    # Validate that the response is as expected and the server is created
    assert status.HTTP_201_CREATED == first_create_response.status_code
    mcp_server_response = LiteLLM_MCPServerTable(**first_create_response.json())
    assert_mcp_server_record_same(mcp_server_request, mcp_server_response)

    # Update the mcp server
    mcp_server_request.server_id = mcp_server_response.server_id
    mcp_server_request.spec_version = MCPSpecVersion.nov_2024
    mcp_server_request.transport = MCPTransport.http
    mcp_server_request.description = "Some updated description"    
    mcp_server_request.url = "http://localhost.com:4040/mcp"
    mcp_server_request.auth_type = MCPAuth.basic

    # Try to edit the mcp server
    updated_response = await client.put(
        "/v1/mcp/server",
        json=mcp_server_request.json(),
        headers=headers,
    )
    assert status.HTTP_202_ACCEPTED == updated_response.status_code
    updated_server = LiteLLM_MCPServerTable(**updated_response.json())
    assert_mcp_server_record_same(mcp_server_request, updated_server)
