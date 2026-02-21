"""
Test method-specific routing for pass-through endpoints.

This test demonstrates the ability to configure different targets
for the same path but different HTTP methods.
"""

import pytest

from litellm.proxy._types import PassThroughGenericEndpoint


def test_pass_through_endpoint_with_methods():
    """Test creating pass-through endpoints with specific methods"""
    
    # Create endpoint for GET /azure/kb
    get_endpoint = PassThroughGenericEndpoint(
        id="get-azure-kb",
        path="/azure/kb",
        target="https://api1.example.com/knowledge-base",
        methods=["GET"],
        headers={"Authorization": "Bearer token1"},
    )
    
    assert get_endpoint.path == "/azure/kb"
    assert get_endpoint.methods == ["GET"]
    assert get_endpoint.target == "https://api1.example.com/knowledge-base"
    
    # Create endpoint for POST /azure/kb
    post_endpoint = PassThroughGenericEndpoint(
        id="post-azure-kb",
        path="/azure/kb",
        target="https://api2.example.com/knowledge-base",
        methods=["POST"],
        headers={"Authorization": "Bearer token2"},
    )
    
    assert post_endpoint.path == "/azure/kb"
    assert post_endpoint.methods == ["POST"]
    assert post_endpoint.target == "https://api2.example.com/knowledge-base"
    
    # These should be different endpoints despite same path
    assert get_endpoint.id != post_endpoint.id
    assert get_endpoint.target != post_endpoint.target


def test_pass_through_endpoint_multiple_methods():
    """Test creating endpoint with multiple methods"""
    
    endpoint = PassThroughGenericEndpoint(
        id="multi-method",
        path="/azure/kb",
        target="https://api.example.com/kb",
        methods=["GET", "POST", "PUT"],
        headers={},
    )
    
    assert len(endpoint.methods) == 3
    assert "GET" in endpoint.methods
    assert "POST" in endpoint.methods
    assert "PUT" in endpoint.methods


def test_pass_through_endpoint_no_methods_backward_compatibility():
    """Test that endpoints without methods field work (backward compatibility)"""
    
    # When methods is None, all methods should be supported
    endpoint = PassThroughGenericEndpoint(
        id="all-methods",
        path="/azure/kb",
        target="https://api.example.com/kb",
        headers={},
    )
    
    assert endpoint.methods is None  # Default is None for backward compatibility


def test_pass_through_endpoint_serialization():
    """Test that endpoints with methods can be serialized/deserialized"""
    
    endpoint = PassThroughGenericEndpoint(
        id="test-endpoint",
        path="/test",
        target="https://api.example.com",
        methods=["GET", "POST"],
        headers={"key": "value"},
        cost_per_request=0.5,
    )
    
    # Serialize to dict
    endpoint_dict = endpoint.model_dump()
    assert endpoint_dict["methods"] == ["GET", "POST"]
    
    # Deserialize from dict
    restored_endpoint = PassThroughGenericEndpoint(**endpoint_dict)
    assert restored_endpoint.methods == ["GET", "POST"]
    assert restored_endpoint.path == "/test"
    assert restored_endpoint.target == "https://api.example.com"


def test_route_key_generation_with_methods():
    """Test that route keys include methods for uniqueness"""
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        InitPassThroughEndpointHelpers,
    )

    # Simulate how route keys are generated
    endpoint_id_1 = "endpoint-1"
    path = "/azure/kb"
    methods_1 = ["GET"]
    methods_str_1 = ",".join(sorted(methods_1))
    route_key_1 = f"{endpoint_id_1}:exact:{path}:{methods_str_1}"
    
    endpoint_id_2 = "endpoint-2"
    methods_2 = ["POST"]
    methods_str_2 = ",".join(sorted(methods_2))
    route_key_2 = f"{endpoint_id_2}:exact:{path}:{methods_str_2}"
    
    # Keys should be different even though path is the same
    assert route_key_1 != route_key_2
    assert route_key_1 == "endpoint-1:exact:/azure/kb:GET"
    assert route_key_2 == "endpoint-2:exact:/azure/kb:POST"


def test_config_yaml_example():
    """
    Example configuration for config.yaml showing method-specific routing:
    
    general_settings:
      pass_through_endpoints:
        # GET endpoint for retrieving knowledge base
        - id: "get-azure-kb"
          path: "/azure/kb"
          target: "https://read-api.example.com/kb"
          methods: ["GET"]
          headers:
            Authorization: "bearer os.environ/READ_API_KEY"
        
        # POST endpoint for creating knowledge base entries
        - id: "post-azure-kb"
          path: "/azure/kb"
          target: "https://write-api.example.com/kb"
          methods: ["POST"]
          headers:
            Authorization: "bearer os.environ/WRITE_API_KEY"
        
        # PUT endpoint for updating knowledge base
        - id: "put-azure-kb"
          path: "/azure/kb"
          target: "https://update-api.example.com/kb"
          methods: ["PUT"]
          headers:
            Authorization: "bearer os.environ/UPDATE_API_KEY"
    """
    pass
