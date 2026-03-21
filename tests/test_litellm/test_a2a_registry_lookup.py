"""
Test A2A provider registry lookup functionality.

Maps to: litellm/llms/a2a/chat/transformation.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import pytest

import litellm
from litellm.llms.a2a.chat.transformation import A2AConfig


def test_resolve_agent_config_from_registry_static_method():
    """Test the static helper method for registry resolution"""
    
    # Test 1: No agent name in model
    api_base, api_key, headers = A2AConfig.resolve_agent_config_from_registry(
        model="a2a",
        api_base="http://test.com",
        api_key=None,
        headers=None,
        optional_params={}
    )
    assert api_base == "http://test.com"
    
    # Test 2: All params provided - should not lookup registry
    api_base, api_key, headers = A2AConfig.resolve_agent_config_from_registry(
        model="a2a/test-agent",
        api_base="http://explicit.com",
        api_key="explicit-key",
        headers={"X-Test": "value"},
        optional_params={}
    )
    assert api_base == "http://explicit.com"
    assert api_key == "explicit-key"


def test_a2a_registry_integration():
    """Test registry lookup in proxy context"""
    
    try:
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
        from litellm.types.agents import AgentResponse

        # Create test agent
        test_agent = AgentResponse(
            agent_id="test-id",
            agent_name="test-agent",
            agent_card_params={"url": "http://registry-url.example.com:9999"},
            litellm_params={"api_key": "registry-key"},
        )
        
        # Register and test
        original_agents = global_agent_registry.agent_list.copy()
        global_agent_registry.register_agent(test_agent)
        
        try:
            litellm.completion(
                model="a2a/test-agent",
                messages=[{"role": "user", "content": "Hello"}]
            )
        except Exception as e:
            # Should use registry URL (connection error expected)
            assert "registry-url.example.com" in str(e) or "APIConnectionError" in str(type(e).__name__)
        finally:
            global_agent_registry.agent_list = original_agents
            
    except ImportError:
        pytest.skip("Registry not available (not in proxy context)")
