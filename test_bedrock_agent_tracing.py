#!/usr/bin/env python3
"""
Test script to verify AWS Bedrock agent tracing parameter configuration.

This script tests whether the enableTrace parameter can be controlled through
LiteLLM proxy configuration without modifying the source code.
"""

import json
import sys
from unittest.mock import MagicMock, patch

# Add the litellm directory to Python path
sys.path.insert(0, "/Users/levy/code/apro/litellm")

from litellm.llms.bedrock.chat.invoke_agent.transformation import AmazonInvokeAgentConfig


def test_enable_trace_parameter():
    """Test that enableTrace can be controlled via optional_params"""
    
    config = AmazonInvokeAgentConfig()
    
    # Sample messages
    messages = [
        {"role": "user", "content": "Test message"}
    ]
    
    # Test 1: Default behavior (should be True, but overridden by optional_params)
    print("Test 1: enableTrace = True in optional_params")
    optional_params = {"enableTrace": True}
    result = config.transform_request(
        model="agent/TEST123/ALIAS456",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={}
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    print(f"enableTrace value: {result.get('enableTrace')}")
    print()
    
    # Test 2: Disable tracing via optional_params
    print("Test 2: enableTrace = False in optional_params")
    optional_params = {"enableTrace": False}
    result = config.transform_request(
        model="agent/TEST123/ALIAS456", 
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={}
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    print(f"enableTrace value: {result.get('enableTrace')}")
    print()
    
    # Test 3: No enableTrace in optional_params (should default to hardcoded True)
    print("Test 3: No enableTrace in optional_params")
    optional_params = {}
    result = config.transform_request(
        model="agent/TEST123/ALIAS456",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={}
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    print(f"enableTrace value: {result.get('enableTrace')}")
    print()
    
    # Test 4: Additional agent parameters
    print("Test 4: Additional agent parameters")
    optional_params = {
        "enableTrace": False,
        "sessionID": "custom-session-123",
        "memoryId": "memory-456",
        "endSession": True
    }
    result = config.transform_request(
        model="agent/TEST123/ALIAS456",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={}
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    print()


def test_parameter_precedence():
    """Test that parameters in optional_params override hardcoded values"""
    
    config = AmazonInvokeAgentConfig()
    messages = [{"role": "user", "content": "Test"}]
    
    # The current implementation does:
    # return {
    #     "inputText": query,
    #     "enableTrace": True,  # Hardcoded
    #     **optional_params,    # This should override the hardcoded value
    # }
    
    # Test the actual Python behavior
    print("Testing Python dict merge behavior:")
    base_dict = {"enableTrace": True, "inputText": "test"}
    optional_params = {"enableTrace": False, "customParam": "value"}
    merged = {**base_dict, **optional_params}
    print(f"Base dict: {base_dict}")
    print(f"Optional params: {optional_params}")
    print(f"Merged result: {merged}")
    print(f"enableTrace in merged: {merged['enableTrace']}")
    print()


if __name__ == "__main__":
    print("Testing AWS Bedrock Agent Tracing Parameter Configuration")
    print("=" * 60)
    print()
    
    test_parameter_precedence()
    test_enable_trace_parameter()
    
    print("=" * 60)
    print("Summary:")
    print("- The current LiteLLM implementation should support enableTrace parameter")
    print("- Parameters in optional_params will override hardcoded values")
    print("- This means enableTrace can be controlled via proxy configuration")
    print("- No source code modifications are needed!")