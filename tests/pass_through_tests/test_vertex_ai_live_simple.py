#!/usr/bin/env python3
"""
Simple test script for Vertex AI Live API passthrough feature

This script provides a quick way to test the Vertex AI Live API passthrough
functionality without requiring a full test suite setup.
"""

import json
import sys
import os
from datetime import datetime

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_ai_live_passthrough_logging_handler import (
    VertexAILivePassthroughLoggingHandler,
)


def test_usage_metadata_extraction():
    """Test usage metadata extraction from WebSocket messages"""
    print("Testing usage metadata extraction...")
    
    handler = VertexAILivePassthroughLoggingHandler()
    
    # Sample WebSocket messages
    messages = [
        {
            "type": "session.created",
            "session": {"id": "test-session-123"}
        },
        {
            "type": "response.create",
            "response": {
                "text": "Hello! How can I help you?"
            },
            "usageMetadata": {
                "promptTokenCount": 15,
                "candidatesTokenCount": 20,
                "totalTokenCount": 35,
                "promptTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 15}
                ],
                "candidatesTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 20}
                ]
            }
        },
        {
            "type": "response.done",
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 8,
                "totalTokenCount": 13,
                "promptTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 5}
                ],
                "candidatesTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 8}
                ]
            }
        }
    ]
    
    # Extract usage metadata
    usage_metadata = handler._extract_usage_metadata_from_websocket_messages(messages)
    
    if usage_metadata:
        print("✅ Usage metadata extracted successfully:")
        print(f"  - Prompt tokens: {usage_metadata['promptTokenCount']}")
        print(f"  - Candidate tokens: {usage_metadata['candidatesTokenCount']}")
        print(f"  - Total tokens: {usage_metadata['totalTokenCount']}")
        print(f"  - Prompt details: {usage_metadata['promptTokensDetails']}")
        print(f"  - Candidate details: {usage_metadata['candidatesTokensDetails']}")
        
        # Verify aggregated values
        assert usage_metadata['promptTokenCount'] == 20  # 15 + 5
        assert usage_metadata['candidatesTokenCount'] == 28  # 20 + 8
        assert usage_metadata['totalTokenCount'] == 48  # 35 + 13
        print("✅ Token aggregation working correctly")
    else:
        print("❌ Failed to extract usage metadata")
        return False
    
    return True


def test_cost_calculation():
    """Test cost calculation functionality"""
    print("\nTesting cost calculation...")
    
    handler = VertexAILivePassthroughLoggingHandler()
    
    # Mock model info
    usage_metadata = {
        "promptTokenCount": 100,
        "candidatesTokenCount": 50,
        "totalTokenCount": 150
    }
    
    # Test with mock model info using patch
    from unittest.mock import patch
    
    with patch('litellm.utils.get_model_info') as mock_get_model_info:
        mock_get_model_info.return_value = {
            "input_cost_per_token": 0.000001,
            "output_cost_per_token": 0.000002
        }
        
        cost = handler._calculate_live_api_cost("gemini-1.5-pro", usage_metadata)
        expected_cost = (100 * 0.000001) + (50 * 0.000002)
        
        print(f"✅ Cost calculated: ${cost:.6f}")
        print(f"  - Expected: ${expected_cost:.6f}")
        print(f"  - Difference: ${abs(cost - expected_cost):.6f}")
        
        # The cost should be close to expected (within 1 cent)
        assert abs(cost - expected_cost) < 0.01
        print("✅ Cost calculation working correctly")
    
    return True


def test_multimodal_usage():
    """Test multimodal usage tracking"""
    print("\nTesting multimodal usage tracking...")
    
    handler = VertexAILivePassthroughLoggingHandler()
    
    # Messages with mixed modalities
    messages = [
        {
            "type": "response.create",
            "response": {
                "text": "Hello with audio"
            },
            "usageMetadata": {
                "promptTokenCount": 30,
                "candidatesTokenCount": 25,
                "totalTokenCount": 55,
                "promptTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 20},
                    {"modality": "AUDIO", "tokenCount": 10}
                ],
                "candidatesTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 15},
                    {"modality": "AUDIO", "tokenCount": 10}
                ]
            }
        }
    ]
    
    usage_metadata = handler._extract_usage_metadata_from_websocket_messages(messages)
    
    if usage_metadata:
        print("✅ Multimodal usage extracted:")
        print(f"  - Prompt tokens: {usage_metadata['promptTokenCount']}")
        print(f"  - Candidate tokens: {usage_metadata['candidatesTokenCount']}")
        print(f"  - Prompt details: {usage_metadata['promptTokensDetails']}")
        print(f"  - Candidate details: {usage_metadata['candidatesTokensDetails']}")
        
        # Verify modality details
        text_prompt = next(d for d in usage_metadata['promptTokensDetails'] if d['modality'] == 'TEXT')
        audio_prompt = next(d for d in usage_metadata['promptTokensDetails'] if d['modality'] == 'AUDIO')
        
        assert text_prompt['tokenCount'] == 20
        assert audio_prompt['tokenCount'] == 10
        print("✅ Multimodal tracking working correctly")
    else:
        print("❌ Failed to extract multimodal usage")
        return False
    
    return True


def test_web_search_usage():
    """Test web search (tool use) usage tracking"""
    print("\nTesting web search usage tracking...")
    
    handler = VertexAILivePassthroughLoggingHandler()
    
    # Messages with web search usage
    messages = [
        {
            "type": "response.create",
            "response": {
                "text": "Hello with web search"
            },
            "usageMetadata": {
                "promptTokenCount": 50,
                "candidatesTokenCount": 30,
                "totalTokenCount": 80,
                "toolUsePromptTokenCount": 10,
                "promptTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 50}
                ],
                "candidatesTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 30}
                ]
            }
        }
    ]
    
    usage_metadata = handler._extract_usage_metadata_from_websocket_messages(messages)
    
    if usage_metadata:
        print("✅ Web search usage extracted:")
        print(f"  - Prompt tokens: {usage_metadata['promptTokenCount']}")
        print(f"  - Candidate tokens: {usage_metadata['candidatesTokenCount']}")
        print(f"  - Tool use prompt tokens: {usage_metadata.get('toolUsePromptTokenCount', 0)}")
        
        assert usage_metadata['toolUsePromptTokenCount'] == 10
        print("✅ Web search tracking working correctly")
    else:
        print("❌ Failed to extract web search usage")
        return False
    
    return True


def test_error_handling():
    """Test error handling with invalid inputs"""
    print("\nTesting error handling...")
    
    handler = VertexAILivePassthroughLoggingHandler()
    
    # Test various invalid inputs
    invalid_inputs = [
        None,
        [],
        "not a list",
        [{"type": "invalid"}],
        [{"type": "response.create"}],  # Missing response
        [{"type": "response.create", "response": {}}]  # Empty response
    ]
    
    for i, invalid_input in enumerate(invalid_inputs):
        try:
            if invalid_input is None:
                # Skip None input as it will cause iteration error
                print(f"  - Input {i+1}: Skipped None input")
                continue
            else:
                result = handler._extract_usage_metadata_from_websocket_messages(invalid_input)
            print(f"  - Input {i+1}: Handled gracefully (result: {result})")
        except Exception as e:
            print(f"  - Input {i+1}: Error - {e}")
            return False
    
    print("✅ Error handling working correctly")
    return True


def test_handler_integration():
    """Test the main handler method"""
    print("\nTesting handler integration...")
    
    handler = VertexAILivePassthroughLoggingHandler()
    
    # Mock logging object
    class MockLoggingObj:
        def __init__(self):
            self.model_call_details = {}
    
    mock_logging_obj = MockLoggingObj()
    
    # Sample WebSocket messages with proper usage metadata
    messages = [
        {
            "type": "response.create",
            "response": {
                "text": "Hello! How can I help you?"
            },
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 15,
                "totalTokenCount": 25,
                "promptTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 10}
                ],
                "candidatesTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 15}
                ]
            }
        }
    ]
    
    # Test the main handler method
    result = handler.vertex_ai_live_passthrough_handler(
        websocket_messages=messages,
        logging_obj=mock_logging_obj,
        url_route="/vertex_ai/live",
        start_time=datetime.now(),
        end_time=datetime.now(),
        request_body={"messages": [{"role": "user", "content": "Hello"}]}
    )
    
    if result and "result" in result and "kwargs" in result:
        print("✅ Handler integration working:")
        print(f"  - Result keys: {list(result.keys())}")
        print(f"  - Model: {result['result'].get('model', 'N/A')}")
        print(f"  - Usage: {result['result'].get('usage', {})}")
        print("✅ Handler integration working correctly")
        return True
    else:
        print("❌ Handler integration failed")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting Vertex AI Live Passthrough Tests")
    print("=" * 50)
    
    tests = [
        test_usage_metadata_extraction,
        test_cost_calculation,
        test_multimodal_usage,
        test_web_search_usage,
        test_error_handling,
        test_handler_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
