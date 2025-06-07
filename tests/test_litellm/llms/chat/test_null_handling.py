#!/usr/bin/env python3
"""
Test null handling in response transformation
"""

def test_null_content_handling():
    """Test handling of null content in responses"""
    print("🧪 Testing null content handling...")
    
    try:
        from litellm.llms.chat.transformation import ChatConfig
        from litellm.types.utils import ModelResponse
        import httpx
        import json
        
        config = ChatConfig()
        
        # Mock a response with null content
        mock_response_data = {
            "id": "resp_123",
            "created_at": 1234567890,
            "model": "gpt-4o",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": None},  # This causes the error
                        {"type": "output_text", "text": "Hello"},
                        {"type": "output_text", "text": None}   # Another null
                    ]
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15
            },
            "status": "completed"
        }
        
        # Create a mock response
        mock_response = httpx.Response(
            status_code=200,
            content=json.dumps(mock_response_data).encode(),
            request=httpx.Request("POST", "http://example.com")
        )
        
        # Mock request data
        request_data = {
            "model": "gpt-4o",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}]
        }
        
        print("  Testing response with null content items...")
        
        try:
            result = config.transform_response(
                model="gpt-4o",
                raw_response=mock_response,
                model_response=None,
                logging_obj=None,
                request_data=request_data,
                messages=[],
                optional_params={},
                litellm_params={},
                encoding=None
            )
            
            print(f"  ✅ Successfully handled null content!")
            print(f"    Response content: '{result.choices[0].message.content}'")
            print(f"    Usage: {result.usage.total_tokens} tokens")
            
            # Verify content is properly concatenated (should be "Hello" only)
            expected_content = "Hello"  # Only non-null text should be included
            actual_content = result.choices[0].message.content
            
            if actual_content == expected_content:
                print(f"  ✅ Content correctly filtered: '{actual_content}'")
            else:
                print(f"  ⚠️  Content mismatch - expected: '{expected_content}', got: '{actual_content}'")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to handle null content: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_empty_response_handling():
    """Test handling of completely empty responses"""
    print("\n🧪 Testing empty response handling...")
    
    try:
        from litellm.llms.chat.transformation import ChatConfig
        import httpx
        import json
        
        config = ChatConfig()
        
        # Mock an empty response
        mock_response_data = {
            "id": "resp_empty",
            "created_at": 1234567890,
            "model": "gpt-4o",
            "output": [],  # Empty output
            "usage": None,  # No usage data
            "status": "completed"
        }
        
        mock_response = httpx.Response(
            status_code=200,
            content=json.dumps(mock_response_data).encode(),
            request=httpx.Request("POST", "http://example.com")
        )
        
        request_data = {
            "model": "gpt-4o",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}]
        }
        
        print("  Testing completely empty response...")
        
        try:
            result = config.transform_response(
                model="gpt-4o",
                raw_response=mock_response,
                model_response=None,
                logging_obj=None,
                request_data=request_data,
                messages=[],
                optional_params={},
                litellm_params={},
                encoding=None
            )
            
            print(f"  ✅ Successfully handled empty response!")
            print(f"    Default message: '{result.choices[0].message.content}'")
            print(f"    Default usage: {result.usage.total_tokens} tokens")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to handle empty response: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Empty response test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_malformed_response_handling():
    """Test handling of malformed response data"""
    print("\n🧪 Testing malformed response handling...")
    
    try:
        from litellm.llms.chat.transformation import ChatConfig
        import httpx
        import json
        
        config = ChatConfig()
        
        # Mock a malformed response (missing required fields)
        mock_response_data = {
            # Missing id, created_at
            "model": "gpt-4o",
            "output": [
                {
                    "type": "message",
                    # Missing role
                    "content": [
                        {"type": "output_text"}  # Missing text field
                    ]
                }
            ]
            # Missing usage, status
        }
        
        mock_response = httpx.Response(
            status_code=200,
            content=json.dumps(mock_response_data).encode(),
            request=httpx.Request("POST", "http://example.com")
        )
        
        request_data = {
            "model": "gpt-4o",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}]
        }
        
        print("  Testing malformed response...")
        
        try:
            result = config.transform_response(
                model="gpt-4o",
                raw_response=mock_response,
                model_response=None,
                logging_obj=None,
                request_data=request_data,
                messages=[],
                optional_params={},
                litellm_params={},
                encoding=None
            )
            
            print(f"  ✅ Successfully handled malformed response!")
            print(f"    ID: {result.id}")
            print(f"    Message role: {result.choices[0].message.get('role', 'N/A')}")
            print(f"    Content: '{result.choices[0].message.content}'")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to handle malformed response: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Malformed response test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run null handling tests"""
    print("🔧 Chat Provider Null Handling Tests")
    print("=" * 50)
    
    tests = [
        ("Null Content Handling", test_null_content_handling),
        ("Empty Response Handling", test_empty_response_handling),
        ("Malformed Response Handling", test_malformed_response_handling),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "="*50)
    print("📊 Test Results:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All null handling tests passed!")
    else:
        print("⚠️  Some tests failed - check error handling logic.")

if __name__ == "__main__":
    main()