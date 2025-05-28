#!/usr/bin/env python3
"""
Debug streaming issues with chat provider
"""

def test_streaming_response_transformation():
    """Test the streaming response transformation with various event types"""
    print("🧪 Testing streaming response transformation...")
    
    try:
        from litellm.llms.chat.transformation import ChatConfig
        
        config = ChatConfig()
        
        # Test various Responses API event types
        test_events = [
            # Response created event
            {
                "type": "response.created",
                "response": {
                    "id": "resp_123",
                    "created_at": 1234567890,
                    "model": "gpt-4o"
                }
            },
            
            # Output item added
            {
                "type": "response.output_item.added",
                "response_id": "resp_123",
                "output_item": {
                    "type": "message",
                    "role": "assistant"
                }
            },
            
            # Content part added
            {
                "type": "response.content_part.added",
                "response_id": "resp_123",
                "part": {
                    "type": "text",
                    "text": "Hello"
                }
            },
            
            # Content part done
            {
                "type": "response.content_part.done",
                "response_id": "resp_123"
            },
            
            # Response done
            {
                "type": "response.done",
                "response": {
                    "id": "resp_123",
                    "created_at": 1234567890,
                    "model": "gpt-4o",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15
                    }
                }
            },
            
            # Unknown event type
            {
                "type": "response.unknown_event",
                "response_id": "resp_123",
                "data": "test"
            },
            
            # Empty chunk
            None,
            
            # Invalid chunk
            "invalid"
        ]
        
        for i, event in enumerate(test_events):
            print(f"\n  Test {i+1}: {event.get('type') if isinstance(event, dict) else type(event)}")
            
            try:
                result = config.transform_streaming_response("gpt-4o", event, None)
                
                if result is None:
                    print(f"    ✅ Returned None (acceptable for some events)")
                elif isinstance(result, dict) and "object" in result:
                    print(f"    ✅ Valid chunk: {result.get('object')}")
                    if "choices" in result:
                        delta = result["choices"][0].get("delta", {})
                        if "content" in delta:
                            print(f"    📝 Content: '{delta['content']}'")
                        if "role" in delta:
                            print(f"    👤 Role: {delta['role']}")
                else:
                    print(f"    ⚠️  Unexpected result: {type(result)}")
                    
            except Exception as e:
                print(f"    ❌ Exception: {e}")
                import traceback
                traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stream_request_setup():
    """Test how streaming requests are set up"""
    print("\n🧪 Testing stream request setup...")
    
    try:
        from litellm.llms.chat.transformation import ChatConfig
        
        config = ChatConfig()
        
        # Test streaming request transformation
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {"stream": True, "max_tokens": 100}
        litellm_params = {"custom_llm_provider": "chat"}
        headers = {}
        
        print("  Input parameters:")
        print(f"    messages: {messages}")
        print(f"    optional_params: {optional_params}")
        print(f"    litellm_params: {litellm_params}")
        
        result = config.transform_request(
            model="gpt-4o",
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers
        )
        
        print(f"\n  ✅ Request transformation successful!")
        print(f"    Stream parameter: {result.get('stream')}")
        print(f"    Model: {result.get('model')}")
        print(f"    Input items: {len(result.get('input', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Stream request test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run streaming debug tests"""
    print("🔧 Chat Provider Streaming Debug Tests")
    print("=" * 50)
    
    tests = [
        ("Streaming Response Transformation", test_streaming_response_transformation),
        ("Stream Request Setup", test_stream_request_setup),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        test_func()
    
    print("\n" + "="*50)
    print("🎯 Debugging Notes:")
    print("  • Check for 'NoneType' object is not an iterator errors")
    print("  • Verify streaming events are properly formatted")
    print("  • Ensure transform_streaming_response handles all cases")
    print("  • Look for missing stream parameter in requests")

if __name__ == "__main__":
    main()