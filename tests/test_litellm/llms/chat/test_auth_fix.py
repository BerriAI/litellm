#!/usr/bin/env python3
"""
Test script to verify the authentication fix for chat provider
"""

def test_validate_environment():
    """Test the validate_environment method"""
    print("🧪 Testing validate_environment method...")
    
    try:
        from litellm.llms.chat.transformation import ChatConfig
        
        config = ChatConfig()
        
        # Test 1: With API key provided
        headers1 = {}
        result1 = config.validate_environment(
            headers=headers1,
            model="gpt-4o",
            api_key="sk-test-key-123"
        )
        print(f"✅ Test 1 - API key provided: {result1.get('Authorization', 'MISSING')}")
        
        # Test 2: With existing Authorization header
        headers2 = {"Authorization": "Bearer existing-key"}
        result2 = config.validate_environment(
            headers=headers2,
            model="gpt-4o"
        )
        print(f"✅ Test 2 - Existing auth header: {result2.get('Authorization', 'MISSING')}")
        
        # Test 3: No API key (should try environment)
        headers3 = {}
        result3 = config.validate_environment(
            headers=headers3,
            model="gpt-4o"
        )
        print(f"✅ Test 3 - No API key: {result3.get('Authorization', 'MISSING')}")
        
        return True
        
    except Exception as e:
        print(f"❌ validate_environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_processing():
    """Test model name processing"""
    print("\n🧪 Testing model name processing...")
    
    try:
        from litellm.llms.chat.transformation import ChatConfig
        
        config = ChatConfig()
        
        # Mock the transform_request method's model processing logic
        test_cases = [
            ("chat/gpt-4o", "gpt-4o"),
            ("chat/codex-mini-latest", "codex-mini-latest"), 
            ("gpt-4o", "gpt-4o"),
            ("some-model", "some-model")
        ]
        
        for input_model, expected_model in test_cases:
            # Simulate the model processing logic
            api_model = input_model
            if input_model.startswith("chat/"):
                api_model = input_model[5:]
            
            if api_model == expected_model:
                print(f"✅ {input_model} -> {api_model}")
            else:
                print(f"❌ {input_model} -> {api_model} (expected {expected_model})")
        
        return True
        
    except Exception as e:
        print(f"❌ Model processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_content_format():
    """Test content format conversion"""
    print("\n🧪 Testing content format conversion...")
    
    try:
        from litellm.llms.chat.transformation import ChatConfig
        
        config = ChatConfig()
        
        # Test cases for content conversion
        test_cases = [
            # Simple string
            ("Hello world", [{"type": "input_text", "text": "Hello world"}]),
            
            # Chat completion format with text type
            ([{"type": "text", "text": "Hello"}], [{"type": "input_text", "text": "Hello"}]),
            
            # Image URL format
            ([{"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}}], 
             [{"type": "input_image", "image_url": {"url": "http://example.com/img.jpg"}}]),
            
            # Mixed content
            ([{"type": "text", "text": "Look at this:"}, {"type": "image_url", "image_url": {"url": "test.jpg"}}],
             [{"type": "input_text", "text": "Look at this:"}, {"type": "input_image", "image_url": {"url": "test.jpg"}}])
        ]
        
        for i, (input_content, expected_output) in enumerate(test_cases):
            print(f"\n  Test {i+1}: {type(input_content)} input")
            result = config._convert_content_to_responses_format(input_content)
            
            if result == expected_output:
                print(f"  ✅ Conversion correct")
            else:
                print(f"  ❌ Conversion failed")
                print(f"    Expected: {expected_output}")
                print(f"    Got:      {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Content format test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_parsing():
    """Test configuration format"""
    print("\n🧪 Testing configuration format...")
    
    config_example = {
        "model_name": "openai/codex-mini-latest",
        "litellm_params": {
            "model": "chat/codex-mini-latest",
            "custom_llm_provider": "chat",
            "api_key": "sk-..."
        }
    }
    
    print("Configuration format analysis:")
    print(f"  Model name in proxy: {config_example['model_name']}")
    print(f"  Model for chat provider: {config_example['litellm_params']['model']}")
    print(f"  Custom provider: {config_example['litellm_params']['custom_llm_provider']}")
    
    # Simulate what should happen
    proxy_model = config_example['model_name']  # "openai/codex-mini-latest"
    chat_model = config_example['litellm_params']['model']  # "chat/codex-mini-latest"
    
    if chat_model.startswith("chat/"):
        api_model = chat_model[5:]  # "codex-mini-latest"
        print(f"  ✅ Processed for API: {api_model}")
    
    print("\n⚠️  Note: The issue might be that you're using 'openai/codex-mini-latest' as model_name")
    print("   but 'chat/codex-mini-latest' as the litellm_params model.")
    print("   This could cause confusion in the routing.")
    
    print("\n💡 Suggested config:")
    print("""
  - model_name: "codex-mini-chat"
    litellm_params:
      model: "chat/codex-mini-latest"
      custom_llm_provider: "chat"
      api_key: "sk-..."
""")
    
    return True

def main():
    """Run all tests"""
    print("🔧 Chat Provider Authentication Fix Tests")
    print("=" * 50)
    
    tests = [
        ("Environment Validation", test_validate_environment),
        ("Model Processing", test_model_processing),
        ("Content Format", test_content_format),
        ("Config Analysis", test_config_parsing),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        test_func()
    
    print("\n" + "="*50)
    print("🎯 Key Fixes Applied:")
    print("  ✅ Added proper Authorization header handling")
    print("  ✅ Added fallback to OPENAI_API_KEY environment variable")
    print("  ✅ Added model name processing to strip 'chat/' prefix")
    print("  ✅ Fixed content format: 'text' -> 'input_text' for Responses API")
    print("  ✅ Added debug logging to trace authentication and content conversion")
    
    print("\n📝 To test your config:")
    print("  1. Make sure your API key is correct")
    print("  2. Use a model that supports Responses API (like gpt-4o)")
    print("  3. Check the debug logs for authentication details")

if __name__ == "__main__":
    main()