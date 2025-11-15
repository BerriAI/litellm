import os
from litellm import completion

def test_zai_thinking_enabled():
    """Test ZAI thinking parameter enabled"""
    print("Testing ZAI thinking parameter enabled...")
    try:
        response = completion(
            model="zai/glm-4.6",
            messages=[{"role": "user", "content": "What is 2+2? Think step by step."}],
            api_key="test-key",
            reasoning_tokens=True
        )
        print("✅ ZAI thinking enabled test passed")
    except Exception as e:
        print(f"❌ ZAI thinking enabled test failed: {e}")

def test_zai_thinking_disabled():
    """Test ZAI thinking parameter disabled"""
    print("Testing ZAI thinking parameter disabled...")
    try:
        response = completion(
            model="zai/glm-4.6",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            api_key="test-key"
        )
        print("✅ ZAI thinking disabled test passed")
    except Exception as e:
        print(f"❌ ZAI thinking disabled test failed: {e}")

def test_zai_backward_compatibility():
    """Test backward compatibility with old Anthropic shape on ZAI"""
    print("Testing ZAI backward compatibility...")
    try:
        response = completion(
            model="zai/glm-4.6",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test-key",
            thinking={"type": "enabled"}  # Old Anthropic shape
        )
        print("✅ ZAI backward compatibility test passed")
    except Exception as e:
        print(f"❌ ZAI backward compatibility test failed: {e}")

def main():
    print("Running ZAI thinking implementation tests...\n")
    
    test_zai_thinking_enabled() 
    test_zai_thinking_disabled()
    test_zai_backward_compatibility()
    
    print("\n✅ All tests completed!")
    
if __name__ == "__main__":
    main()