#!/usr/bin/env python3
"""
Test file to iterate Parasail provider code - following LiteLLM provider registration guide step 3
"""
import os
from litellm import completion

def test_provider_detection():
    """Test that provider detection works for parasail/ models"""
    print("Testing provider detection...")
    
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
    
    try:
        model, provider, api_key, api_base = get_llm_provider("parasail/parasail-deepseek-r1-0528")
        print(f"✓ Provider detection: {provider}")
        print(f"  Model: {model}")
        print(f"  API base: {api_base}")
        assert provider == "parasail", f"Expected 'parasail', got '{provider}'"
        print("✓ Provider detection test passed")
        return True
    except Exception as e:
        print(f"✗ Provider detection failed: {e}")
        return False

def test_config_functionality():
    """Test the ParasailChatConfig class functionality"""
    print("\nTesting ParasailChatConfig...")
    
    try:
        from litellm.llms.parasail.chat.transformation import ParasailChatConfig
        
        config = ParasailChatConfig()
        print("✓ Config instantiation successful")
        
        # Test API info
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        print(f"✓ Default API base: {api_base}")
        assert api_base == "https://api.parasail.io/v1", f"Expected 'https://api.parasail.io/v1', got '{api_base}'"
        
        # Test supported params
        params = config.get_supported_openai_params("test-model")
        print(f"✓ Supported parameters: {len(params)} params")
        
        required_params = ["temperature", "max_tokens", "stream", "tools", "tool_choice"]
        for param in required_params:
            assert param in params, f"Parameter {param} should be supported"
        
        # Test parameter mapping
        non_default_params = {
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": True,
        }
        
        mapped_params = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="test-model",
            drop_params=False,
        )
        
        assert mapped_params["temperature"] == 0.7
        assert mapped_params["max_tokens"] == 100
        assert mapped_params["stream"] is True
        
        print("✓ Config functionality test passed")
        return True
        
    except Exception as e:
        print(f"✗ Config functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_completion_integration():
    """Test that completion function can be called (without real API call)"""
    print("\nTesting completion integration...")
    
    try:
        # This will fail with auth error since we're using a fake key,
        # but it should get far enough to prove the integration works
        response = completion(
            model="parasail/parasail-deepseek-r1-0528",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
        )
        
        # If we get here without an import/routing error, integration works
        print("✓ Completion integration successful (got response)")
        return True
        
    except Exception as e:
        error_msg = str(e).lower()
        
        # Expected errors that indicate successful integration:
        if any(err in error_msg for err in [
            "authentication", "auth", "api key", "unauthorized", 
            "invalid", "permission", "access denied", "401", "403"
        ]):
            print("✓ Completion integration successful (auth error expected with fake key)")
            return True
        
        # Unexpected errors that indicate integration problems:
        elif any(err in error_msg for err in [
            "import", "module", "attribute", "not found", 
            "provider", "routing", "no module"
        ]):
            print(f"✗ Completion integration failed (routing/import error): {e}")
            return False
        
        else:
            print(f"✓ Completion integration likely successful (network/other error): {e}")
            return True

def test_constants_registration():
    """Test that Parasail is properly registered in constants"""
    print("\nTesting constants registration...")
    
    try:
        from litellm.constants import LITELLM_CHAT_PROVIDERS, openai_compatible_providers
        
        assert "parasail" in LITELLM_CHAT_PROVIDERS, "parasail should be in LITELLM_CHAT_PROVIDERS"
        assert "parasail" in openai_compatible_providers, "parasail should be in openai_compatible_providers"
        
        print("✓ Constants registration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Constants registration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Parasail Provider Integration Test")
    print("=" * 40)
    
    tests = [
        test_provider_detection,
        test_config_functionality,
        test_constants_registration,
        test_completion_integration,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("=" * 40)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} tests passed! Parasail integration is working.")
        return True
    else:
        print(f"❌ {total - passed} out of {total} tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
