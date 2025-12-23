"""
Test script for Google Imagen via JSON-configured SDK provider.

This demonstrates how to use litellm.image_generation() with JSON-configured
providers that support transformations and cost tracking.
"""

import os
import sys

# Add workspace to path for testing
sys.path.insert(0, "/workspace")


def test_configuration_loading():
    """Test that configuration loads correctly"""
    print("=" * 70)
    print("TEST 1: Configuration Loading")
    print("=" * 70)
    
    try:
        from litellm.llms.json_providers.sdk_provider_registry import SDKProviderRegistry
        
        # Reload to ensure we have latest
        SDKProviderRegistry.reload()
        
        # Get Google Imagen config
        config = SDKProviderRegistry.get("google_imagen")
        
        if not config:
            print("❌ FAILED: google_imagen configuration not found")
            return False
        
        print("✅ SUCCESS: Configuration loaded")
        print()
        print("Configuration Details:")
        print("-" * 70)
        print(f"Provider Name:      {config.provider_name}")
        print(f"Provider Type:      {config.provider_type}")
        print(f"API Base:           {config.api_base}")
        print(f"Auth Type:          {config.authentication.type}")
        print(f"Auth Env Var:       {config.authentication.env_var}")
        print(f"Transform Request:  {config.transformations['request'].type}")
        print(f"Transform Response: {config.transformations['response'].type}")
        print(f"Cost Tracking:      {config.cost_tracking.enabled}")
        print(f"Supported Models:   {len(config.endpoints['generate'].supported_models)}")
        print("-" * 70)
        print()
        
        # Show supported models
        print("Supported Models:")
        for model in config.endpoints['generate'].supported_models:
            cost = config.cost_tracking.cost_per_image.get(model, "N/A")
            print(f"  • {model} (${cost}/image)")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transformation_engine():
    """Test the transformation engine"""
    print("=" * 70)
    print("TEST 2: Transformation Engine")
    print("=" * 70)
    
    try:
        from litellm.llms.json_providers.transformation_engine import TransformationEngine
        from litellm.llms.json_providers.sdk_provider_registry import SDKProviderRegistry
        
        config = SDKProviderRegistry.get("google_imagen")
        if not config:
            print("❌ Configuration not loaded")
            return False
        
        # Test request transformation
        litellm_params = {
            "prompt": "A cute otter swimming",
            "n": 2,
            "aspect_ratio": "16:9",
            "negative_prompt": "blurry"
        }
        
        print("Input (LiteLLM format):")
        print(f"  {litellm_params}")
        print()
        
        transformed = TransformationEngine.transform_request(
            litellm_params,
            config.transformations["request"]
        )
        
        print("Output (Provider format):")
        print(f"  {transformed}")
        print()
        
        # Verify structure
        assert "instances" in transformed, "Missing 'instances' key"
        assert "parameters" in transformed, "Missing 'parameters' key"
        assert transformed["parameters"]["sampleCount"] == 2
        
        print("✅ SUCCESS: Request transformation works")
        print()
        
        # Test response transformation
        provider_response = {
            "predictions": [
                {"bytesBase64Encoded": "base64_image_data_1"},
                {"bytesBase64Encoded": "base64_image_data_2"}
            ]
        }
        
        print("Provider Response:")
        print(f"  {provider_response}")
        print()
        
        transformed_response = TransformationEngine.transform_response(
            provider_response,
            config.transformations["response"]
        )
        
        print("Transformed Response:")
        print(f"  {transformed_response}")
        print()
        
        assert "images" in transformed_response
        assert len(transformed_response["images"]) == 2
        
        print("✅ SUCCESS: Response transformation works")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cost_calculation():
    """Test cost tracking"""
    print("=" * 70)
    print("TEST 3: Cost Tracking")
    print("=" * 70)
    
    try:
        from litellm.llms.json_providers.cost_tracker import CostTracker
        from litellm.llms.json_providers.sdk_provider_registry import SDKProviderRegistry
        from openai.types.image import Image
        from litellm.types.utils import ImageResponse
        
        config = SDKProviderRegistry.get("google_imagen")
        if not config:
            print("❌ Configuration not loaded")
            return False
        
        # Create mock response
        images = [
            Image(b64_json="image1", url=None, revised_prompt=None),
            Image(b64_json="image2", url=None, revised_prompt=None),
        ]
        response = ImageResponse(created=0, data=images)
        
        # Test cost for fast model
        model = "imagen-3.0-fast-generate-001"
        cost = CostTracker.calculate_image_generation_cost(
            response, model, config.cost_tracking
        )
        
        expected_cost = 2 * 0.02  # 2 images × $0.02
        
        print(f"Model: {model}")
        print(f"Images: {len(images)}")
        print(f"Cost per image: $0.02")
        print(f"Calculated cost: ${cost:.4f}")
        print(f"Expected cost: ${expected_cost:.4f}")
        print()
        
        assert cost == expected_cost, f"Cost mismatch: {cost} != {expected_cost}"
        
        print("✅ SUCCESS: Cost tracking works")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sdk_usage_example():
    """Show how to use with litellm.image_generation()"""
    print("=" * 70)
    print("TEST 4: SDK Usage Example (Dry Run)")
    print("=" * 70)
    
    print("Example usage with litellm.image_generation():")
    print()
    print("```python")
    print("import litellm")
    print("import os")
    print()
    print("# Set API key")
    print('os.environ["GOOGLE_API_KEY"] = "your-api-key"')
    print()
    print("# Generate image")
    print("response = litellm.image_generation(")
    print('    prompt="A cute otter swimming",')
    print('    model="imagen-3.0-fast-generate-001",')
    print('    custom_llm_provider="google_imagen",')
    print("    n=2,")
    print('    aspect_ratio="16:9"')
    print(")")
    print()
    print("# Access results")
    print("for img in response.data:")
    print("    print(img.b64_json[:50])  # First 50 chars of base64")
    print()
    print('# Check cost')
    print('cost = response._hidden_params["response_cost"]')
    print('print(f"Cost: ${cost:.4f}")')
    print("```")
    print()
    
    if os.getenv("GOOGLE_API_KEY"):
        print("✅ GOOGLE_API_KEY is set - ready for real API calls")
    else:
        print("⚠️  GOOGLE_API_KEY not set - set it to test real API calls")
    
    print()
    return True


def main():
    """Run all tests"""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "SDK-Level JSON Provider Test Suite" + " " * 23 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    tests = [
        ("Configuration Loading", test_configuration_loading),
        ("Transformation Engine", test_transformation_engine),
        ("Cost Tracking", test_cost_calculation),
        ("SDK Usage Example", test_sdk_usage_example),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    print()
    
    if passed == total:
        print("╔" + "=" * 68 + "╗")
        print("║" + " " * 15 + "🎉 ALL TESTS PASSED! 🎉" + " " * 28 + "║")
        print("╚" + "=" * 68 + "╝")
        print()
        print("The SDK-level JSON provider system is working correctly!")
        print()
        print("Next steps:")
        print("  1. Set GOOGLE_API_KEY environment variable")
        print("  2. Install dependencies: pip install jinja2 jsonpath-ng")
        print("  3. Test with real API: python test_sdk_google_imagen_live.py")
        return 0
    else:
        print("❌ Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
