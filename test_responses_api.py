#!/usr/bin/env python3
"""
Comprehensive test for LiteLLM Responses API with multiple providers
Tests session management and context retention
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:4000"
HEADERS = {
    "Authorization": "Bearer sk-test-key-1234",
    "Content-Type": "application/json"
}

# Test configuration for different providers
PROVIDERS = [
    {
        "name": "Claude 3.5 Sonnet",
        "model": "claude-3-5-sonnet",
        "supports_context": True
    },
    {
        "name": "DeepSeek Chat",
        "model": "deepseek-chat",
        "supports_context": True
    },
    {
        "name": "Gemini 2.0 Flash",
        "model": "gemini-2.0-flash",
        "supports_context": True
    }
]

def test_basic_response(model_name, provider_name):
    """Test basic request/response functionality"""
    print(f"\n1. Testing basic response for {provider_name}...")
    
    response = requests.post(
        f"{BASE_URL}/v1/responses",
        headers=HEADERS,
        json={
            "model": model_name,
            "input": "Say 'Hello World' and nothing else.",
            "max_tokens": 20
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        response_text = result['output'][0]['content'][0]['text']
        print(f"   ✅ Basic response successful")
        print(f"   Response: {response_text[:100]}")
        return True
    else:
        print(f"   ❌ Failed: {response.status_code}")
        print(f"   Error: {response.text[:200]}")
        return False

def test_session_management(model_name, provider_name, supports_context=True):
    """Test session management with context retention"""
    print(f"\n2. Testing session management for {provider_name}...")
    
    # First message - establish context
    response1 = requests.post(
        f"{BASE_URL}/v1/responses",
        headers=HEADERS,
        json={
            "model": model_name,
            "input": "My name is Alice and I love Python programming. Remember this.",
            "max_tokens": 100
        }
    )
    
    if response1.status_code != 200:
        print(f"   ❌ First request failed: {response1.status_code}")
        return False
    
    result1 = response1.json()
    response_id = result1["id"]
    print(f"   ✅ First message sent, Response ID: {response_id[:50]}...")
    print(f"   Assistant: {result1['output'][0]['content'][0]['text'][:100]}...")
    
    # Small delay to ensure session is saved
    time.sleep(1)
    
    # Second message - test context retention
    response2 = requests.post(
        f"{BASE_URL}/v1/responses",
        headers=HEADERS,
        json={
            "model": model_name,
            "input": "What's my name and what programming language do I love?",
            "previous_response_id": response_id,
            "max_tokens": 100
        }
    )
    
    if response2.status_code != 200:
        print(f"   ❌ Second request failed: {response2.status_code}")
        return False
    
    result2 = response2.json()
    response_text = result2['output'][0]['content'][0]['text'].lower()
    print(f"   ✅ Second message sent with context")
    print(f"   Assistant: {result2['output'][0]['content'][0]['text'][:200]}...")
    
    # Check if context was maintained
    if supports_context:
        if "alice" in response_text and "python" in response_text:
            print(f"   ✅ Context successfully maintained!")
            return True
        else:
            print(f"   ⚠️  Context not maintained (expected for some models)")
            return False
    else:
        print(f"   ℹ️  Context retention not expected for this provider")
        return True

def test_streaming(model_name, provider_name):
    """Test streaming responses"""
    print(f"\n3. Testing streaming for {provider_name}...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/v1/responses",
            headers=HEADERS,
            json={
                "model": model_name,
                "input": "Count from 1 to 3",
                "max_tokens": 30,
                "stream": True
            },
            stream=True,
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"   ✅ Streaming response received")
            chunks_received = 0
            for line in response.iter_lines():
                if line:
                    chunks_received += 1
                    if chunks_received > 10:  # Limit output
                        break
            print(f"   Received {chunks_received} chunks")
            return True
        else:
            print(f"   ❌ Streaming failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Streaming error: {str(e)[:100]}")
        return False

def main():
    """Run all tests for each provider"""
    print("=" * 70)
    print("LITELLM RESPONSES API COMPREHENSIVE TEST")
    print("=" * 70)
    
    # Check if proxy is running
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=2)
        if health.status_code != 200:
            print("❌ Proxy server is not responding at http://localhost:4000")
            print("Please start the proxy with: litellm --config responses_api_config.yaml --port 4000")
            sys.exit(1)
    except:
        print("❌ Cannot connect to proxy server at http://localhost:4000")
        print("Please start the proxy with: litellm --config responses_api_config.yaml --port 4000")
        sys.exit(1)
    
    print("✅ Proxy server is running\n")
    
    results = {}
    
    for provider in PROVIDERS:
        print(f"\n{'='*70}")
        print(f"Testing {provider['name']}")
        print(f"{'='*70}")
        
        results[provider['name']] = {
            'basic': False,
            'session': False,
            'streaming': False
        }
        
        # Run tests
        results[provider['name']]['basic'] = test_basic_response(
            provider['model'], provider['name']
        )
        
        if results[provider['name']]['basic']:
            results[provider['name']]['session'] = test_session_management(
                provider['model'], provider['name'], provider['supports_context']
            )
            results[provider['name']]['streaming'] = test_streaming(
                provider['model'], provider['name']
            )
    
    # Print summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    
    for provider_name, test_results in results.items():
        print(f"\n{provider_name}:")
        print(f"  Basic Response:     {'✅' if test_results['basic'] else '❌'}")
        print(f"  Session Management: {'✅' if test_results['session'] else '❌'}")
        print(f"  Streaming:          {'✅' if test_results['streaming'] else '❌'}")
    
    # Overall result
    all_passed = all(
        test_results['basic'] and test_results['session'] 
        for test_results in results.values()
    )
    
    print(f"\n{'='*70}")
    if all_passed:
        print("✅ ALL CRITICAL TESTS PASSED!")
        print("The Responses API is working correctly with session management.")
    else:
        print("⚠️  Some tests failed. Check the details above.")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()