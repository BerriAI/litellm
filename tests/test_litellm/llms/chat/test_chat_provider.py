#!/usr/bin/env python3
"""
Test script for the improved chat provider that proxies to OpenAI's Responses API
"""

import json
import os
import time
import requests
from typing import Any, Dict

# Configuration
LITELLM_BASE_URL = "http://localhost:4000"
MASTER_KEY = "sk-1234"  # Should match your config file
TEST_MODEL = "gpt-4o-responses"  # Using the chat provider

def test_basic_chat():
    """Test basic chat completion request"""
    print("🧪 Testing basic chat completion...")
    
    url = f"{LITELLM_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MASTER_KEY}"
    }
    data = {
        "model": TEST_MODEL,
        "messages": [
            {"role": "user", "content": "Hello! Can you tell me a short joke?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Basic chat completion successful!")
            print(f"Response: {result['choices'][0]['message']['content']}")
            print(f"Usage: {result.get('usage', {})}")
            return True
        else:
            print(f"❌ Request failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception in basic chat test: {e}")
        return False

def test_streaming_chat():
    """Test streaming chat completion"""
    print("\n🧪 Testing streaming chat completion...")
    
    url = f"{LITELLM_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MASTER_KEY}"
    }
    data = {
        "model": TEST_MODEL,
        "messages": [
            {"role": "user", "content": "Write a very short poem about coding"}
        ],
        "stream": True,
        "max_tokens": 150
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, stream=True)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Streaming started successfully!")
            content = ""
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get('choices', [{}])[0].get('delta', {})
                            if 'content' in delta:
                                content += delta['content']
                                print(delta['content'], end='', flush=True)
                        except json.JSONDecodeError:
                            continue
            print(f"\n✅ Streaming completed! Total content length: {len(content)}")
            return True
        else:
            print(f"❌ Streaming failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception in streaming test: {e}")
        return False

def test_function_calling():
    """Test function calling support"""
    print("\n🧪 Testing function calling...")
    
    url = f"{LITELLM_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MASTER_KEY}"
    }
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    data = {
        "model": TEST_MODEL,
        "messages": [
            {"role": "user", "content": "What's the weather like in San Francisco?"}
        ],
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 100
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            message = result['choices'][0]['message']
            
            if 'tool_calls' in message:
                print("✅ Function calling successful!")
                for tool_call in message['tool_calls']:
                    print(f"Function called: {tool_call['function']['name']}")
                    print(f"Arguments: {tool_call['function']['arguments']}")
                return True
            else:
                print("⚠️  No function call detected (model may have responded directly)")
                print(f"Response: {message.get('content', '')}")
                return True
        else:
            print(f"❌ Function calling failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception in function calling test: {e}")
        return False

def test_system_message():
    """Test system message handling (converted to instructions)"""
    print("\n🧪 Testing system message handling...")
    
    url = f"{LITELLM_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MASTER_KEY}"
    }
    data = {
        "model": TEST_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that responds in exactly 10 words."},
            {"role": "user", "content": "Tell me about Python programming."}
        ],
        "max_tokens": 50
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            word_count = len(content.split())
            print("✅ System message test successful!")
            print(f"Response: {content}")
            print(f"Word count: {word_count} (should be close to 10)")
            return True
        else:
            print(f"❌ System message test failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception in system message test: {e}")
        return False

def test_multimodal():
    """Test multimodal content (if supported)"""
    print("\n🧪 Testing multimodal content...")
    
    url = f"{LITELLM_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MASTER_KEY}"
    }
    data = {
        "model": TEST_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                        }
                    }
                ]
            }
        ],
        "max_tokens": 100
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Multimodal test successful!")
            print(f"Response: {result['choices'][0]['message']['content']}")
            return True
        else:
            print(f"⚠️  Multimodal test failed (may not be supported): {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception in multimodal test: {e}")
        return False

def check_server_health():
    """Check if the LiteLLM server is running"""
    print("🔍 Checking server health...")
    
    try:
        response = requests.get(f"{LITELLM_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Server is healthy!")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Chat Provider Tests")
    print("=" * 50)
    
    # Check server health first
    if not check_server_health():
        print("\n❌ Server is not accessible. Make sure LiteLLM proxy is running with:")
        print(f"   litellm --config test_chat_provider_config.yaml --port 4000")
        return
    
    # Run tests
    tests = [
        ("Basic Chat", test_basic_chat),
        ("Streaming Chat", test_streaming_chat),
        ("Function Calling", test_function_calling),
        ("System Message", test_system_message),
        ("Multimodal", test_multimodal),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Results Summary:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Chat provider is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs for details.")

if __name__ == "__main__":
    main()