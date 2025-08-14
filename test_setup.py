#!/usr/bin/env python3
"""
Quick test to verify your proxy setup before running the full locust test
"""

import requests
import json

def test_proxy_setup():
    """Test basic proxy functionality"""
    base_url = "http://localhost:4000"
    headers = {'Authorization': 'Bearer sk-1234'}
    
    print("🔍 Testing LiteLLM proxy setup...")
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health", headers=headers, timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to proxy: {e}")
        return False
    
    # Test 2: Memory endpoint  
    try:
        response = requests.get(f"{base_url}/memory-usage", headers=headers, timeout=5)
        if response.status_code == 200:
            memory_data = response.json()
            print(f"✅ Memory endpoint working: {memory_data}")
        else:
            print(f"⚠️  Memory endpoint status: {response.status_code}")
    except Exception as e:
        print(f"⚠️  Memory endpoint error: {e}")
    
    # Test 3: Chat completions with your model
    try:
        payload = {
            "model": "openai/my-fake-model",
            "messages": [{"role": "user", "content": "Hello test"}],
            "max_tokens": 10
        }
        
        response = requests.post(f"{base_url}/chat/completions", 
                               headers=headers, 
                               json=payload, 
                               timeout=10)
        
        if response.status_code == 200:
            print("✅ Chat completions working with openai/my-fake-model")
            response_data = response.json()
            print(f"   Response preview: {json.dumps(response_data, indent=2)[:200]}...")
        else:
            print(f"❌ Chat completions failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Chat completions error: {e}")
        return False
    
    print("\n✅ Proxy setup looks good! Ready for data leak testing.")
    return True

if __name__ == "__main__":
    if test_proxy_setup():
        print("\n🚀 Run the data leak test with:")
        print("   locust -f test_data_leak_simple.py --host=http://localhost:4000")
        print("\n📊 Or for web UI:")
        print("   locust -f test_data_leak_simple.py --host=http://localhost:4000 --web-host=127.0.0.1")
        print("\n🎯 Recommended test parameters:")
        print("   - Users: 10-20")
        print("   - Spawn rate: 2-5 users/second") 
        print("   - Duration: 2-5 minutes")
    else:
        print("\n❌ Fix proxy setup before running data leak test")