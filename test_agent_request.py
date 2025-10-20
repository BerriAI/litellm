#!/usr/bin/env python3
"""
Test script to make a request to the Bedrock agent with enableTrace: true
and observe the logs to confirm the parameter is working.
"""

import requests
import json
import os

def test_bedrock_agent_tracing():
    """Test the lagasafn model which has enableTrace: true configured"""
    
    # Get the master key from environment
    master_key = os.environ.get('LITELLM_MASTER_KEY', 'sk-123456789')  # Default from logs
    
    # LiteLLM proxy endpoint (assuming it's running on localhost:4000)
    url = "http://localhost:4000/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {master_key}"
    }
    
    # Test with the lagasafn model that has enableTrace: true
    data = {
        "model": "lagasafn",
        "messages": [
            {"role": "user", "content": "Test request to verify tracing is enabled"}
        ]
    }
    
    print("Making request to lagasafn model (enableTrace: true)...")
    print(f"Request data: {json.dumps(data, indent=2)}")
    print()
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print()
        print("Response body:")
        print(json.dumps(response.json(), indent=2))
        
    except requests.exceptions.Timeout:
        print("Request timed out - this might indicate the agent is processing")
    except requests.exceptions.ConnectionError:
        print("Connection error - is the LiteLLM proxy running?")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_bedrock_agent_tracing()