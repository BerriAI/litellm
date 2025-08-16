#!/usr/bin/env python3
import os
import litellm

# Test regular Ollama with localhost
print("Testing regular Ollama with localhost...")
try:
    response = litellm.completion(
        model="ollama/llama2",
        messages=[{"role": "user", "content": "Say hello"}],
        api_base="http://localhost:11434",
        api_key="test-key"  # Should add "Bearer " prefix
    )
    print("✓ Regular Ollama call succeeded")
    print(f"Response: {response.choices[0].message.content[:50]}...")
except Exception as e:
    print(f"✗ Regular Ollama call failed: {e}")

# Test with custom Ollama endpoint (not ollama.com)
print("\nTesting with custom Ollama endpoint...")
try:
    response = litellm.completion(
        model="ollama/llama2",
        messages=[{"role": "user", "content": "Say hello"}],
        api_base="https://my-ollama-server.com",
        api_key="custom-key"  # Should add "Bearer " prefix
    )
    print("✓ Custom Ollama endpoint call would succeed (if server existed)")
except Exception as e:
    # Expected to fail since server doesn't exist, but we can check the error
    if "Bearer custom-key" in str(e):
        print("✓ Authorization header correctly includes 'Bearer' prefix")
    else:
        print(f"✗ Authorization header might be incorrect: {e}")