#!/usr/bin/env python3
import litellm

# Test with local Ollama using gpt-oss:latest
print("Testing local Ollama with gpt-oss:latest...")
try:
    response = litellm.completion(
        model="ollama/gpt-oss:latest",
        messages=[{"role": "user", "content": "Say hello in one word"}],
        api_base="http://localhost:11434",
        api_key="dummy-key"  # Should add "Bearer " prefix for local Ollama
    )
    print("✓ Local Ollama call succeeded")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"✗ Local Ollama call failed: {e}")

# Test streaming with local Ollama
print("\nTesting streaming with local Ollama...")
try:
    response = litellm.completion(
        model="ollama/gpt-oss:latest",
        messages=[{"role": "user", "content": "Count to 3"}],
        api_base="http://localhost:11434",
        api_key="dummy-key",
        stream=True
    )
    print("✓ Streaming response: ", end="")
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
    print()
except Exception as e:
    print(f"✗ Streaming call failed: {e}")

# Test embeddings with local Ollama (using gemma3 which supports embeddings)
print("\nTesting embeddings with local Ollama...")
try:
    response = litellm.embedding(
        model="ollama/gemma3:12b",
        input=["Hello world"],
        api_base="http://localhost:11434",
        api_key="dummy-key"
    )
    print("✓ Embeddings call succeeded")
    print(f"Embedding dimensions: {len(response.data[0].embedding) if response.data else 'No data'}")
except Exception as e:
    print(f"✗ Embeddings call failed: {e}")