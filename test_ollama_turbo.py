#!/usr/bin/env python3
import os
import litellm


# Set your Ollama Turbo API key
api_key = os.getenv("OLLAMA_API_KEY", "1015f5c1ee374b78b3475a55cb6c48df.D5V7jPuQf3Ek2R9C8_SrZBwg")

# Make a simple call to Ollama Turbo
response = litellm.completion(
    model="ollama/gpt-oss:120b",
    messages=[{"role": "user", "content": "Say hello"}],
    api_base="https://ollama.com",
    api_key=api_key
)

print(response.choices[0].message.content)