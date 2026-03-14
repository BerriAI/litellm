"""
Simple regression test: call Perplexity through LiteLLM
to verify chat completions and responses API both work.
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

import litellm

# Show which branch we're on
branch = os.popen("git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown").read().strip()
print(f"=== Branch: {branch} ===\n")

# 1. Chat completions
print("--- Test 1: Chat Completions ---")
try:
    resp = litellm.completion(
        model="perplexity/sonar",
        messages=[{"role": "user", "content": "Say hello in 3 words"}],
        max_tokens=20,
    )
    print(f"OK: {resp.choices[0].message.content[:80]}")
    print(f"  model: {resp.model}")
    print(f"  usage: {resp.usage}")
except Exception as e:
    print(f"FAIL: {e}")

# 2. Responses API (string input)
print("\n--- Test 2: Responses API (string input) ---")
try:
    resp = litellm.responses(
        model="perplexity/sonar",
        input="Say hello in 3 words",
        max_output_tokens=20,
    )
    print(f"OK: {resp.output[0].content[0].text[:80]}")
    print(f"  model: {resp.model}")
except Exception as e:
    print(f"FAIL: {e}")

# 3. Responses API (list input - the _format_input concern)
print("\n--- Test 3: Responses API (list input without type field) ---")
try:
    resp = litellm.responses(
        model="perplexity/sonar",
        input=[{"role": "user", "content": "Say hello in 3 words"}],
        max_output_tokens=20,
    )
    print(f"OK: {resp.output[0].content[0].text[:80]}")
except Exception as e:
    print(f"FAIL: {e}")

# 4. Check which config class is resolved for chat
print("\n--- Test 4: Config class resolution ---")
from litellm.utils import ProviderConfigManager
from litellm.types.utils import LlmProviders

chat_config = ProviderConfigManager.get_provider_chat_config(
    model="perplexity/sonar", provider=LlmProviders.PERPLEXITY
)
print(f"Chat config class: {type(chat_config).__name__}")
print(f"  module: {type(chat_config).__module__}")

resp_config = ProviderConfigManager.get_provider_responses_api_config(
    provider=LlmProviders.PERPLEXITY
)
print(f"Responses config class: {type(resp_config).__name__}")
print(f"  module: {type(resp_config).__module__}")

# 5. Check supported params include preset/models for responses
print("\n--- Test 5: Supported params ---")
if resp_config:
    params = resp_config.get_supported_openai_params("sonar")
    print(f"Responses supported params: {params}")
    has_preset = "preset" in params
    has_models = "models" in params
    print(f"  Has 'preset': {has_preset}")
    print(f"  Has 'models': {has_models}")

print("\n=== Done ===")
