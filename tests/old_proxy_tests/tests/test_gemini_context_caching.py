import datetime

import httpx
import openai

# Set Litellm proxy variables here
LITELLM_BASE_URL = "http://0.0.0.0:4000"
LITELLM_PROXY_API_KEY = "sk-1234"

client = openai.OpenAI(api_key=LITELLM_PROXY_API_KEY, base_url=LITELLM_BASE_URL)
httpx_client = httpx.Client(timeout=30)

################################
# First create a cachedContents object
print("creating cached content")
create_cache = httpx_client.post(
    url=f"{LITELLM_BASE_URL}/vertex-ai/cachedContents",
    headers={"Authorization": f"Bearer {LITELLM_PROXY_API_KEY}"},
    json={
        "model": "gemini-1.5-pro-001",
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": "This is sample text to demonstrate explicit caching."
                        * 4000
                    }
                ],
            }
        ],
    },
)
print("response from create_cache", create_cache)
create_cache_response = create_cache.json()
print("json from create_cache", create_cache_response)
cached_content_name = create_cache_response["name"]

#################################
# Use the `cachedContents` object in your /chat/completions
response = client.chat.completions.create(  # type: ignore
    model="gemini-1.5-pro-001",
    max_tokens=8192,
    messages=[
        {
            "role": "user",
            "content": "what is the sample text about?",
        },
    ],
    temperature="0.7",
    extra_body={"cached_content": cached_content_name},  # 👈 key change
)

print("response from proxy", response)
