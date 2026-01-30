import os

from anthropic import Anthropic

client = Anthropic(
    # This is the default and can be omitted
    base_url="http://localhost:4000",
    # this is a litellm proxy key :) - not a real anthropic key
    api_key="sk-test-proxy-key-123",
)

message = client.messages.create(
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Hello, Claude",
        }
    ],
    model="claude-3-opus-20240229",
)
print(message.content)
