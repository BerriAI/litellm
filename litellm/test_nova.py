import sys
import os


import litellm

litellm.set_verbose = True

response = litellm.completion(
    model = "bedrock/us.amazon.nova-pro-v1:0",
    messages= [
        {
            "role": "user",
            "content": "What is the capital of France? whats the current weather there? "
        }
    ],
    tools = [
        {
           "type": "system_tool",
            "system_tool": {"name": "nova_grounding"}
        }
    ]
)

print(response)