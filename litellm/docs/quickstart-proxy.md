LiteLLM Proxy Quickstart Guide (Updated)

This guide clarifies common issues and environment requirements for running LiteLLM locally.
Step 1: Start the Backend

Choose either a Hugging Face or OpenAI backend.

Hugging Face backend (example):
litellm --model huggingface/bigcode/starcoder Note:Hugging Face models are not always chat-capable. Using them with Step 2 may result in 400 Bad Request.

OpenAI backend (recommended for chat/completions requests):
litellm --model gpt-3.5-turbo

Step 2: Connect Python Script
Use the OpenAI client to connect to the LiteLLM proxy:

import openai

client = openai.OpenAI(
api_key="YOUR_OPENAI_API_KEY", # or set OPENAI_API_KEY in environment
base_url="http://localhost:4000
"
)

response = client.chat.completions.create(
model="gpt-3.5-turbo",
messages=[{"role": "user", "content": "Write a short poem"}]
)

print(response.choices[0].message.content)

Notes:

If api_key is not passed, you must set:
export OPENAI_API_KEY="sk-xxxx"

Only use chat-capable models; otherwise you’ll get 400 Bad Request.

Common Issues:
Error: 401 Unauthorized
Cause: Using Hugging Face backend with Step 2
Solution: Use OpenAI backend or adjust model/script

Error: 400 Bad Request
Cause: Requested model is not chat-enabled
Solution: Use a chat-capable model like gpt-3.5-turbo

Error: 500 Internal Server Error
Cause: Missing OPENAI_API_KEY
Solution: Set OPENAI_API_KEY in environment or pass api_key in client

Summary:
Step 1: Choose backend carefully (Hugging Face vs OpenAI)
Step 2: Ensure OPENAI_API_KEY is set in environment or client
Use chat-capable models for chat/completions requests
Consult the common issues table if you encounter errors