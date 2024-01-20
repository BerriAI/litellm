import openai
from litellm import completion

api_base = "http://0.0.0.0:8000"

openai.base_url = api_base
openai.api_key = "temp-key"
print(openai.base_url)


print("LiteLLM: response from proxy with streaming")
response = completion(
    model="ollama/llama2",
    messages=[
        {
            "role": "user",
            "content": "this is a test request, acknowledge that you got it",
        }
    ],
    stream=True,
)

for chunk in response:
    print(f"LiteLLM: streaming response from proxy {chunk}")

response = completion(
    model="ollama/llama2",
    messages=[
        {
            "role": "user",
            "content": "this is a test request, acknowledge that you got it",
        }
    ],
)

print(f"LiteLLM: response from proxy {response}")
