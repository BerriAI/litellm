import openai

api_base = "http://0.0.0.0:8000"

client = openai.OpenAI(base_url=api_base, api_key="temp-key")
print(client.base_url)


print("LiteLLM: response from proxy with streaming")
response = client.chat.completions.create(
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

response = client.chat.completions.create(
    model="ollama/llama2",
    messages=[
        {
            "role": "user",
            "content": "this is a test request, acknowledge that you got it",
        }
    ],
)

print(f"LiteLLM: response from proxy {response}")
