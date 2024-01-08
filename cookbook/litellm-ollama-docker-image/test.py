import openai

api_base = f"http://0.0.0.0:8000"

openai.api_base = api_base
openai.api_key = "temp-key"
print(openai.api_base)


print(f"LiteLLM: response from proxy with streaming")
response = openai.ChatCompletion.create(
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

response = openai.ChatCompletion.create(
    model="ollama/llama2",
    messages=[
        {
            "role": "user",
            "content": "this is a test request, acknowledge that you got it",
        }
    ],
)

print(f"LiteLLM: response from proxy {response}")
