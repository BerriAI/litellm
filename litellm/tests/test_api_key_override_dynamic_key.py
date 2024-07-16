from litellm import completion

model = "perplexity/llama-3-sonar-large-32k-online"
# model = "groq/mixtral-8x7b-32768"
# model = "together_ai/Qwen/Qwen1.5-32B-Chat"
# model = "fireworks_ai/llama-v3-8b-instruct"
try:
    response = completion(
        model=model,
        messages=[{"role": "user", "content": "hey, how's it going?"}],
        api_key="invalid_key"
    )
    
except Exception as e:
    print(e)

print("==============try original===============")
response = completion(
    model=model,
    messages=[{"role": "user", "content": "hey, how's it going?"}],
)

print(response)