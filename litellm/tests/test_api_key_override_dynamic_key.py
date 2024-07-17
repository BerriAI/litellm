from litellm import completion

try:
    response = completion(
        model="groq/mixtral-8x7b-32768",
        messages=[{"role": "user", "content": "hey, how's it going?"}],
        api_key="invalid_key"
    )
    
except Exception as e:
    print(e)

print("==============try original===============")
response = completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "hey, how's it going?"}],
)

print(response)