import litellm

model = "gpt-3.5-turbo"
messages = [
    {
        "role": "user",
        "content": "hey, how's it going?"
    }
]
repsonse = litellm.completion(model=model, messages=messages)
print(repsonse)
