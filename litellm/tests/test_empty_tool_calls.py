from litellm import token_counter

messages = [
    {
        "role": "user",
        "content": "hey, how's it going?",
        "tool_calls": None
    }
]

result = token_counter(
    messages=messages,
)

print(result)