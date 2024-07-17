from litellm import token_counter

messages = [
    {
        "role": "user",
        "content": "hey, how's it going?",
        "tool_calls": [
            {
                "tool": "token_counter",
                "parameters": "\{count\:5\}"
            }
        ]
    }
]

result = token_counter(
    messages=messages,
)

print(result)