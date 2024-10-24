import openai

client = openai.OpenAI(
    api_key="sk-1234",  # litellm proxy api key
    base_url="http://0.0.0.0:4000",  # litellm proxy base url
)


response = client.chat.completions.create(
    model="anthropic/claude-3-5-sonnet-20240620",
    messages=[
        {  # type: ignore
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an AI assistant tasked with analyzing legal documents.",
                },
                {
                    "type": "text",
                    "text": "Here is the full text of a complex legal agreement" * 100,
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        },
        {
            "role": "user",
            "content": "what are the key terms and conditions in this agreement?",
        },
    ],
    extra_headers={
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "prompt-caching-2024-07-31",
    },
)

print(response)
