import openai

client = openai.OpenAI(api_key="hi", base_url="http://0.0.0.0:8000")

# # request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(
    model="azure/chatgpt-v-2",
    messages=[
        {"role": "user", "content": "this is a test request, write a short poem"}
    ],
    extra_body={
        "metadata": {
            "generation_name": "ishaan-generation-openai-client",
            "generation_id": "openai-client-gen-id22",
            "trace_id": "openai-client-trace-id22",
            "trace_user_id": "openai-client-user-id2",
        }
    },
)

print(response)


# request sent to gpt-4-vision + enhancements

completion_extensions = client.chat.completions.create(
    model="gpt-vision",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image? Output your answer in JSON.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://avatars.githubusercontent.com/u/29436595?v=4",
                        "detail": "low",
                    },
                },
            ],
        }
    ],
    max_tokens=4096,
    temperature=0.0,
    extra_body={
        "enhancements": {"ocr": {"enabled": True}, "grounding": {"enabled": True}},
        "dataSources": [
            {
                "type": "AzureComputerVision",
                "parameters": {
                    "endpoint": "https://gpt-4-vision-enhancement.cognitiveservices.azure.com/",
                    "key": "f015cf8eeb1d4bd1b1467d21dec6063b",
                },
            }
        ],
    },
)
