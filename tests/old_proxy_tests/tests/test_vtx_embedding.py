import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

# # request sent to model set on litellm proxy, `litellm --model`
response = client.embeddings.create(
    model="multimodalembedding@001",
    input=[],
    extra_body={
        "instances": [
            {
                "image": {
                    "gcsUri": "gs://cloud-samples-data/vertex-ai/llm/prompts/landmark1.png"
                },
                "text": "this is a unicorn",
            },
        ],
    },
)

print(response)
