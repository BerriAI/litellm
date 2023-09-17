from litellm import completion, stream_chunk_builder
import litellm
import os

user_message = "Write a short poem about the sky"
messages = [{"content": user_message, "role": "user"}]

def test_stream_chunk_builder():
    litellm.api_key = os.environ["OPENAI_API_KEY"]
    response = completion(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
        max_tokens=10,
    )

    chunks = []

    for chunk in response:
        chunks.append(chunk)

    print(chunks)
test_stream_chunk_builder()
