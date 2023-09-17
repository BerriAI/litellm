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

    try:
        rebuilt_response = stream_chunk_builder(chunks)

        # exract the response from the rebuilt response
        rebuilt_response["id"]
        rebuilt_response["object"]
        rebuilt_response["created"]
        rebuilt_response["model"]
        rebuilt_response["choices"]
        rebuilt_response["choices"][0]["index"]
        choices = rebuilt_response["choices"][0]
        message = choices["message"]
        role = message["role"]
        content = message["content"]
        finnish_reason = choices["finish_reason"]
    except:
        raise Exception("stream_chunk_builder failed to rebuild response")
test_stream_chunk_builder()
