from litellm import completion, stream_chunk_builder
import litellm
import os, dotenv
import pytest
dotenv.load_dotenv()

user_message = "What is the current weather in Boston?"
messages = [{"content": user_message, "role": "user"}]

function_schema = {
  "name": "get_weather",
  "description":
  "gets the current weather",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description":
        "The city and state, e.g. San Francisco, CA"
      },
    },
    "required": ["location"]
  },
}

@pytest.mark.skip
def test_stream_chunk_builder():
    litellm.set_verbose = False
    litellm.api_key = os.environ["OPENAI_API_KEY"]
    response = completion(
        model="gpt-3.5-turbo",
        messages=messages,
        functions=[function_schema],
        stream=True,
    )

    chunks = []

    for chunk in response:
        # print(chunk)
        chunks.append(chunk)

    try:
        print(f"chunks: {chunks}")
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
        finish_reason = choices["finish_reason"]
        print(role, content, finish_reason)
    except Exception as e:
        raise Exception("stream_chunk_builder failed to rebuild response", e)

