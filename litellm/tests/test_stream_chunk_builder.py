import sys, os, time
import traceback, asyncio
import pytest
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
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

def test_stream_chunk_builder():
    try: 
      litellm.set_verbose = False
      response = completion(
          model="gpt-3.5-turbo",
          messages=messages,
          functions=[function_schema],
          stream=True,
          complete_response=True # runs stream_chunk_builder under-the-hood
      )

      print(f"response: {response}")
      print(f"response usage: {response['usage']}")
    except Exception as e: 
       pytest.fail(f"An exception occurred - {str(e)}")

test_stream_chunk_builder()