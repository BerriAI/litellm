
import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
import os, io

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path  
import pytest, asyncio
import litellm
from litellm import embedding, completion, completion_cost, Timeout, acompletion
from litellm import RateLimitError
import json
import os
import tempfile

litellm.num_retries = 3
litellm.cache = None
user_message = "Write a short poem about the sky"
messages = [{"content": user_message, "role": "user"}]


def load_vertex_ai_credentials():
    # Define the path to the vertex_key.json file
    print("loading vertex ai credentials")
    filepath = os.path.dirname(os.path.abspath(__file__))
    vertex_key_path = filepath + '/vertex_key.json'

    # Read the existing content of the file or create an empty dictionary
    try:
        with open(vertex_key_path, 'r') as file:
            # Read the file content
            print("Read vertexai file path")
            content = file.read()

            # If the file is empty or not valid JSON, create an empty dictionary
            if not content or not content.strip():
                service_account_key_data = {}
            else:
                # Attempt to load the existing JSON content
                file.seek(0)
                service_account_key_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty dictionary
        service_account_key_data = {}

    # Update the service_account_key_data with environment variables
    private_key_id = os.environ.get("VERTEX_AI_PRIVATE_KEY_ID", "")
    private_key = os.environ.get("VERTEX_AI_PRIVATE_KEY", "")
    private_key = private_key.replace("\\n", "\n")
    service_account_key_data["private_key_id"] = private_key_id
    service_account_key_data["private_key"] = private_key

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        # Write the updated content to the temporary file
        json.dump(service_account_key_data, temp_file, indent=2)
    

    # Export the temporary file as GOOGLE_APPLICATION_CREDENTIALS
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.abspath(temp_file.name)


def test_vertex_ai():
    import random

    load_vertex_ai_credentials()
    test_models = litellm.vertex_chat_models + litellm.vertex_code_chat_models + litellm.vertex_text_models + litellm.vertex_code_text_models
    litellm.set_verbose=False
    litellm.vertex_project = "hardy-device-386718"

    test_models = random.sample(test_models, 4)
    test_models += litellm.vertex_language_models # always test gemini-pro
    for model in test_models:
        try:
            if model in ["code-gecko@001", "code-gecko@latest", "code-bison@001", "text-bison@001"]:
                # our account does not have access to this model
                continue
            print("making request", model)
            response = completion(model=model, messages=[{'role': 'user', 'content': 'hi'}], temperature=0.7)
            print("\nModel Response", response)
            print(response)
            assert type(response.choices[0].message.content) == str
            assert len(response.choices[0].message.content) > 1
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")
# test_vertex_ai()

def test_vertex_ai_stream():
    load_vertex_ai_credentials()
    litellm.set_verbose=False
    litellm.vertex_project = "hardy-device-386718"
    import random

    test_models = litellm.vertex_chat_models + litellm.vertex_code_chat_models + litellm.vertex_text_models + litellm.vertex_code_text_models 
    test_models = random.sample(test_models, 4)
    test_models += litellm.vertex_language_models # always test gemini-pro
    for model in test_models:
        try:
            if model in ["code-gecko@001", "code-gecko@latest", "code-bison@001", "text-bison@001"]:
                # our account does not have access to this model
                continue
            print("making request", model)
            response = completion(model=model, messages=[{"role": "user", "content": "write 10 line code code for saying hi"}], stream=True)
            completed_str = ""
            for chunk in response:
                print(chunk)
                content = chunk.choices[0].delta.content or ""
                print("\n content", content)
                completed_str += content
                assert type(content) == str
                # pass
            assert len(completed_str) > 4
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")
# test_vertex_ai_stream() 

@pytest.mark.asyncio
async def test_async_vertexai_response():
    import random
    load_vertex_ai_credentials()
    test_models = litellm.vertex_chat_models + litellm.vertex_code_chat_models + litellm.vertex_text_models + litellm.vertex_code_text_models 
    test_models = random.sample(test_models, 4)
    test_models += litellm.vertex_language_models # always test gemini-pro
    for model in test_models:
        print(f'model being tested in async call: {model}')
        try:
            user_message = "Hello, how are you?"
            messages = [{"content": user_message, "role": "user"}]
            response = await acompletion(model=model, messages=messages, temperature=0.7, timeout=5)
            print(f"response: {response}")
        except litellm.Timeout as e: 
            pass
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")

# asyncio.run(test_async_vertexai_response())

@pytest.mark.asyncio
async def test_async_vertexai_streaming_response():
    import random
    load_vertex_ai_credentials()
    test_models = litellm.vertex_chat_models + litellm.vertex_code_chat_models + litellm.vertex_text_models + litellm.vertex_code_text_models 
    test_models = random.sample(test_models, 4)
    test_models += litellm.vertex_language_models # always test gemini-pro
    for model in test_models:
        try:
            user_message = "Hello, how are you?"
            messages = [{"content": user_message, "role": "user"}]
            response = await acompletion(model="gemini-pro", messages=messages, temperature=0.7, timeout=5, stream=True)
            print(f"response: {response}")
            complete_response = ""
            async for chunk in response:
                print(f"chunk: {chunk}")
                complete_response += chunk.choices[0].delta.content
            print(f"complete_response: {complete_response}")
            assert len(complete_response) > 0
        except litellm.Timeout as e: 
            pass
        except Exception as e:
            print(e)
            pytest.fail(f"An exception occurred: {e}")

# asyncio.run(test_async_vertexai_streaming_response())
