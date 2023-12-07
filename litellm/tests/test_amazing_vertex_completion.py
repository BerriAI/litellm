
import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
import os, io

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path  
import pytest
import litellm
from litellm import embedding, completion, completion_cost, Timeout
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
    vertex_key_path = 'vertex_key.json'

    # Read the existing content of the file or create an empty dictionary
    try:
        with open(vertex_key_path, 'r') as file:
            # Read the file content
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

    service_account_key_data["private_key_id"] = private_key_id
    service_account_key_data["private_key"] = private_key

    for key in service_account_key_data:
        if key not in ["private_key_id", "private_key"]:
            print(f"Key: {key}, Value: {service_account_key_data[key]}")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        # Write the updated content to the temporary file
        json.dump(service_account_key_data, temp_file, indent=2)
    

    # Export the temporary file as GOOGLE_APPLICATION_CREDENTIALS
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.abspath(temp_file.name)


def test_vertex_ai():

    load_vertex_ai_credentials()
    test_models = ["codechat-bison"] + litellm.vertex_chat_models + litellm.vertex_code_chat_models + litellm.vertex_text_models + litellm.vertex_code_text_models
    # test_models = ["chat-bison"]
    litellm.set_verbose=True
    for model in test_models:
        try:
            if model in ["code-gecko@001", "code-gecko@latest", "code-bison@001"]:
                # our account does not have access to this model
                continue
            print("making request", model)
            response = completion(model=model, messages=[{'role': 'user', 'content': 'hi'}])
            print(response)

            print(response.usage.completion_tokens)
            print(response['usage']['completion_tokens'])
            assert type(response.choices[0].message.content) == str
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")
test_vertex_ai()

def test_vertex_ai_stream():
    litellm.set_verbose=True

    test_models = litellm.vertex_chat_models + litellm.vertex_code_chat_models + litellm.vertex_text_models + litellm.vertex_code_text_models
    for model in test_models:
        try:
            if model in ["code-gecko@001", "code-gecko@latest", "code-bison@001"]:
                # our account does not have access to this model
                continue
            print("making request", model)
            response = completion(model=model, messages=[{"role": "user", "content": "write 100 line code code for saying hi"}], stream=True)
            for chunk in response:
                print(chunk)
                # pass
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")
test_vertex_ai_stream() 
