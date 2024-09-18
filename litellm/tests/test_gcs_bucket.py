import io
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import json
import logging
import tempfile
import uuid
from datetime import datetime

import pytest

import litellm
from litellm import completion
from litellm._logging import verbose_logger
from litellm.integrations.gcs_bucket import GCSBucketLogger, StandardLoggingPayload

verbose_logger.setLevel(logging.DEBUG)


def load_vertex_ai_credentials():
    # Define the path to the vertex_key.json file
    print("loading vertex ai credentials")
    filepath = os.path.dirname(os.path.abspath(__file__))
    vertex_key_path = filepath + "/adroit-crow-413218-bc47f303efc9.json"

    # Read the existing content of the file or create an empty dictionary
    try:
        with open(vertex_key_path, "r") as file:
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
    private_key_id = os.environ.get("GCS_PRIVATE_KEY_ID", "")
    private_key = os.environ.get("GCS_PRIVATE_KEY", "")
    private_key = private_key.replace("\\n", "\n")
    service_account_key_data["private_key_id"] = private_key_id
    service_account_key_data["private_key"] = private_key

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        # Write the updated content to the temporary files
        json.dump(service_account_key_data, temp_file, indent=2)

    # Export the temporary file as GOOGLE_APPLICATION_CREDENTIALS
    os.environ["GCS_PATH_SERVICE_ACCOUNT"] = os.path.abspath(temp_file.name)
    print("created gcs path service account=", os.environ["GCS_PATH_SERVICE_ACCOUNT"])


@pytest.mark.asyncio
async def test_basic_gcs_logger():
    load_vertex_ai_credentials()
    gcs_logger = GCSBucketLogger()
    print("GCSBucketLogger", gcs_logger)

    litellm.callbacks = [gcs_logger]
    response = await litellm.acompletion(
        model="gpt-3.5-turbo",
        temperature=0.7,
        messages=[{"role": "user", "content": "This is a test"}],
        max_tokens=10,
        user="ishaan-2",
        mock_response="Hi!",
        metadata={
            "tags": ["model-anthropic-claude-v2.1", "app-ishaan-prod"],
            "user_api_key": "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",
            "user_api_key_alias": None,
            "user_api_end_user_max_budget": None,
            "litellm_api_version": "0.0.0",
            "global_max_parallel_requests": None,
            "user_api_key_user_id": "116544810872468347480",
            "user_api_key_org_id": None,
            "user_api_key_team_id": None,
            "user_api_key_team_alias": None,
            "user_api_key_metadata": {},
            "requester_ip_address": "127.0.0.1",
            "requester_metadata": {"foo": "bar"},
            "spend_logs_metadata": {"hello": "world"},
            "headers": {
                "content-type": "application/json",
                "user-agent": "PostmanRuntime/7.32.3",
                "accept": "*/*",
                "postman-token": "92300061-eeaa-423b-a420-0b44896ecdc4",
                "host": "localhost:4000",
                "accept-encoding": "gzip, deflate, br",
                "connection": "keep-alive",
                "content-length": "163",
            },
            "endpoint": "http://localhost:4000/chat/completions",
            "model_group": "gpt-3.5-turbo",
            "deployment": "azure/chatgpt-v-2",
            "model_info": {
                "id": "4bad40a1eb6bebd1682800f16f44b9f06c52a6703444c99c7f9f32e9de3693b4",
                "db_model": False,
            },
            "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com/",
            "caching_groups": None,
            "raw_request": "\n\nPOST Request Sent from LiteLLM:\ncurl -X POST \\\nhttps://openai-gpt-4-test-v-1.openai.azure.com//openai/ \\\n-H 'Authorization: *****' \\\n-d '{'model': 'chatgpt-v-2', 'messages': [{'role': 'system', 'content': 'you are a helpful assistant.\\n'}, {'role': 'user', 'content': 'bom dia'}], 'stream': False, 'max_tokens': 10, 'user': '116544810872468347480', 'extra_body': {}}'\n",
        },
    )

    print("response", response)

    await asyncio.sleep(5)

    # Get the current date
    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Modify the object_name to include the date-based folder
    object_name = f"{current_date}%2F{response.id}"

    print("object_name", object_name)

    # Check if object landed on GCS
    object_from_gcs = await gcs_logger.download_gcs_object(object_name=object_name)
    print("object from gcs=", object_from_gcs)
    # convert object_from_gcs from bytes to DICT
    parsed_data = json.loads(object_from_gcs)
    print("object_from_gcs as dict", parsed_data)

    print("type of object_from_gcs", type(parsed_data))

    gcs_payload = StandardLoggingPayload(**parsed_data)

    print("gcs_payload", gcs_payload)

    assert gcs_payload["model"] == "gpt-3.5-turbo"
    assert gcs_payload["messages"] == [{"role": "user", "content": "This is a test"}]

    assert gcs_payload["response"]["choices"][0]["message"]["content"] == "Hi!"

    assert gcs_payload["response_cost"] > 0.0

    assert gcs_payload["status"] == "success"

    assert (
        gcs_payload["metadata"]["user_api_key_hash"]
        == "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b"
    )
    assert gcs_payload["metadata"]["user_api_key_user_id"] == "116544810872468347480"

    assert gcs_payload["metadata"]["requester_metadata"] == {"foo": "bar"}

    # Delete Object from GCS
    print("deleting object from GCS")
    await gcs_logger.delete_gcs_object(object_name=object_name)


@pytest.mark.asyncio
async def test_basic_gcs_logger_failure():
    load_vertex_ai_credentials()
    gcs_logger = GCSBucketLogger()
    print("GCSBucketLogger", gcs_logger)

    gcs_log_id = f"failure-test-{uuid.uuid4().hex}"

    litellm.callbacks = [gcs_logger]

    try:
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            temperature=0.7,
            messages=[{"role": "user", "content": "This is a test"}],
            max_tokens=10,
            user="ishaan-2",
            mock_response=litellm.BadRequestError(
                model="gpt-3.5-turbo",
                message="Error: 400: Bad Request: Invalid API key, please check your API key and try again.",
                llm_provider="openai",
            ),
            metadata={
                "gcs_log_id": gcs_log_id,
                "tags": ["model-anthropic-claude-v2.1", "app-ishaan-prod"],
                "user_api_key": "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",
                "user_api_key_alias": None,
                "user_api_end_user_max_budget": None,
                "litellm_api_version": "0.0.0",
                "global_max_parallel_requests": None,
                "user_api_key_user_id": "116544810872468347480",
                "user_api_key_org_id": None,
                "user_api_key_team_id": None,
                "user_api_key_team_alias": None,
                "user_api_key_metadata": {},
                "requester_ip_address": "127.0.0.1",
                "spend_logs_metadata": {"hello": "world"},
                "headers": {
                    "content-type": "application/json",
                    "user-agent": "PostmanRuntime/7.32.3",
                    "accept": "*/*",
                    "postman-token": "92300061-eeaa-423b-a420-0b44896ecdc4",
                    "host": "localhost:4000",
                    "accept-encoding": "gzip, deflate, br",
                    "connection": "keep-alive",
                    "content-length": "163",
                },
                "endpoint": "http://localhost:4000/chat/completions",
                "model_group": "gpt-3.5-turbo",
                "deployment": "azure/chatgpt-v-2",
                "model_info": {
                    "id": "4bad40a1eb6bebd1682800f16f44b9f06c52a6703444c99c7f9f32e9de3693b4",
                    "db_model": False,
                },
                "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com/",
                "caching_groups": None,
                "raw_request": "\n\nPOST Request Sent from LiteLLM:\ncurl -X POST \\\nhttps://openai-gpt-4-test-v-1.openai.azure.com//openai/ \\\n-H 'Authorization: *****' \\\n-d '{'model': 'chatgpt-v-2', 'messages': [{'role': 'system', 'content': 'you are a helpful assistant.\\n'}, {'role': 'user', 'content': 'bom dia'}], 'stream': False, 'max_tokens': 10, 'user': '116544810872468347480', 'extra_body': {}}'\n",
            },
        )
    except:
        pass

    await asyncio.sleep(5)

    # Get the current date
    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Modify the object_name to include the date-based folder
    object_name = gcs_log_id

    print("object_name", object_name)

    # Check if object landed on GCS
    object_from_gcs = await gcs_logger.download_gcs_object(object_name=object_name)
    print("object from gcs=", object_from_gcs)
    # convert object_from_gcs from bytes to DICT
    parsed_data = json.loads(object_from_gcs)
    print("object_from_gcs as dict", parsed_data)

    print("type of object_from_gcs", type(parsed_data))

    gcs_payload = StandardLoggingPayload(**parsed_data)

    print("gcs_payload", gcs_payload)

    assert gcs_payload["model"] == "gpt-3.5-turbo"
    assert gcs_payload["messages"] == [{"role": "user", "content": "This is a test"}]

    assert gcs_payload["response_cost"] == 0
    assert gcs_payload["status"] == "failure"

    assert (
        gcs_payload["metadata"]["user_api_key_hash"]
        == "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b"
    )
    assert gcs_payload["metadata"]["user_api_key_user_id"] == "116544810872468347480"

    # Delete Object from GCS
    print("deleting object from GCS")
    await gcs_logger.delete_gcs_object(object_name=object_name)
