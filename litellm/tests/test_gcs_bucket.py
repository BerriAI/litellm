import io
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import json
import logging
import tempfile
import uuid

import pytest

import litellm
from litellm import completion
from litellm._logging import verbose_logger
from litellm.integrations.gcs_bucket import GCSBucketLogger, GCSBucketPayload

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
    )

    print("response", response)

    await asyncio.sleep(5)

    # Check if object landed on GCS
    object_from_gcs = await gcs_logger.download_gcs_object(object_name=response.id)
    # convert object_from_gcs from bytes to DICT
    object_from_gcs = json.loads(object_from_gcs)
    print("object_from_gcs", object_from_gcs)

    gcs_payload = GCSBucketPayload(**object_from_gcs)

    print("gcs_payload", gcs_payload)

    assert gcs_payload["request_kwargs"]["model"] == "gpt-3.5-turbo"
    assert gcs_payload["request_kwargs"]["messages"] == [
        {"role": "user", "content": "This is a test"}
    ]
    assert gcs_payload["response_obj"]["choices"][0]["message"]["content"] == "Hi!"

    # Delete Object from GCS
    print("deleting object from GCS")
    await gcs_logger.delete_gcs_object(object_name=response.id)
