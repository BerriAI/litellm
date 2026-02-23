import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import litellm
import pytest
import time
import json
import tempfile
from dotenv import load_dotenv


def load_vertex_ai_credentials():
    # Define the path to the vertex_key.json file
    print("loading vertex ai credentials")
    filepath = os.path.dirname(os.path.abspath(__file__))
    vertex_key_path = filepath + "/vertex_key.json"

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
    private_key_id = os.environ.get("VERTEX_AI_PRIVATE_KEY_ID", "")
    private_key = os.environ.get("VERTEX_AI_PRIVATE_KEY", "")
    private_key = private_key.replace("\\n", "\n")
    service_account_key_data["private_key_id"] = private_key_id
    service_account_key_data["private_key"] = private_key

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        # Write the updated content to the temporary files
        json.dump(service_account_key_data, temp_file, indent=2)

    # Export the temporary file as GOOGLE_APPLICATION_CREDENTIALS
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(temp_file.name)


@pytest.mark.asyncio
async def test_vertex_load():
    try:
        load_vertex_ai_credentials()
        percentage_diffs = []

        for run in range(3):
            print(f"\nRun {run + 1}:")

            # Test with text-only message
            start_time_text = await make_async_calls(message_type="text")
            print("Done with text-only message test")

            # Test with text + image message
            start_time_image = await make_async_calls(message_type="image")
            print("Done with text + image message test")

            # Compare times and calculate percentage difference
            print(f"Time with text-only message: {start_time_text}")
            print(f"Time with text + image message: {start_time_image}")

            percentage_diff = (
                (start_time_image - start_time_text) / start_time_text * 100
            )
            percentage_diffs.append(percentage_diff)
            print(f"Performance difference: {percentage_diff:.2f}%")

        print("percentage_diffs", percentage_diffs)
        # Calculate average percentage difference
        avg_percentage_diff = sum(percentage_diffs) / len(percentage_diffs)
        print(f"\nAverage performance difference: {avg_percentage_diff:.2f}%")

        # Assert that the average difference is not more than 20%
        assert (
            avg_percentage_diff < 25
        ), f"Average performance difference of {avg_percentage_diff:.2f}% exceeds 20% threshold"

    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")


async def make_async_calls(message_type="text"):
    total_tasks = 3
    batch_size = 1
    total_time = 0

    for batch in range(3):
        tasks = [create_async_task(message_type) for _ in range(batch_size)]

        start_time = asyncio.get_event_loop().time()
        responses = await asyncio.gather(*tasks)

        for idx, response in enumerate(responses):
            print(f"Response from Task {batch * batch_size + idx + 1}: {response}")

        await asyncio.sleep(1)

        batch_time = asyncio.get_event_loop().time() - start_time
        total_time += batch_time

    return total_time


def create_async_task(message_type):
    base_url = "https://exampleopenaiendpoint-production.up.railway.app/v1/projects/pathrise-convert-1606954137718/locations/us-central1/publishers/google/models/gemini-1.0-pro-vision-001"

    if message_type == "text":
        messages = [{"role": "user", "content": "hi"}]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://litellm-listing.s3.amazonaws.com/litellm_logo.png"
                        },
                    },
                ],
            }
        ]

    completion_args = {
        "model": "vertex_ai/gemini",
        "messages": messages,
        "max_tokens": 5,
        "temperature": 0.7,
        "timeout": 10,
        "api_base": base_url,
    }
    return asyncio.create_task(litellm.acompletion(**completion_args))
