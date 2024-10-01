import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import litellm
import pytest
import time
from statistics import mean, median
import json
import tempfile


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
async def test_vertex_embeddings_load():
    try:
        # load_vertex_ai_credentials()
        embedding_times = []

        # Set test parameters
        test_duration = 60  # seconds
        requests_per_second = 20

        print(
            f"\nStarting load test: {requests_per_second} RPS for {test_duration} seconds"
        )

        start_time = time.time()
        while time.time() - start_time < test_duration:
            batch_start = time.time()

            # Make 20 requests
            responses = await asyncio.gather(
                *[create_async_embedding_task() for _ in range(requests_per_second)]
            )

            # Record response times
            embedding_times.extend([resp[1] for resp in responses])

            # Wait until the next second
            elapsed = time.time() - batch_start
            if elapsed < 1:
                await asyncio.sleep(1 - elapsed)

        # Calculate statistics
        avg_time = mean(embedding_times)
        median_time = median(embedding_times)

        print(f"\nTotal requests: {len(embedding_times)}")
        print(f"Average response time: {avg_time:.2f} seconds")
        print(f"Median response time: {median_time:.2f} seconds")

        # Assert that the average and median times are below 150ms
        assert (
            avg_time < 0.15
        ), f"Average response time of {avg_time:.2f} seconds exceeds 150ms threshold"
        assert (
            median_time < 0.15
        ), f"Median response time of {median_time:.2f} seconds exceeds 150ms threshold"

    except Exception as e:
        import traceback

        traceback.print_exc()
        pytest.fail(f"An exception occurred - {e}")


async def create_async_embedding_task():
    base_url = "https://exampleopenaiendpoint-production.up.railway.app/v1/projects/adroit-crow-413218/locations/us-central1/publishers/google/models/embedding-gecko-001:predict"
    embedding_args = {
        "model": "vertex_ai/textembedding-gecko",
        "input": "This is a test sentence for embedding.",
        "timeout": 10,
        "api_base": base_url,
    }
    start_time = time.time()
    response = await litellm.aembedding(**embedding_args)
    end_time = time.time()
    return response, end_time - start_time
