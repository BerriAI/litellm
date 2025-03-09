"""
Load test on vertex AI embeddings to ensure vertex median response time is less than 300ms

"""

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


async def create_async_vertex_embedding_task():
    load_vertex_ai_credentials()
    base_url = "https://exampleopenaiendpoint-production.up.railway.app/v1/projects/pathrise-convert-1606954137718/locations/us-central1/publishers/google/models/embedding-gecko-001:predict"
    embedding_args = {
        "model": "vertex_ai/textembedding-gecko",
        "input": "This is a test sentence for embedding.",
        "timeout": 10,
        "api_base": base_url,
    }
    start_time = time.time()
    response = await litellm.aembedding(**embedding_args)
    end_time = time.time()
    print(f"Vertex AI embedding time: {end_time - start_time:.2f} seconds")
    return response, end_time - start_time


async def run_load_test(duration_seconds, requests_per_second):
    end_time = time.time() + duration_seconds
    vertex_times = []

    print(
        f"Running Load Test for {duration_seconds} seconds at {requests_per_second} RPS..."
    )
    while time.time() < end_time:
        vertex_tasks = [
            create_async_vertex_embedding_task() for _ in range(requests_per_second)
        ]

        vertex_results = await asyncio.gather(*vertex_tasks)

        vertex_times.extend([duration for _, duration in vertex_results])

        # Sleep for 1 second to maintain the desired RPS
        await asyncio.sleep(1)

    return vertex_times


def analyze_results(vertex_times):
    median_vertex = median(vertex_times)
    print(f"Vertex AI median response time: {median_vertex:.4f} seconds")

    if median_vertex > 0.3:
        pytest.fail(
            f"Vertex AI median response time is greater than 300ms: {median_vertex:.4f} seconds"
        )
    else:
        print("Performance is good")
        return True


@pytest.mark.asyncio
async def test_embedding_performance():
    """
    Run load test on vertex AI embeddings to ensure vertex median response time is less than 300ms

    20 RPS for 20 seconds
    """
    duration_seconds = 20
    requests_per_second = 20
    vertex_times = await run_load_test(duration_seconds, requests_per_second)
    result = analyze_results(vertex_times)
