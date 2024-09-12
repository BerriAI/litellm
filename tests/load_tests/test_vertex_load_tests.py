import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import litellm
import pytest
import time


@pytest.mark.asyncio
async def test_vertex_load():
    try:
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
            avg_percentage_diff < 20
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
    base_url = "https://exampleopenaiendpoint-production.up.railway.app/v1/projects/adroit-crow-413218/locations/us-central1/publishers/google/models/gemini-1.0-pro-vision-001"

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
