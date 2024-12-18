import pytest
import asyncio
import aiohttp
from typing import Optional, List, Union


async def generate_key(session, rpm_limit: int, port: int = 4000):
    url = f"http://0.0.0.0:{port}/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"rpm_limit": rpm_limit}

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        return await response.json()


async def update_key(session, key: str, rpm_limit: int, port: int = 4000):
    url = f"http://0.0.0.0:{port}/key/update"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"key": key, "rpm_limit": rpm_limit}

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        return await response.json()


async def chat_completion(session, key: str, port: int = 4000):
    url = f"http://0.0.0.0:{port}/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "fake-openai-endpoint-all-users",
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    async with session.post(url, headers=headers, json=data) as response:
        return response.status


async def simulated_traffic_for_key(session, key: str, port: int = 4000):
    print(f"simulating traffic for key {key} on port {port}")
    for i in range(100):
        print(f"simulating traffic - chat completion number {i}, port {port}")
        await chat_completion(session, key, port=port)
        await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_multi_instance_key_management():
    """
    Test key management across multiple LiteLLM instances:
    1. Create key on instance 1 (port 4000) with RPM=1
    2. Test key on instance 2 (port 4001) - expect only 1/5 requests to succeed
    3. Update key on instance 1 to RPM=1000
    4. Test key on instance 2 - expect all 5 requests to succeed
    """
    async with aiohttp.ClientSession() as session:
        # Create key on instance 1 with RPM=1
        key_response = await generate_key(session, rpm_limit=1, port=4000)
        test_key = key_response["key"]
        print("created key with rpm limit 1")

        # Test key on instance 2 with 5 requests
        print("running 5 requests on instance 2. Expecting 1 to succeed")
        statuses = []
        for _ in range(5):
            status = await chat_completion(session, test_key, port=4001)
            statuses.append(status)
            await asyncio.sleep(0.1)  # Small delay between requests

        # Expect only first request to succeed
        success_count = sum(1 for status in statuses if status == 200)
        print(
            f"statuses count of 5 /chat/completions: {statuses}. EXPECTED 1 SUCCESS (200)"
        )
        assert success_count == 1, f"Expected 1 successful request, got {success_count}"

        # Update key on instance 1 to RPM=1000
        await update_key(session, test_key, rpm_limit=1000, port=4000)
        print("updated key on instance 1 to rpm limit 1000")
        print("simulating /chat/completion straffic for 60 seconds")

        # create task to simulate traffic for key
        asyncio.create_task(simulated_traffic_for_key(session, test_key, port=4000))
        asyncio.create_task(simulated_traffic_for_key(session, test_key, port=4001))

        # wait for 60 seconds for traffic to propagate
        await asyncio.sleep(60)  # Wait for key update to propagate

        print("\n\n Done simulating traffic for 60 seconds \n\n")
        print("\n\n Now testing if key has new rpm_limit=1000 \n\n")
        print("running 5 requests on instance 2. Expecting 5 to succeed")
        # Test key again on instance 2 with 5 requests
        statuses = []
        for _ in range(5):
            status = await chat_completion(session, test_key, port=4001)
            statuses.append(status)
            await asyncio.sleep(0.1)
        print(f"status of 5 /chat/completions: {statuses}. Expecting 200 for all 5")

        # Expect all requests to succeed
        success_count = sum(1 for status in statuses if status == 200)
        assert (
            success_count == 5
        ), f"Expected 5 successful requests, got {success_count}"
