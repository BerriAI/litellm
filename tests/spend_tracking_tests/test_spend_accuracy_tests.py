import pytest
import asyncio
import aiohttp
import json
import time
from httpx import AsyncClient
from typing import Any, Optional
from litellm._uuid import uuid

"""
Tests to run

Basic Tests:
1. Basic Spend Accuracy Test:
    - Make 1 calibration request, poll for spend to derive SPEND_PER_REQUEST
    - Make N-1 more requests (N total)
    - Expect the spend for each of the following to be N * SPEND_PER_REQUEST
        Key, Team, User, Org (call /info endpoint for each object to validate)

2. Long term spend accuracy test (with 2 bursts of requests)
    - Burst 1: Make requests, derive SPEND_PER_REQUEST from first request
    - Burst 2: Make more requests
    - Verify total spend = (burst1 + burst2) * SPEND_PER_REQUEST

Additional Test Scenarios:

3. Concurrent Request Accuracy Test:
    - Make 20 concurrent requests
    - Check for race conditions in spend tracking

4. Error Case Test:
    - Make 10 successful requests
    - Make 5 failed requests
    - Verify spend is only counted for successful requests

5. Mixed Request Type Test:
    - Make different types of requests with varying costs
    - Verify accurate total spend calculation
"""


async def create_organization(session, organization_alias: str):
    """Helper function to create a new organization"""
    url = "http://0.0.0.0:4000/organization/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"organization_alias": organization_alias}
    async with session.post(url, headers=headers, json=data) as response:
        return await response.json()


async def create_team(session, org_id: str):
    """Helper function to create a new team under an organization"""
    url = "http://0.0.0.0:4000/team/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"organization_id": org_id, "team_alias": f"test-team-{uuid.uuid4()}"}
    async with session.post(url, headers=headers, json=data) as response:
        return await response.json()


async def create_user(session, org_id: str):
    """Helper function to create a new user"""
    url = "http://0.0.0.0:4000/user/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"user_name": f"test-user-{uuid.uuid4()}"}
    async with session.post(url, headers=headers, json=data) as response:
        return await response.json()


async def generate_key(session, user_id: str, team_id: str):
    """Helper function to generate a key for a specific user and team"""
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"user_id": user_id, "team_id": team_id}
    async with session.post(url, headers=headers, json=data) as response:
        return await response.json()


async def chat_completion(session, key: str):
    """Make a chat completion request"""
    from openai import AsyncOpenAI
    from litellm._uuid import uuid

    client = AsyncOpenAI(api_key=key, base_url="http://0.0.0.0:4000/v1")

    response = await client.chat.completions.create(
        model="fake-openai-endpoint",
        messages=[{"role": "user", "content": f"Test message {uuid.uuid4()}"}],
    )
    return response


async def get_spend_info(session, entity_type: str, entity_id: str):
    """Helper function to get spend information for an entity"""
    url = f"http://0.0.0.0:4000/{entity_type}/info"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    if entity_type == "key":
        data = {"key": entity_id}
    else:
        data = {f"{entity_type}_id": entity_id}

    async with session.get(url, headers=headers, params=data) as response:
        return await response.json()


async def poll_key_spend_until_nonzero(
    session, key: str, timeout: int = 120, interval: int = 10
):
    """Poll key spend until it becomes non-zero or timeout is reached."""
    start = time.time()
    while time.time() - start < timeout:
        key_info = await get_spend_info(session, "key", key)
        spend = key_info["info"]["spend"]
        if spend > 0:
            print(f"Key spend became non-zero ({spend}) after {time.time() - start:.1f}s")
            return spend
        print(f"Key spend still 0.0, waiting... ({time.time() - start:.1f}s elapsed)")
        await asyncio.sleep(interval)
    raise TimeoutError(
        f"Key spend remained 0.0 after {timeout}s — batch writer may not be running"
    )


async def calibrate_spend_per_request(session, key: str, max_retries: int = 5):
    """
    Make a single calibration request and poll for its spend to derive SPEND_PER_REQUEST.
    Fails fast with pytest.fail() if spend cannot be determined.
    """
    response = await chat_completion(session, key)
    print(f"Calibration request completed: {response}")

    for attempt in range(1, max_retries + 1):
        try:
            spend = await poll_key_spend_until_nonzero(
                session, key, timeout=120, interval=10
            )
            print(
                f"Calibrated SPEND_PER_REQUEST = {spend} "
                f"(attempt {attempt}/{max_retries})"
            )
            return spend
        except TimeoutError:
            if attempt < max_retries:
                print(
                    f"Calibration attempt {attempt}/{max_retries} timed out, retrying..."
                )
            else:
                pytest.fail(
                    f"Failed to calibrate SPEND_PER_REQUEST after {max_retries} attempts. "
                    "The batch writer may not be running or the model may have 0 cost."
                )


@pytest.mark.asyncio
async def test_basic_spend_accuracy():
    """
    Test basic spend accuracy across different entities:
    1. Create org, team, user, and key
    2. Make 1 calibration request to derive SPEND_PER_REQUEST
    3. Make remaining requests (NUM_LLM_REQUESTS total)
    4. Verify spend accuracy for key, team, user, and org
    """
    NUM_LLM_REQUESTS = 20
    TOLERANCE = 1e-10

    async with aiohttp.ClientSession() as session:
        # Create organization
        org_response = await create_organization(
            session=session, organization_alias=f"test-org-{uuid.uuid4()}"
        )
        print("org_response: ", org_response)
        org_id = org_response["organization_id"]

        # Create team under organization
        team_response = await create_team(session, org_id)
        print("team_response: ", team_response)
        team_id = team_response["team_id"]

        # Create user
        user_response = await create_user(session, org_id)
        print("user_response: ", user_response)
        user_id = user_response["user_id"]

        # Generate key
        key_response = await generate_key(session, user_id, team_id)
        print("key_response: ", key_response)
        key = key_response["key"]

        # Calibrate: make 1 request and derive SPEND_PER_REQUEST
        spend_per_request = await calibrate_spend_per_request(session, key)
        expected_spend = NUM_LLM_REQUESTS * spend_per_request
        print(f"SPEND_PER_REQUEST={spend_per_request}, expected_spend={expected_spend}")

        # Make remaining requests (1 already made during calibration)
        for i in range(NUM_LLM_REQUESTS - 1):
            response = await chat_completion(session, key)
            print(f"Request {i + 2}/{NUM_LLM_REQUESTS} completed")

        # Poll until batch writer has flushed all spend
        start = time.time()
        while time.time() - start < 120:
            key_info = await get_spend_info(session, "key", key)
            current_spend = key_info["info"]["spend"]
            if abs(current_spend - expected_spend) < TOLERANCE:
                print(f"Key spend reached expected {expected_spend} after {time.time() - start:.1f}s")
                break
            print(f"Key spend {current_spend}, expected {expected_spend}, waiting...")
            await asyncio.sleep(10)

        # Allow extra time for all entity spend aggregations to complete
        await asyncio.sleep(5)

        # Get spend information for each entity
        key_info = await get_spend_info(session, "key", key)
        print("key_info: ", key_info)
        team_info = await get_spend_info(session, "team", team_id)
        print("team_info: ", team_info)
        user_info = await get_spend_info(session, "user", user_id)
        print("user_info: ", user_info)
        org_info = await get_spend_info(session, "organization", org_id)
        print("org_info: ", org_info)

        # Verify spend for each entity
        assert (
            abs(key_info["info"]["spend"] - expected_spend) < TOLERANCE
        ), f"Key spend {key_info['info']['spend']} does not match expected {expected_spend}"

        assert (
            abs(user_info["user_info"]["spend"] - expected_spend) < TOLERANCE
        ), f"User spend {user_info['user_info']['spend']} does not match expected {expected_spend}"

        assert (
            abs(team_info["team_info"]["spend"] - expected_spend) < TOLERANCE
        ), f"Team spend {team_info['team_info']['spend']} does not match expected {expected_spend}"

        assert (
            abs(org_info["spend"] - expected_spend) < TOLERANCE
        ), f"Organization spend {org_info['spend']} does not match expected {expected_spend}"


@pytest.mark.asyncio
async def test_long_term_spend_accuracy_with_bursts():
    """
    Test long-term spend accuracy with multiple bursts of requests:
    1. Create org, team, user, and key
    2. Calibrate SPEND_PER_REQUEST from first request
    3. Burst 1: Make remaining requests
    4. Burst 2: Make more requests
    5. Verify the total spend is tracked accurately across all entities
    """
    BURST_1_REQUESTS = 22
    BURST_2_REQUESTS = 12
    TOTAL_REQUESTS = BURST_1_REQUESTS + BURST_2_REQUESTS
    TOLERANCE = 1e-10

    async with aiohttp.ClientSession() as session:
        # Create organization
        org_response = await create_organization(
            session=session, organization_alias=f"test-org-{uuid.uuid4()}"
        )
        print("org_response: ", org_response)
        org_id = org_response["organization_id"]

        # Create team under organization
        team_response = await create_team(session, org_id)
        print("team_response: ", team_response)
        team_id = team_response["team_id"]

        # Create user
        user_response = await create_user(session, org_id)
        print("user_response: ", user_response)
        user_id = user_response["user_id"]

        # Generate key
        key_response = await generate_key(session, user_id, team_id)
        print("key_response: ", key_response)
        key = key_response["key"]

        # Calibrate: make 1 request and derive SPEND_PER_REQUEST
        spend_per_request = await calibrate_spend_per_request(session, key)
        expected_spend = TOTAL_REQUESTS * spend_per_request
        print(f"SPEND_PER_REQUEST={spend_per_request}, expected_spend={expected_spend}")

        # First burst: remaining requests (1 already made during calibration)
        print(f"Starting first burst ({BURST_1_REQUESTS - 1} remaining requests)...")
        for i in range(BURST_1_REQUESTS - 1):
            response = await chat_completion(session, key)
            print(f"Burst 1 - Request {i + 2}/{BURST_1_REQUESTS} completed")

        # Poll until batch writer has flushed burst 1 spend
        burst_1_expected = BURST_1_REQUESTS * spend_per_request
        start = time.time()
        while time.time() - start < 120:
            key_info_check = await get_spend_info(session, "key", key)
            current_spend = key_info_check["info"]["spend"]
            if abs(current_spend - burst_1_expected) < TOLERANCE:
                print(f"Burst 1 spend reached expected {burst_1_expected} after {time.time() - start:.1f}s")
                break
            print(f"Key spend {current_spend}, expected {burst_1_expected}, waiting...")
            await asyncio.sleep(10)

        # Check intermediate spend
        intermediate_key_info = await get_spend_info(session, "key", key)
        print(f"After Burst 1 - Key spend: {intermediate_key_info['info']['spend']}")

        # Second burst
        print(f"Starting second burst of {BURST_2_REQUESTS} requests...")
        for i in range(BURST_2_REQUESTS):
            response = await chat_completion(session, key)
            print(f"Burst 2 - Request {i + 1}/{BURST_2_REQUESTS} completed")

        # Poll until key spend reaches expected total (burst 1 + burst 2)
        start = time.time()
        while time.time() - start < 120:
            key_info_check = await get_spend_info(session, "key", key)
            current_spend = key_info_check["info"]["spend"]
            if abs(current_spend - expected_spend) < TOLERANCE:
                print(
                    f"Total spend reached expected {expected_spend} after {time.time() - start:.1f}s"
                )
                break
            print(
                f"Key spend {current_spend}, expected {expected_spend}, waiting..."
            )
            await asyncio.sleep(10)

        # Allow extra time for all entity spend aggregations
        await asyncio.sleep(5)

        # Get final spend information for each entity
        key_info = await get_spend_info(session, "key", key)
        team_info = await get_spend_info(session, "team", team_id)
        user_info = await get_spend_info(session, "user", user_id)
        org_info = await get_spend_info(session, "organization", org_id)

        print(f"Final key spend: {key_info['info']['spend']}")
        print(f"Final team spend: {team_info['team_info']['spend']}")
        print(f"Final user spend: {user_info['user_info']['spend']}")
        print(f"Final org spend: {org_info['spend']}")

        # Verify total spend for each entity
        assert (
            abs(key_info["info"]["spend"] - expected_spend) < TOLERANCE
        ), f"Key spend {key_info['info']['spend']} does not match expected {expected_spend}"

        assert (
            abs(user_info["user_info"]["spend"] - expected_spend) < TOLERANCE
        ), f"User spend {user_info['user_info']['spend']} does not match expected {expected_spend}"

        assert (
            abs(team_info["team_info"]["spend"] - expected_spend) < TOLERANCE
        ), f"Team spend {team_info['team_info']['spend']} does not match expected {expected_spend}"

        assert (
            abs(org_info["spend"] - expected_spend) < TOLERANCE
        ), f"Organization spend {org_info['spend']} does not match expected {expected_spend}"
