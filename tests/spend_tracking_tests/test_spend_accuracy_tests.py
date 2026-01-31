import pytest
import asyncio
import aiohttp
import json
from httpx import AsyncClient
from typing import Any, Optional
from litellm._uuid import uuid

"""
Tests to run

Basic Tests:
1. Basic Spend Accuracy Test:
    - 1 Request costs $0.037
    - Make 12 requests 
    - Expect the spend for each of the following to be 12 * $0.037
        Key: $0.444 (call /info endpoint for each object to validate)
        Team: $0.444
        User: $0.444
        Org: $0.444
        End User: $0.444

2. Long term spend accuracy test (with 2 bursts of requests)
    - 1 Request costs $0.037
    - Burst 1: 12 requests
    - Burst 2: 22 requests
    
    - Expect the spend for each of the following to be (12 + 22) * $0.037
        Key: $1.296
        Team: $1.296
        User: $1.296
        Org: $1.296
        End User: $1.296

Additional Test Scenarios:

3. Concurrent Request Accuracy Test:
    - Make 20 concurrent requests
    - Verify total spend is 20 * $0.037
    - Check for race conditions in spend tracking

4. Error Case Test:
    - Make 10 successful requests ($0.037 each)
    - Make 5 failed requests
    - Verify spend is only counted for successful requests (10 * $0.037)

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


@pytest.mark.asyncio
async def test_basic_spend_accuracy():
    """
    Test basic spend accuracy across different entities:
    1. Create org, team, user, and key
    2. Make 12 requests at $0.037 each
    3. Verify spend accuracy for key, team, user, org, and end user
    """
    SPEND_PER_REQUEST = 3.75 * 10**-5
    NUM_LLM_REQUESTS = 20
    expected_spend = NUM_LLM_REQUESTS * SPEND_PER_REQUEST  # 12 requests at $0.037 each

    # Add tolerance constant at the top of the test
    TOLERANCE = 1e-10  # Small number to account for floating-point precision

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

        # Make 12 requests
        for _ in range(NUM_LLM_REQUESTS):
            response = await chat_completion(session, key)
            print("response: ", response)

        # wait 15 seconds for spend to be updated
        await asyncio.sleep(15)

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
        ), f"User spend {user_info['info']['spend']} does not match expected {expected_spend}"

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
    2. Burst 1: Make 12 requests
    3. Burst 2: Make 22 more requests
    4. Verify the total spend (34 requests) is tracked accurately across all entities
    """
    SPEND_PER_REQUEST = 3.75 * 10**-5  # Cost per request
    BURST_1_REQUESTS = 22  # Number of requests in first burst
    BURST_2_REQUESTS = 12  # Number of requests in second burst
    TOTAL_REQUESTS = BURST_1_REQUESTS + BURST_2_REQUESTS
    expected_spend = TOTAL_REQUESTS * SPEND_PER_REQUEST

    # Tolerance for floating-point comparison
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

        # First burst: 12 requests
        print(f"Starting first burst of {BURST_1_REQUESTS} requests...")
        for i in range(BURST_1_REQUESTS):
            response = await chat_completion(session, key)
            print(f"Burst 1 - Request {i+1}/{BURST_1_REQUESTS} completed")

        # Wait for spend to be updated
        await asyncio.sleep(15)

        # Check intermediate spend
        intermediate_key_info = await get_spend_info(session, "key", key)
        print(f"After Burst 1 - Key spend: {intermediate_key_info['info']['spend']}")

        # Second burst: 22 requests
        print(f"Starting second burst of {BURST_2_REQUESTS} requests...")
        for i in range(BURST_2_REQUESTS):
            response = await chat_completion(session, key)
            print(f"Burst 2 - Request {i+1}/{BURST_2_REQUESTS} completed")

        # Wait for spend to be updated
        await asyncio.sleep(15)

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
