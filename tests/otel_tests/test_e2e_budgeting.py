import pytest
import asyncio
import aiohttp
import json
from httpx import AsyncClient
from typing import Any, Optional


async def make_calls_until_budget_exceeded(session, key: str, call_function, **kwargs):
    """Helper function to make API calls until budget is exceeded. Verify that the budget is exceeded error is returned."""
    MAX_CALLS = 50
    call_count = 0
    try:
        while call_count < MAX_CALLS:
            await call_function(session=session, key=key, **kwargs)
            call_count += 1
        pytest.fail(f"Budget was not exceeded after {MAX_CALLS} calls")
    except Exception as e:
        print("vars: ", vars(e))
        print("e.body: ", e.body)

        error_dict = e.body
        print("error_dict: ", error_dict)

        # Check error structure and values that should be consistent
        assert (
            error_dict["code"] == "400"
        ), f"Expected error code 400, got: {error_dict['code']}"
        assert (
            error_dict["type"] == "budget_exceeded"
        ), f"Expected error type budget_exceeded, got: {error_dict['type']}"

        # Check message contains required parts without checking specific values
        message = error_dict["message"]
        assert (
            "Budget has been exceeded!" in message
        ), f"Expected message to start with 'Budget has been exceeded!', got: {message}"
        assert (
            "Current cost:" in message
        ), f"Expected message to contain 'Current cost:', got: {message}"
        assert (
            "Max budget:" in message
        ), f"Expected message to contain 'Max budget:', got: {message}"

        return call_count


async def generate_key(
    session,
    max_budget=None,
):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "max_budget": max_budget,
    }
    async with session.post(url, headers=headers, json=data) as response:
        return await response.json()


async def chat_completion(session, key: str, model: str):
    """Make a chat completion request using OpenAI SDK"""
    from openai import AsyncOpenAI
    from litellm._uuid import uuid

    client = AsyncOpenAI(
        api_key=key, base_url="http://0.0.0.0:4000/v1"  # Point to our local proxy
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"Say hello! {uuid.uuid4()}" * 100}],
    )
    return response


async def update_key_budget(session, key: str, max_budget: float):
    """Helper function to update a key's max budget"""
    url = "http://0.0.0.0:4000/key/update"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "key": key,
        "max_budget": max_budget,
    }
    async with session.post(url, headers=headers, json=data) as response:
        return await response.json()


@pytest.mark.asyncio
async def test_chat_completion_low_budget():
    """
    Test budget enforcement for chat completions:
    1. Create key with $0.01 budget
    2. Make chat completion calls until budget exceeded
    3. Verify budget exceeded error
    """
    async with aiohttp.ClientSession() as session:
        # Create key with $0.01 budget
        key_gen = await generate_key(session=session, max_budget=0.0000000005)
        print("response from key generation: ", key_gen)
        key = key_gen["key"]

        # Make calls until budget exceeded
        calls_made = await make_calls_until_budget_exceeded(
            session=session,
            key=key,
            call_function=chat_completion,
            model="fake-openai-endpoint",
        )

        assert (
            calls_made > 0
        ), "Should make at least one successful call before budget exceeded"


@pytest.mark.asyncio
async def test_chat_completion_zero_budget():
    """
    Test budget enforcement for chat completions:
    1. Create key with $0.01 budget
    2. Make chat completion calls until budget exceeded
    3. Verify budget exceeded error
    """
    async with aiohttp.ClientSession() as session:
        # Create key with $0.01 budget
        key_gen = await generate_key(session=session, max_budget=0.000000000)
        print("response from key generation: ", key_gen)
        key = key_gen["key"]

        # Make calls until budget exceeded
        calls_made = await make_calls_until_budget_exceeded(
            session=session,
            key=key,
            call_function=chat_completion,
            model="fake-openai-endpoint",
        )

        assert calls_made == 0, "Should make no calls before budget exceeded"


@pytest.mark.asyncio
async def test_chat_completion_high_budget():
    """
    Test budget enforcement for chat completions:
    1. Create key with $0.01 budget
    2. Make chat completion calls until budget exceeded
    3. Verify budget exceeded error
    """
    async with aiohttp.ClientSession() as session:
        # Create key with $0.01 budget
        key_gen = await generate_key(session=session, max_budget=0.001)
        print("response from key generation: ", key_gen)
        key = key_gen["key"]

        # Make calls until budget exceeded
        calls_made = await make_calls_until_budget_exceeded(
            session=session,
            key=key,
            call_function=chat_completion,
            model="fake-openai-endpoint",
        )

        assert (
            calls_made > 0
        ), "Should make at least one successful call before budget exceeded"


@pytest.mark.asyncio
async def test_chat_completion_budget_update():
    """
    Test that requests continue working after updating a key's budget:
    1. Create key with low budget
    2. Make calls until budget exceeded
    3. Update key with higher budget
    4. Verify calls work again
    """
    async with aiohttp.ClientSession() as session:
        # Create key with very low budget
        key_gen = await generate_key(session=session, max_budget=0.0000000005)
        key = key_gen["key"]

        # Make calls until budget exceeded
        calls_made = await make_calls_until_budget_exceeded(
            session=session,
            key=key,
            call_function=chat_completion,
            model="fake-openai-endpoint",
        )

        assert (
            calls_made > 0
        ), "Should make at least one successful call before budget exceeded"

        # Update key with higher budget
        await update_key_budget(session, key, max_budget=0.001)

        # Verify calls work again
        for _ in range(3):
            try:
                response = await chat_completion(
                    session=session, key=key, model="fake-openai-endpoint"
                )
                print("response: ", response)
                assert (
                    response is not None
                ), "Should get valid response after budget update"
            except Exception as e:
                pytest.fail(
                    f"Request should succeed after budget update but got error: {e}"
                )


@pytest.mark.parametrize(
    "field",
    [
        "max_budget",
        "rpm_limit",
        "tpm_limit",
    ],
)
@pytest.mark.asyncio
async def test_key_limit_modifications(field):
    # Create initial key
    client = AsyncClient(base_url="http://0.0.0.0:4000")
    key_data = {"max_budget": None, "rpm_limit": None, "tpm_limit": None}
    headers = {"Authorization": "Bearer sk-1234"}
    response = await client.post("/key/generate", json=key_data, headers=headers)
    assert response.status_code == 200
    generate_key_response = response.json()
    print("generate_key_response: ", json.dumps(generate_key_response, indent=4))
    key_id = generate_key_response["key"]

    # Update key with any non-null value for the field
    update_data = {"key": key_id}
    update_data[field] = 10  # Any non-null value works
    print("update_data: ", json.dumps(update_data, indent=4))
    response = await client.post(f"/key/update", json=update_data, headers=headers)
    assert response.status_code == 200
    assert response.json()[field] is not None

    # Reset limit to null
    print(f"resetting {field} to null")
    update_data[field] = None
    response = await client.post(f"/key/update", json=update_data, headers=headers)
    print("response: ", json.dumps(response.json(), indent=4))
    assert response.status_code == 200
    assert response.json()[field] is None


@pytest.mark.parametrize(
    "field",
    [
        "max_budget",
    ],
)
@pytest.mark.asyncio
async def test_team_limit_modifications(field):
    # Create initial team
    client = AsyncClient(base_url="http://0.0.0.0:4000")
    team_data = {"max_budget": None, "rpm_limit": None, "tpm_limit": None}
    headers = {"Authorization": "Bearer sk-1234"}
    response = await client.post("/team/new", json=team_data, headers=headers)
    print("response: ", json.dumps(response.json(), indent=4))
    assert response.status_code == 200
    team_id = response.json()["team_id"]

    # Update team with any non-null value for the field
    update_data = {"team_id": team_id}
    update_data[field] = 10  # Any non-null value works
    response = await client.post(f"/team/update", json=update_data, headers=headers)
    print("response: ", json.dumps(response.json(), indent=4))
    assert response.status_code == 200
    assert response.json()["data"][field] is not None

    # Reset limit to null
    print(f"resetting {field} to null")
    update_data[field] = None
    response = await client.post(f"/team/update", json=update_data, headers=headers)
    print("response: ", json.dumps(response.json(), indent=4))
    assert response.status_code == 200
    assert response.json()["data"][field] is None


async def generate_team_key(
    session,
    team_id: str,
    max_budget: Optional[float] = None,
):
    """Helper function to generate a key for a specific team"""
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data: dict[str, Any] = {"team_id": team_id}
    if max_budget is not None:
        data["max_budget"] = max_budget
    async with session.post(url, headers=headers, json=data) as response:
        return await response.json()


async def create_team(
    session,
    max_budget=None,
):
    """Helper function to create a new team"""
    url = "http://0.0.0.0:4000/team/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "max_budget": max_budget,
    }
    async with session.post(url, headers=headers, json=data) as response:
        return await response.json()


@pytest.mark.asyncio
async def test_team_budget_enforcement():
    """
    Test budget enforcement for team-wide budgets:
    1. Create team with low budget
    2. Create key for that team
    3. Make calls until team budget exceeded
    4. Verify budget exceeded error
    """
    async with aiohttp.ClientSession() as session:
        # Create team with low budget
        team_response = await create_team(session=session, max_budget=0.0000000005)
        team_id = team_response["team_id"]

        # Create key for team (no specific budget)
        key_gen = await generate_team_key(session=session, team_id=team_id)
        key = key_gen["key"]

        # Make calls until budget exceeded
        calls_made = await make_calls_until_budget_exceeded(
            session=session,
            key=key,
            call_function=chat_completion,
            model="fake-openai-endpoint",
        )

        assert (
            calls_made > 0
        ), "Should make at least one successful call before team budget exceeded"


@pytest.mark.asyncio
async def test_team_and_key_budget_enforcement():
    """
    Test budget enforcement when both team and key have budgets:
    1. Create team with low budget
    2. Create key with higher budget
    3. Verify team budget is enforced first
    """
    async with aiohttp.ClientSession() as session:
        # Create team with very low budget
        team_response = await create_team(session=session, max_budget=0.0000000005)
        team_id = team_response["team_id"]

        # Create key with higher budget
        key_gen = await generate_team_key(
            session=session,
            team_id=team_id,
            max_budget=0.001,  # Higher than team budget
        )
        key = key_gen["key"]

        # Make calls until budget exceeded
        calls_made = await make_calls_until_budget_exceeded(
            session=session,
            key=key,
            call_function=chat_completion,
            model="fake-openai-endpoint",
        )

        assert (
            calls_made > 0
        ), "Should make at least one successful call before team budget exceeded"

        # Verify it was the team budget that was exceeded
        try:
            await chat_completion(
                session=session, key=key, model="fake-openai-endpoint"
            )
        except Exception as e:
            error_dict = e.body
            assert (
                "Budget has been exceeded! Team=" in error_dict["message"]
            ), "Error should mention team budget being exceeded"

            assert team_id in error_dict["message"], "Error should mention team id"


async def update_team_budget(session, team_id: str, max_budget: float):
    """Helper function to update a team's max budget"""
    url = "http://0.0.0.0:4000/team/update"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "team_id": team_id,
        "max_budget": max_budget,
    }
    async with session.post(url, headers=headers, json=data) as response:
        return await response.json()


@pytest.mark.asyncio
async def test_team_budget_update():
    """
    Test that requests continue working after updating a team's budget:
    1. Create team with low budget
    2. Create key for that team
    3. Make calls until team budget exceeded
    4. Update team with higher budget
    5. Verify calls work again
    """
    async with aiohttp.ClientSession() as session:
        # Create team with very low budget
        team_response = await create_team(session=session, max_budget=0.0000000005)
        team_id = team_response["team_id"]

        # Create key for team (no specific budget)
        key_gen = await generate_team_key(session=session, team_id=team_id)
        key = key_gen["key"]

        # Make calls until budget exceeded
        calls_made = await make_calls_until_budget_exceeded(
            session=session,
            key=key,
            call_function=chat_completion,
            model="fake-openai-endpoint",
        )

        assert (
            calls_made > 0
        ), "Should make at least one successful call before team budget exceeded"

        # Update team with higher budget
        await update_team_budget(session, team_id, max_budget=0.001)

        # Verify calls work again
        for _ in range(3):
            try:
                response = await chat_completion(
                    session=session, key=key, model="fake-openai-endpoint"
                )
                print("response: ", response)
                assert (
                    response is not None
                ), "Should get valid response after budget update"
            except Exception as e:
                pytest.fail(
                    f"Request should succeed after team budget update but got error: {e}"
                )

        # Verify it was the team budget that was exceeded
