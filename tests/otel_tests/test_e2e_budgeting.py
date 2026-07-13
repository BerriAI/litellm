import asyncio
import json
import secrets
import uuid
from typing import Any, Optional

import aiohttp
import pytest
from httpx import AsyncClient

PROXY_BASE = "http://0.0.0.0:4000"
MASTER_HEADERS = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
CLI_SSO_MODEL = "fake-openai-endpoint"


async def make_calls_until_budget_exceeded(session, key: str, call_function, **kwargs):
    """Helper function to make API calls until budget is exceeded. Verify that the budget is exceeded error is returned."""
    MAX_CALLS = 200
    call_count = 0
    try:
        while call_count < MAX_CALLS:
            await call_function(session=session, key=key, **kwargs)
            call_count += 1
            await asyncio.sleep(0.1)  # allow spend tracking to catch up
        pytest.fail(f"Budget was not exceeded after {MAX_CALLS} calls")
    except Exception as e:
        print("vars: ", vars(e))
        print("e.body: ", e.body)

        error_dict = e.body
        print("error_dict: ", error_dict)

        # Check error structure and values that should be consistent
        assert (
            error_dict["code"] == "429"
        ), f"Expected error code 429, got: {error_dict['code']}"
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
    models: Optional[list[str]] = None,
    team_alias: Optional[str] = None,
):
    """Helper function to create a new team"""
    url = f"{PROXY_BASE}/team/new"
    data: dict[str, Any] = {"max_budget": max_budget}
    if models is not None:
        data["models"] = models
    if team_alias is not None:
        data["team_alias"] = team_alias
    async with session.post(url, headers=MASTER_HEADERS, json=data) as response:
        return await response.json()


async def create_user(
    session,
    *,
    user_id: str,
    user_email: str,
    teams: list[str],
    models: list[str],
):
    url = f"{PROXY_BASE}/user/new"
    data = {
        "user_id": user_id,
        "user_email": user_email,
        "teams": teams,
        "models": models,
        "auto_create_key": False,
    }
    async with session.post(url, headers=MASTER_HEADERS, json=data) as response:
        return await response.json()


async def add_team_member(
    session,
    *,
    team_id: str,
    user_id: str,
    user_email: str,
):
    url = f"{PROXY_BASE}/team/member_add"
    data = {
        "team_id": team_id,
        "member": [{"user_id": user_id, "user_email": user_email, "role": "user"}],
    }
    async with session.post(url, headers=MASTER_HEADERS, json=data) as response:
        return await response.json()


async def obtain_cli_sso_token_via_poll_flow(
    session,
    *,
    user_id: str,
    user_email: str,
    team_id: str,
    team_alias: str,
    models: list[str],
) -> str:
    """
    Obtain a CLI SSO JWT through the same HTTP flow as `litellm-proxy login`:
    /sso/cli/start -> (SSO callback) -> /sso/cli/complete -> /sso/cli/poll.

    When the proxy SSO session cache is not shared with the test runner (otel CI
    uses an isolated in-container cache), falls back to minting the identical JWT
    that /sso/cli/poll would return.
    """
    async with session.post(f"{PROXY_BASE}/sso/cli/start") as resp:
        resp.raise_for_status()
        start = await resp.json()

    login_id = start["login_id"]
    poll_secret = start["poll_secret"]
    user_code = start["user_code"]
    browser_complete_token = secrets.token_urlsafe(32)

    seeded = await _seed_cli_sso_flow_in_shared_redis(
        login_id=login_id,
        user_id=user_id,
        user_email=user_email,
        team_id=team_id,
        team_alias=team_alias,
        models=models,
        browser_complete_token=browser_complete_token,
    )
    if not seeded:
        pytest.skip("Shared Redis not available; skipping full poll-flow test")

    async with session.post(
        f"{PROXY_BASE}/sso/cli/complete/{login_id}",
        data={
            "user_code": user_code,
            "browser_complete_token": browser_complete_token,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    ) as resp:
        assert resp.status == 200, await resp.text()

    poll_headers = {
        "x-litellm-cli-poll-secret": poll_secret,
    }
    async with session.get(
        f"{PROXY_BASE}/sso/cli/poll/{login_id}",
        params={"team_id": team_id},
        headers=poll_headers,
    ) as resp:
        poll = await resp.json()

    assert poll.get("status") == "ready", poll
    assert "key" in poll, poll
    return poll["key"]


async def _seed_cli_sso_flow_in_shared_redis(
    *,
    login_id: str,
    user_id: str,
    user_email: str,
    team_id: str,
    team_alias: str,
    models: list[str],
    browser_complete_token: str,
) -> bool:
    """Seed the CLI SSO flow in Redis when tests share the proxy's Redis instance."""
    import ast
    import json
    import os

    try:
        import redis
    except ImportError:
        return False

    host = os.getenv("REDIS_HOST")
    if not host:
        return False

    try:
        client = redis.Redis(
            host=host,
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD") or None,
            decode_responses=True,
        )
        client.ping()
    except Exception:
        return False

    from litellm.proxy.management_endpoints.ui_sso import (
        _get_cli_sso_flow_cache_key,
        _hash_cli_sso_secret,
    )

    cache_key = _get_cli_sso_flow_cache_key(login_id)
    raw_flow = client.get(cache_key)
    if raw_flow is None:
        return False

    try:
        flow = ast.literal_eval(raw_flow)
    except (SyntaxError, ValueError):
        return False

    if not isinstance(flow, dict):
        return False

    updated_flow = {
        **flow,
        "sso_complete": True,
        "user_code_verified": False,
        "session_data": {
            "user_id": user_id,
            "user_role": "internal_user",
            "models": models,
            "user_email": user_email,
            "teams": [team_id],
            "team_details": [{"team_id": team_id, "team_alias": team_alias}],
        },
        "browser_complete_token_hash": _hash_cli_sso_secret(browser_complete_token),
    }
    client.setex(cache_key, 600, json.dumps(updated_flow))
    return True


async def make_calls_until_team_budget_exceeded_cli_sso(
    session,
    token: str,
    team_id: str,
    model: str,
):
    """Like make_calls_until_budget_exceeded but asserts team budget blocked the CLI SSO token."""
    MAX_CALLS = 200
    call_count = 0
    try:
        while call_count < MAX_CALLS:
            await chat_completion(session=session, key=token, model=model)
            call_count += 1
            await asyncio.sleep(0.1)
        pytest.fail(f"Budget was not exceeded after {MAX_CALLS} calls")
    except Exception as e:
        error_dict = e.body
        assert error_dict["code"] == "429"
        assert error_dict["type"] == "budget_exceeded"
        message = error_dict["message"]
        assert "Budget has been exceeded!" in message
        assert "Team=" in message, f"Expected team budget error, got: {message}"
        assert team_id in message, f"Expected team id in error, got: {message}"
        return call_count


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
async def test_team_budget_enforcement_cli_sso_token():
    """
    Team budget enforcement for CLI SSO session tokens (litellm-proxy login JWT).

    1. Create team with a tiny max_budget and a user on that team
    2. Obtain a CLI SSO JWT (HTTP poll flow when Redis is shared, else mint)
    3. Make chat completion calls until the team budget is exceeded
    4. Verify HTTP 429 budget_exceeded names the team
    """
    user_id = f"cli-budget-user-{uuid.uuid4().hex[:8]}"
    user_email = f"{user_id}@example.com"
    team_alias = f"cli-budget-team-{uuid.uuid4().hex[:8]}"

    async with aiohttp.ClientSession() as session:
        team_response = await create_team(
            session=session,
            max_budget=0.0000000005,
            models=[CLI_SSO_MODEL],
            team_alias=team_alias,
        )
        team_id = team_response["team_id"]

        await create_user(
            session,
            user_id=user_id,
            user_email=user_email,
            teams=[team_id],
            models=[CLI_SSO_MODEL],
        )
        await add_team_member(
            session,
            team_id=team_id,
            user_id=user_id,
            user_email=user_email,
        )

        cli_token = await obtain_cli_sso_token_via_poll_flow(
            session,
            user_id=user_id,
            user_email=user_email,
            team_id=team_id,
            team_alias=team_alias,
            models=[CLI_SSO_MODEL],
        )
        assert not cli_token.startswith(
            "sk-"
        ), "CLI SSO token must not be a virtual key"

        calls_made = await make_calls_until_team_budget_exceeded_cli_sso(
            session=session,
            token=cli_token,
            team_id=team_id,
            model=CLI_SSO_MODEL,
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
