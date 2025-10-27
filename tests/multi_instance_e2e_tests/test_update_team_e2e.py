import pytest
import asyncio
import aiohttp
import json
from httpx import AsyncClient
from typing import Any, Optional


# =====================================================================
# NEW HELPER FUNCTIONS FOR TEAM BLOCKING TESTS
# =====================================================================
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


async def update_team_block_status(session, team_id: str, blocked: bool, port: int):
    """Helper to update a team's 'blocked' status on a given instance port."""
    url = f"http://0.0.0.0:{port}/team/update"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"team_id": team_id, "blocked": blocked}
    async with session.post(url, headers=headers, json=data) as response:
        return await response.json()


async def get_team_info(session, team_id: str, port: int):
    """Helper to retrieve team info from a specific instance port."""
    url = f"http://0.0.0.0:{port}/team/info"
    headers = {"Authorization": "Bearer sk-1234"}
    async with session.get(
        url, headers=headers, params={"team_id": team_id}
    ) as response:
        data = await response.json()
        return data["team_info"]


async def chat_completion_on_port(
    session, key: str, model: str, port: int, prompt: Optional[str] = None
):
    """
    Helper function to make a chat completion request on a specified instance port.
    Accepts an optional prompt string.
    """
    from openai import AsyncOpenAI
    from litellm._uuid import uuid

    if prompt is None:
        prompt = f"Say hello! {uuid.uuid4()}" * 100
    client = AsyncOpenAI(api_key=key, base_url=f"http://0.0.0.0:{port}/v1")
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response


# =====================================================================
# NEW END‑TO‑END TEST FOR TEAM BLOCKING ACROSS MULTI‑INSTANCE SETUP
# =====================================================================


@pytest.mark.asyncio
async def test_team_blocking_behavior_multi_instance():
    """
    Test team blocking scenario across multi-instance setup:

    1. Create a new team on port 4000.
    2. Verify (via team/info on port 4001) that the team is not blocked.
    3. Create a key for that team.
    4. Make a chat completion request (via instance on port 4000) and verify that it works.
    6. Update the team to set 'blocked': True via the update endpoint on port 4001.
    --- Sleep for 61 seconds --- the in-memory team obj ttl is 60 seconds
    7. Verify (via team/info on port 4000) that the team is now blocked.
    8. Make a chat completion request (using instance on port 4000) with a new prompt; expect it to be blocked.
    9. Repeat the chat completion request with another new prompt; expect it to be blocked.
    10. Confirm via team/info endpoints on both ports that the team remains blocked.
    """
    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": "Bearer sk-1234",
            "Content-Type": "application/json",
        }

        # 1. Create a new team on instance (port 4000)
        url_new_team = "http://0.0.0.0:4000/team/new"
        team_data = {}
        async with session.post(
            url_new_team, headers=headers, json=team_data
        ) as response:
            assert response.status == 200, "Failed to create team"
            team_resp = await response.json()
        team_id = team_resp["team_id"]

        # 2. Verify via team/info on port 4001 that team is not blocked.
        team_info_4001 = await get_team_info(session, team_id, port=4001)
        assert "blocked" in team_info_4001, "Team info missing 'blocked' field"
        assert (
            team_info_4001["blocked"] is False
        ), "Team should not be blocked initially"

        # 3. Create a key for the team using the existing helper.
        key_gen = await generate_team_key(session=session, team_id=team_id)
        key = key_gen["key"]

        # 4. Make a chat completion request on port 4000 and verify it works.
        response = await chat_completion_on_port(
            session,
            key=key,
            model="fake-openai-endpoint",
            port=4000,
            prompt="Non-cached prompt 1",
        )
        assert (
            response is not None
        ), "Chat completion should succeed when team is not blocked"

        # 5. Update the team to set 'blocked': True on instance port 4001.
        await update_team_block_status(session, team_id, blocked=True, port=4001)
        print("sleeping for 61 seconds")
        await asyncio.sleep(61)

        # 6. Verify via team/info on port 4000 that the team is blocked.
        team_info_4000 = await get_team_info(session, team_id, port=4000)
        assert "blocked" in team_info_4000, "Team info missing 'blocked' field"
        print(
            "Team info on port 4000: ",
            json.dumps(team_info_4000, indent=4, default=str),
        )
        assert team_info_4000["blocked"] is True, "Team should be blocked after update"
        # 7.  Verify via team/info on port 4001 that the team is blocked.
        team_info_4001 = await get_team_info(session, team_id, port=4001)
        assert "blocked" in team_info_4001, "Team info missing 'blocked' field"
        assert team_info_4001["blocked"] is True, "Team should be blocked after update"

        # 8. Make a chat completion request on port 4000 with a new prompt; expect it to be blocked.
        with pytest.raises(Exception) as excinfo:
            await chat_completion_on_port(
                session,
                key=key,
                model="fake-openai-endpoint",
                port=4001,
                prompt="Non-cached prompt 2",
            )
        error_msg = str(excinfo.value)
        assert (
            "blocked" in error_msg.lower()
        ), f"Expected error indicating team blocked, got: {error_msg}"

        # 9. Make a chat completion request on port 4000 with a new prompt; expect it to be blocked.
        with pytest.raises(Exception) as excinfo:
            await chat_completion_on_port(
                session,
                key=key,
                model="fake-openai-endpoint",
                port=4000,
                prompt="Non-cached prompt 2",
            )
        error_msg = str(excinfo.value)
        assert (
            "blocked" in error_msg.lower()
        ), f"Expected error indicating team blocked, got: {error_msg}"

        # 9. Repeat the chat completion request with another new prompt; expect it to be blocked.
        with pytest.raises(Exception) as excinfo_second:
            await chat_completion_on_port(
                session,
                key=key,
                model="fake-openai-endpoint",
                port=4000,
                prompt="Non-cached prompt 3",
            )
        error_msg_second = str(excinfo_second.value)
        assert (
            "blocked" in error_msg_second.lower()
        ), f"Expected error indicating team blocked, got: {error_msg_second}"

        # 10. Final verification: check team info on both ports indicates the team is blocked.
        final_team_info_4000 = await get_team_info(session, team_id, port=4000)
        final_team_info_4001 = await get_team_info(session, team_id, port=4001)
        assert (
            final_team_info_4000.get("blocked") is True
        ), "Team on port 4000 should be blocked"
        assert (
            final_team_info_4001.get("blocked") is True
        ), "Team on port 4001 should be blocked"
