import pytest
import asyncio
import aiohttp
import json
from openai import AsyncOpenAI
import uuid
from httpx import AsyncClient
import uuid
import os

TEST_MASTER_KEY = "sk-1234"
PROXY_BASE_URL = "http://0.0.0.0:4000"


@pytest.mark.asyncio
async def test_team_model_alias():
    """
    Test model alias functionality with teams:
    1. Add a new model with model_name="gpt-4-team1" and litellm_params.model="gpt-4o"
    2. Create a new team
    3. Update team with model_alias mapping
    4. Generate key for team
    5. Make request with aliased model name
    """
    client = AsyncClient(base_url=PROXY_BASE_URL)
    headers = {"Authorization": f"Bearer {TEST_MASTER_KEY}"}

    # Add new model
    model_response = await client.post(
        "/model/new",
        json={
            "model_name": "gpt-4o-team1",
            "litellm_params": {
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        headers=headers,
    )
    assert model_response.status_code == 200

    # Create new team
    team_response = await client.post(
        "/team/new",
        json={
            "models": ["gpt-4o-team1"],
        },
        headers=headers,
    )
    assert team_response.status_code == 200
    team_data = team_response.json()
    team_id = team_data["team_id"]

    # Update team with model alias
    update_response = await client.post(
        "/team/update",
        json={"team_id": team_id, "model_aliases": {"gpt-4o": "gpt-4o-team1"}},
        headers=headers,
    )
    assert update_response.status_code == 200

    # Generate key for team
    key_response = await client.post(
        "/key/generate", json={"team_id": team_id}, headers=headers
    )
    assert key_response.status_code == 200
    key = key_response.json()["key"]

    # Make request with model alias
    openai_client = AsyncOpenAI(api_key=key, base_url=f"{PROXY_BASE_URL}/v1")

    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Test message {uuid.uuid4()}"}],
    )

    assert response is not None, "Should get valid response when using model alias"

    # Cleanup - delete the model
    model_id = model_response.json()["model_info"]["id"]
    delete_response = await client.post(
        "/model/delete",
        json={"id": model_id},
        headers={"Authorization": f"Bearer {TEST_MASTER_KEY}"},
    )
    assert delete_response.status_code == 200
