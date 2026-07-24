import pytest
import asyncio
import aiohttp
import json
from openai import AsyncOpenAI
from litellm._uuid import uuid
from httpx import AsyncClient
from litellm._uuid import uuid
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


@pytest.mark.asyncio
async def test_team_model_association():
    """
    Test that models created with a team_id are properly associated with the team:
    1. Create a new team
    2. Add a model with team_id in model_info
    3. Verify the model appears in team info
    """
    client = AsyncClient(base_url=PROXY_BASE_URL)
    headers = {"Authorization": f"Bearer {TEST_MASTER_KEY}"}

    # Create new team
    team_response = await client.post(
        "/team/new",
        json={
            "models": [],  # Start with empty model list
        },
        headers=headers,
    )
    assert team_response.status_code == 200
    team_data = team_response.json()
    team_id = team_data["team_id"]

    # Add new model with team_id
    model_response = await client.post(
        "/model/new",
        json={
            "model_name": "gpt-4-team-test",
            "litellm_params": {
                "model": "gpt-4",
                "custom_llm_provider": "openai",
                "api_key": "fake_key",
            },
            "model_info": {"team_id": team_id},
        },
        headers=headers,
    )
    assert model_response.status_code == 200

    # Get team info and verify model association
    team_info_response = await client.get(
        f"/team/info",
        headers=headers,
        params={"team_id": team_id},
    )
    assert team_info_response.status_code == 200
    team_info = team_info_response.json()["team_info"]

    print("team_info", json.dumps(team_info, indent=4))

    # Verify the model is in team_models
    assert (
        "gpt-4-team-test" in team_info["models"]
    ), "Model should be associated with team"

    # Cleanup - delete the model
    model_id = model_response.json()["model_info"]["id"]
    delete_response = await client.post(
        "/model/delete",
        json={"id": model_id},
        headers=headers,
    )
    assert delete_response.status_code == 200


@pytest.mark.asyncio
async def test_team_model_visibility_in_models_endpoint():
    """
    Test that team-specific models are only visible to the correct team in /models endpoint:
    1. Create two teams
    2. Add a model associated with team1
    3. Generate keys for both teams
    4. Verify team1's key can see the model in /models
    5. Verify team2's key cannot see the model in /models
    """
    client = AsyncClient(base_url=PROXY_BASE_URL)
    headers = {"Authorization": f"Bearer {TEST_MASTER_KEY}"}

    # Create team1
    team1_response = await client.post(
        "/team/new",
        json={"models": []},
        headers=headers,
    )
    assert team1_response.status_code == 200
    team1_id = team1_response.json()["team_id"]

    # Create team2
    team2_response = await client.post(
        "/team/new",
        json={"models": []},
        headers=headers,
    )
    assert team2_response.status_code == 200
    team2_id = team2_response.json()["team_id"]

    # Add model associated with team1
    model_response = await client.post(
        "/model/new",
        json={
            "model_name": "gpt-4-team-test",
            "litellm_params": {
                "model": "gpt-4",
                "custom_llm_provider": "openai",
                "api_key": "fake_key",
            },
            "model_info": {"team_id": team1_id},
        },
        headers=headers,
    )
    assert model_response.status_code == 200

    # Generate keys for both teams
    team1_key = (
        await client.post("/key/generate", json={"team_id": team1_id}, headers=headers)
    ).json()["key"]
    team2_key = (
        await client.post("/key/generate", json={"team_id": team2_id}, headers=headers)
    ).json()["key"]

    # Check models visibility for team1's key
    team1_models = await client.get(
        "/models", headers={"Authorization": f"Bearer {team1_key}"}
    )
    assert team1_models.status_code == 200
    print("team1_models", json.dumps(team1_models.json(), indent=4))
    assert any(
        model["id"] == "gpt-4-team-test" for model in team1_models.json()["data"]
    ), "Team1 should see their model"

    # Check models visibility for team2's key
    team2_models = await client.get(
        "/models", headers={"Authorization": f"Bearer {team2_key}"}
    )
    assert team2_models.status_code == 200
    print("team2_models", json.dumps(team2_models.json(), indent=4))
    assert not any(
        model["id"] == "gpt-4-team-test" for model in team2_models.json()["data"]
    ), "Team2 should not see team1's model"

    # Cleanup
    model_id = model_response.json()["model_info"]["id"]
    await client.post("/model/delete", json={"id": model_id}, headers=headers)


@pytest.mark.asyncio
async def test_team_model_visibility_in_model_info_endpoint():
    """
    Test that team-specific models are visible to all users in /v2/model/info endpoint:
    Note: /v2/model/info is used by the Admin UI to display model info
    1. Create a team
    2. Add a model associated with the team
    3. Generate a team key
    4. Verify both team key and non-team key can see the model in /v2/model/info
    """
    client = AsyncClient(base_url=PROXY_BASE_URL)
    headers = {"Authorization": f"Bearer {TEST_MASTER_KEY}"}

    # Create team
    team_response = await client.post(
        "/team/new",
        json={"models": []},
        headers=headers,
    )
    assert team_response.status_code == 200
    team_id = team_response.json()["team_id"]

    # Add model associated with team
    model_response = await client.post(
        "/model/new",
        json={
            "model_name": "gpt-4-team-test",
            "litellm_params": {
                "model": "gpt-4",
                "custom_llm_provider": "openai",
                "api_key": "fake_key",
            },
            "model_info": {"team_id": team_id},
        },
        headers=headers,
    )
    assert model_response.status_code == 200

    # Generate team key
    team_key = (
        await client.post("/key/generate", json={"team_id": team_id}, headers=headers)
    ).json()["key"]

    # Generate non-team key
    non_team_key = (
        await client.post("/key/generate", json={}, headers=headers)
    ).json()["key"]

    # Check model info visibility with team key
    team_model_info = await client.get(
        "/v2/model/info",
        headers={"Authorization": f"Bearer {team_key}"},
        params={"model_name": "gpt-4-team-test"},
    )
    assert team_model_info.status_code == 200
    team_model_info = team_model_info.json()
    print("Team 1 model info", json.dumps(team_model_info, indent=4))
    assert any(
        model["model_info"].get("team_public_model_name") == "gpt-4-team-test"
        for model in team_model_info["data"]
    ), "Team1 should see their model"

    # Check model info visibility with non-team key
    non_team_model_info = await client.get(
        "/v2/model/info",
        headers={"Authorization": f"Bearer {non_team_key}"},
        params={"model_name": "gpt-4-team-test"},
    )
    assert non_team_model_info.status_code == 200
    non_team_model_info = non_team_model_info.json()
    print("Non-team model info", json.dumps(non_team_model_info, indent=4))
    assert any(
        model["model_info"].get("team_public_model_name") == "gpt-4-team-test"
        for model in non_team_model_info["data"]
    ), "Non-team should see the model"

    # Cleanup
    model_id = model_response.json()["model_info"]["id"]
    await client.post("/model/delete", json={"id": model_id}, headers=headers)
