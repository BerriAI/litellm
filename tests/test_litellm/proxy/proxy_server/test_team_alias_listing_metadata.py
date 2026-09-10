"""Regression coverage for metadata on public team model aliases."""

from unittest.mock import MagicMock

import pytest

import litellm.proxy.proxy_server as ps
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.router import DeploymentModelListingInfo


def _team_router(*, public_name: str, internal_name: str, underlying_model: str, listing_info):
    deployment = {
        "model_name": internal_name,
        "litellm_params": {"model": underlying_model},
        "model_info": {
            "id": "deployment-id",
            "team_id": "teamx",
            "team_public_model_name": public_name,
            "access_groups": ["team-access"],
        },
    }
    router = MagicMock()
    router.get_model_names.return_value = [internal_name]
    router.get_model_access_groups.return_value = {"team-access": [internal_name]}
    router.get_fully_blocked_model_names.return_value = set()
    router.get_model_listing_info.return_value = listing_info
    router.get_model_group_info.return_value = None
    router.model_list = [deployment]
    router.get_model_list.return_value = [deployment]
    return router


@pytest.mark.asyncio
async def test_team_alias_inherits_deployment_token_limits_and_chat_mode(monkeypatch):
    router = _team_router(
        public_name="GPT Terra",
        internal_name="model_name_teamx_terra_uuid",
        underlying_model="azure/gpt-4.1",
        listing_info=DeploymentModelListingInfo(
            cost_map_keys=("azure/gpt-4.1",),
            max_input_tokens=876000,
            max_output_tokens=128000,
        ),
    )
    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(ps, "general_settings", {"use_team_public_model_name": True})

    key = UserAPIKeyAuth(user_id="user", api_key="***", models=["team-access"], team_models=[])
    response = await ps.model_list(user_api_key_dict=key, include_metadata=True)

    assert response["data"] == [
        {
            "id": "GPT Terra",
            "object": "model",
            "created": 1677610602,
            "owned_by": "openai",
            "mode": "chat",
            "max_input_tokens": 876000,
            "max_output_tokens": 128000,
            "metadata": {"fallbacks": []},
        }
    ]


@pytest.mark.asyncio
async def test_team_image_alias_inherits_image_generation_mode(monkeypatch):
    router = _team_router(
        public_name="image",
        internal_name="model_name_teamx_image_uuid",
        underlying_model="openai/gpt-image-1",
        listing_info=DeploymentModelListingInfo(
            cost_map_keys=("openai/gpt-image-1",),
            max_input_tokens=None,
            max_output_tokens=None,
        ),
    )
    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(ps, "general_settings", {"use_team_public_model_name": True})

    key = UserAPIKeyAuth(user_id="user", api_key="***", models=["team-access"], team_models=[])
    response = await ps.model_list(user_api_key_dict=key)

    assert response["data"] == [
        {
            "id": "image",
            "object": "model",
            "created": 1677610602,
            "owned_by": "openai",
            "mode": "image_generation",
        }
    ]
