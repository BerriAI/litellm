import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from litellm.proxy._types import UserAPIKeyAuth, LitellmUserRoles
from litellm.types.prompts.init_prompts import (
    PromptSpec,
    PromptLiteLLMParams,
    PromptInfo,
)


@pytest.mark.asyncio
async def test_delete_prompt_success():
    """
    Test that delete_prompt correctly identifies the base prompt ID
    and deletes all versions from DB and memory.
    """
    from litellm.proxy.prompts.prompt_endpoints import delete_prompt

    # Mock user auth
    mock_user_auth = UserAPIKeyAuth(
        api_key="sk-1234", user_role=LitellmUserRoles.PROXY_ADMIN
    )

    # Mock DB Client
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_prompttable.delete_many = AsyncMock(return_value=None)

    # Mock In-Memory Registry
    with patch(
        "litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY"
    ) as mock_registry:
        # User passes "test_prompt.v2"
        # We simulate that get_prompt_by_id returns the prompt spec for v2
        prompt_spec = PromptSpec(
            prompt_id="test_prompt.v2",
            litellm_params=PromptLiteLLMParams(
                prompt_id="test_prompt", prompt_integration="dotprompt"
            ),
            prompt_info=PromptInfo(prompt_type="db"),
        )
        mock_registry.get_prompt_by_id.return_value = prompt_spec

        # Patch the prisma client in the endpoint module
        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
            response = await delete_prompt(
                prompt_id="test_prompt.v2", user_api_key_dict=mock_user_auth
            )

            # Assertions
            expected_base_id = "test_prompt"

            # 1. DB deletion should use base ID
            mock_prisma_client.db.litellm_prompttable.delete_many.assert_called_once_with(
                where={"prompt_id": expected_base_id}
            )

            # 2. Memory deletion should use base ID
            mock_registry.delete_prompts_by_base_id.assert_called_once_with(
                expected_base_id
            )

            assert response == {
                "message": f"Prompt {expected_base_id} deleted successfully"
            }


@pytest.mark.asyncio
async def test_delete_prompt_by_base_id_success():
    """
    Test that delete_prompt works when passed a base ID directly,
    finding the latest version to confirm existence, then deleting.
    """
    from litellm.proxy.prompts.prompt_endpoints import delete_prompt

    # Mock user auth
    mock_user_auth = UserAPIKeyAuth(
        api_key="sk-1234", user_role=LitellmUserRoles.PROXY_ADMIN
    )

    # Mock DB Client
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_prompttable.delete_many = AsyncMock(return_value=None)

    # Mock In-Memory Registry
    with patch(
        "litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY"
    ) as mock_registry:
        # User passes "test_prompt" (base ID)
        # 1. get_prompt_by_id("test_prompt") -> None (if it's not registered as base)
        # 2. It calls get_latest_version_prompt_id -> returns "test_prompt.v3"
        # 3. get_prompt_by_id("test_prompt.v3") -> returns Spec

        # Setup mocks behavior
        def get_prompt_side_effect(prompt_id):
            if prompt_id == "test_prompt":
                return None
            if prompt_id == "test_prompt.v3":
                return PromptSpec(
                    prompt_id="test_prompt.v3",
                    litellm_params=PromptLiteLLMParams(
                        prompt_id="test_prompt", prompt_integration="dotprompt"
                    ),
                    prompt_info=PromptInfo(prompt_type="db"),
                )
            return None

        mock_registry.get_prompt_by_id.side_effect = get_prompt_side_effect
        mock_registry.IN_MEMORY_PROMPTS = {
            "test_prompt.v1": {},
            "test_prompt.v2": {},
            "test_prompt.v3": {},
        }

        # Patch the prisma client in the endpoint module
        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
            response = await delete_prompt(
                prompt_id="test_prompt", user_api_key_dict=mock_user_auth
            )

            # Assertions
            expected_base_id = "test_prompt"

            # 1. DB deletion should use base ID
            mock_prisma_client.db.litellm_prompttable.delete_many.assert_called_once_with(
                where={"prompt_id": expected_base_id}
            )

            # 2. Memory deletion should use base ID
            mock_registry.delete_prompts_by_base_id.assert_called_once_with(
                expected_base_id
            )

            assert response == {
                "message": f"Prompt {expected_base_id} deleted successfully"
            }


@pytest.mark.asyncio
async def test_get_prompt_info_by_base_id():
    """
    Test that get_prompt_info correctly resolves a base ID to the latest version.
    """
    from litellm.proxy.prompts.prompt_endpoints import get_prompt_info

    # Mock user auth
    mock_user_auth = UserAPIKeyAuth(
        api_key="sk-1234", user_role=LitellmUserRoles.PROXY_ADMIN
    )

    # Mock In-Memory Registry
    with patch(
        "litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY"
    ) as mock_registry:
        # Setup mocks behavior
        prompt_spec_v3 = PromptSpec(
            prompt_id="test_prompt.v3",
            litellm_params=PromptLiteLLMParams(
                prompt_id="test_prompt", prompt_integration="dotprompt"
            ),
            prompt_info=PromptInfo(prompt_type="db"),
        )

        # When get_prompt_by_id is called with "test_prompt", return None (so it searches versions)
        # When called with "test_prompt.v3", return the spec
        def get_prompt_side_effect(prompt_id):
            if prompt_id == "test_prompt":
                return None
            if prompt_id == "test_prompt.v3":
                return prompt_spec_v3
            return None

        mock_registry.get_prompt_by_id.side_effect = get_prompt_side_effect
        mock_registry.IN_MEMORY_PROMPTS = {
            "test_prompt.v1": {},
            "test_prompt.v2": {},
            "test_prompt.v3": {},
        }

        # We also need to mock get_prompt_callback_by_id to avoid content extraction errors/logic
        mock_registry.get_prompt_callback_by_id.return_value = None

        response = await get_prompt_info(
            prompt_id="test_prompt", user_api_key_dict=mock_user_auth
        )

        assert (
            response.prompt_spec.prompt_id == "test_prompt"
        )  # Should return base ID in spec response
        assert response.prompt_spec.version == 3  # Should identify it as version 3
