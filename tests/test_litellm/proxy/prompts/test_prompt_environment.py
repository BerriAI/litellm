import json
import pytest
from unittest.mock import MagicMock
from litellm.types.prompts.init_prompts import (
    PromptInfo,
    PromptSpec,
    PromptLiteLLMParams,
)


def test_prompt_info_default_environment():
    """PromptInfo should default environment to 'development'."""
    info = PromptInfo(prompt_type="db")
    assert info.environment == "development"


def test_prompt_info_custom_environment():
    """PromptInfo should accept a custom environment."""
    info = PromptInfo(prompt_type="db", environment="production")
    assert info.environment == "production"


def test_prompt_spec_includes_environment_and_created_by():
    """PromptSpec should carry environment and created_by fields."""
    spec = PromptSpec(
        prompt_id="test",
        litellm_params=PromptLiteLLMParams(
            prompt_id="test", prompt_integration="dotprompt"
        ),
        prompt_info=PromptInfo(prompt_type="db", environment="staging"),
        environment="staging",
        created_by="user-123",
    )
    assert spec.environment == "staging"
    assert spec.created_by == "user-123"


def test_prompt_spec_default_environment():
    """PromptSpec environment should default to 'development'."""
    spec = PromptSpec(
        prompt_id="test",
        litellm_params=PromptLiteLLMParams(
            prompt_id="test", prompt_integration="dotprompt"
        ),
        prompt_info=PromptInfo(prompt_type="db"),
    )
    assert spec.environment == "development"
    assert spec.created_by is None


def test_create_versioned_prompt_spec_includes_environment():
    """create_versioned_prompt_spec should populate environment and created_by from DB row."""
    from litellm.proxy.prompts.prompt_endpoints import create_versioned_prompt_spec

    mock_db_prompt = MagicMock()
    mock_db_prompt.model_dump.return_value = {
        "id": "uuid-123",
        "prompt_id": "test_prompt",
        "version": 2,
        "environment": "staging",
        "created_by": "user-456",
        "litellm_params": json.dumps(
            {
                "prompt_id": "test_prompt",
                "prompt_integration": "dotprompt",
            }
        ),
        "prompt_info": json.dumps({"prompt_type": "db", "environment": "staging"}),
        "created_at": None,
        "updated_at": None,
    }
    spec = create_versioned_prompt_spec(mock_db_prompt)
    assert spec.environment == "staging"
    assert spec.created_by == "user-456"
    assert spec.prompt_id == "test_prompt.v2"


@pytest.mark.asyncio
async def test_create_prompt_stores_environment_and_created_by():
    """create_prompt should pass environment and created_by to the DB."""
    from unittest.mock import AsyncMock, patch
    from litellm.proxy._types import UserAPIKeyAuth, LitellmUserRoles
    from litellm.proxy.prompts.prompt_endpoints import create_prompt, Prompt

    mock_user_auth = UserAPIKeyAuth(
        api_key="sk-1234",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="user-789",
    )

    mock_prisma_client = MagicMock()
    mock_db_entry = MagicMock()
    mock_db_entry.model_dump.return_value = {
        "id": "uuid-1",
        "prompt_id": "my_prompt",
        "version": 1,
        "environment": "staging",
        "created_by": "user-789",
        "litellm_params": json.dumps(
            {
                "prompt_id": "my_prompt",
                "prompt_integration": "dotprompt",
            }
        ),
        "prompt_info": json.dumps({"prompt_type": "db", "environment": "staging"}),
        "created_at": None,
        "updated_at": None,
    }
    mock_prisma_client.db.litellm_prompttable.create = AsyncMock(
        return_value=mock_db_entry
    )
    mock_prisma_client.db.litellm_prompttable.find_many = AsyncMock(return_value=[])

    request = Prompt(
        prompt_id="my_prompt",
        litellm_params=PromptLiteLLMParams(
            prompt_id="my_prompt", prompt_integration="dotprompt"
        ),
        prompt_info=PromptInfo(prompt_type="db", environment="staging"),
    )

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
        with patch(
            "litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY"
        ) as mock_registry:
            mock_registry.initialize_prompt.return_value = PromptSpec(
                prompt_id="my_prompt.v1",
                litellm_params=request.litellm_params,
                prompt_info=request.prompt_info,
                environment="staging",
                created_by="user-789",
            )
            await create_prompt(request=request, user_api_key_dict=mock_user_auth)

            create_call = mock_prisma_client.db.litellm_prompttable.create.call_args
            data = create_call.kwargs["data"]
            assert data["environment"] == "staging"
            assert data["created_by"] == "user-789"


@pytest.mark.asyncio
async def test_update_prompt_stores_environment_and_created_by():
    """update_prompt should pass environment and created_by to new version."""
    from unittest.mock import AsyncMock, patch
    from litellm.proxy._types import UserAPIKeyAuth, LitellmUserRoles
    from litellm.proxy.prompts.prompt_endpoints import update_prompt, Prompt

    mock_user_auth = UserAPIKeyAuth(
        api_key="sk-1234",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="user-update",
    )

    mock_prisma_client = MagicMock()
    mock_existing = MagicMock()
    mock_existing.version = 1
    mock_prisma_client.db.litellm_prompttable.find_many = AsyncMock(
        return_value=[mock_existing]
    )

    mock_db_entry = MagicMock()
    mock_db_entry.model_dump.return_value = {
        "id": "uuid-2",
        "prompt_id": "my_prompt",
        "version": 2,
        "environment": "production",
        "created_by": "user-update",
        "litellm_params": json.dumps(
            {
                "prompt_id": "my_prompt",
                "prompt_integration": "dotprompt",
            }
        ),
        "prompt_info": json.dumps({"prompt_type": "db", "environment": "production"}),
        "created_at": None,
        "updated_at": None,
    }
    mock_prisma_client.db.litellm_prompttable.create = AsyncMock(
        return_value=mock_db_entry
    )

    request = Prompt(
        prompt_id="my_prompt",
        litellm_params=PromptLiteLLMParams(
            prompt_id="my_prompt", prompt_integration="dotprompt"
        ),
        prompt_info=PromptInfo(prompt_type="db", environment="production"),
    )

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
        with patch(
            "litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY"
        ) as mock_registry:
            mock_registry.get_prompt_by_id.return_value = PromptSpec(
                prompt_id="my_prompt.v1",
                litellm_params=request.litellm_params,
                prompt_info=PromptInfo(prompt_type="db"),
            )
            mock_registry.initialize_prompt.return_value = PromptSpec(
                prompt_id="my_prompt.v2",
                litellm_params=request.litellm_params,
                prompt_info=request.prompt_info,
                environment="production",
                created_by="user-update",
            )
            await update_prompt(
                prompt_id="my_prompt", request=request, user_api_key_dict=mock_user_auth
            )

            create_call = mock_prisma_client.db.litellm_prompttable.create.call_args
            data = create_call.kwargs["data"]
            assert data["environment"] == "production"
            assert data["created_by"] == "user-update"


@pytest.mark.asyncio
async def test_delete_prompt_scoped_to_environment():
    """delete_prompt with environment param should scope deletion."""
    from unittest.mock import AsyncMock, patch
    from litellm.proxy._types import UserAPIKeyAuth, LitellmUserRoles
    from litellm.proxy.prompts.prompt_endpoints import delete_prompt

    mock_user_auth = UserAPIKeyAuth(
        api_key="sk-1234",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_prompttable.delete_many = AsyncMock(return_value=None)

    with patch(
        "litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY"
    ) as mock_registry:
        prompt_spec = PromptSpec(
            prompt_id="test_prompt.v1",
            litellm_params=PromptLiteLLMParams(
                prompt_id="test_prompt", prompt_integration="dotprompt"
            ),
            prompt_info=PromptInfo(prompt_type="db"),
            environment="staging",
        )
        mock_registry.get_prompt_by_id.return_value = prompt_spec

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
            await delete_prompt(
                prompt_id="test_prompt",
                user_api_key_dict=mock_user_auth,
                environment="staging",
            )

            mock_prisma_client.db.litellm_prompttable.delete_many.assert_called_once_with(
                where={"prompt_id": "test_prompt", "environment": "staging"}
            )
