import json

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from litellm.proxy._types import UserAPIKeyAuth, LitellmUserRoles
from litellm.types.prompts.init_prompts import (
    PromptSpec,
    PromptLiteLLMParams,
    PromptInfo,
)


def _db_row(content: str) -> MagicMock:
    row = MagicMock()
    row.id = "row-1"
    row.version = 1
    row.model_dump.return_value = {
        "prompt_id": "test_prompt",
        "version": 1,
        "environment": "development",
        "created_by": None,
        "litellm_params": {
            "prompt_id": "test_prompt",
            "prompt_integration": "dotprompt",
            "prompt_data": {"content": content, "metadata": {}},
        },
        "prompt_info": {"prompt_type": "db"},
        "created_at": None,
        "updated_at": None,
    }
    return row


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
                expected_base_id, environment=None
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
                expected_base_id, environment=None
            )

            assert response == {
                "message": f"Prompt {expected_base_id} deleted successfully"
            }


@pytest.mark.asyncio
async def test_delete_prompt_environment_scope_reaches_db_and_registry():
    from litellm.proxy.prompts.prompt_endpoints import delete_prompt

    mock_user_auth = UserAPIKeyAuth(api_key="sk-1234", user_role=LitellmUserRoles.PROXY_ADMIN)
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_prompttable.delete_many = AsyncMock(return_value=None)

    with patch(  # test-quality-ok: stubs the collaborator so the test pins what the endpoint deletes
        "litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY"
    ) as mock_registry:
        mock_registry.get_prompt_by_id.return_value = PromptSpec(
            prompt_id="test_prompt.v2",
            litellm_params=PromptLiteLLMParams(prompt_id="test_prompt", prompt_integration="dotprompt"),
            prompt_info=PromptInfo(prompt_type="db"),
        )

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):  # test-quality-ok: proxy_server module global is the endpoint's only injection point
            response = await delete_prompt(
                prompt_id="test_prompt.v2",
                environment="production",
                user_api_key_dict=mock_user_auth,
            )

    mock_prisma_client.db.litellm_prompttable.delete_many.assert_called_once_with(
        where={"prompt_id": "test_prompt", "environment": "production"}
    )
    mock_registry.delete_prompts_by_base_id.assert_called_once_with("test_prompt", environment="production")
    assert response == {"message": "Prompt test_prompt deleted successfully from production"}


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
    # Patch prisma_client to None to avoid leaking state from other tests
    with (
        patch("litellm.proxy.proxy_server.prisma_client", None),
        patch(
            "litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY"
        ) as mock_registry,
    ):
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


@pytest.mark.asyncio
async def test_patch_prompt_row_deleted_mid_update_returns_404():
    """
    A concurrent delete between the version lookup and the write makes Prisma's
    `update` return None. That must reuse the endpoint's existing not-found 404
    contract rather than blowing up into an opaque 500.
    """
    from fastapi import HTTPException

    from litellm.proxy.prompts.prompt_endpoints import PatchPromptRequest, patch_prompt

    mock_user_auth = UserAPIKeyAuth(
        api_key="sk-1234", user_role=LitellmUserRoles.PROXY_ADMIN
    )

    target_row = _db_row("Begin every reply with AHOY")

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_prompttable.find_many = AsyncMock(
        return_value=[target_row]
    )
    mock_prisma_client.db.litellm_prompttable.update = AsyncMock(return_value=None)

    existing_prompt = PromptSpec(
        prompt_id="test_prompt.v1",
        litellm_params=PromptLiteLLMParams(
            prompt_id="test_prompt", prompt_integration="dotprompt"
        ),
        prompt_info=PromptInfo(prompt_type="db"),
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client),  # test-quality-ok: proxy_server module global is the endpoint's only injection point
        patch(  # test-quality-ok: stubs the collaborator so the test pins the endpoint's own error contract
            "litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY"
        ) as mock_registry,
    ):
        mock_registry.get_prompt_by_id.return_value = existing_prompt

        with pytest.raises(HTTPException) as exc_info:
            await patch_prompt(
                prompt_id="test_prompt",
                request=PatchPromptRequest(prompt_info=PromptInfo(prompt_type="db")),
                user_api_key_dict=mock_user_auth,
            )

    assert exc_info.value.status_code == 404
    assert (
        exc_info.value.detail
        == "Prompt with ID test_prompt not found in environment development"
    )


@pytest.mark.asyncio
async def test_patch_prompt_merges_unsent_fields_from_db_row_not_stale_memory():
    from litellm.proxy.prompts.prompt_endpoints import PatchPromptRequest, patch_prompt

    mock_user_auth = UserAPIKeyAuth(api_key="sk-1234", user_role=LitellmUserRoles.PROXY_ADMIN)
    db_row = _db_row("Begin every reply with HOWDY")
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_prompttable.find_many = AsyncMock(return_value=[db_row])
    mock_prisma_client.db.litellm_prompttable.update = AsyncMock(return_value=db_row)
    stale_in_memory = PromptSpec(
        prompt_id="test_prompt.v1",
        litellm_params=PromptLiteLLMParams(
            prompt_id="test_prompt",
            prompt_integration="dotprompt",
            prompt_data={"content": "Begin every reply with AHOY", "metadata": {}},
        ),
        prompt_info=PromptInfo(prompt_type="db"),
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client),  # test-quality-ok: proxy_server module global is the endpoint's only injection point
        patch(  # test-quality-ok: stubs the collaborator so the test pins what the endpoint writes and reloads
            "litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY"
        ) as mock_registry,
    ):
        mock_registry.get_prompt_by_id.return_value = stale_in_memory
        mock_registry.reload_prompt.side_effect = lambda prompt: prompt

        response = await patch_prompt(
            prompt_id="test_prompt",
            request=PatchPromptRequest(prompt_info=PromptInfo(prompt_type="db")),
            user_api_key_dict=mock_user_auth,
        )

    written_params = json.loads(mock_prisma_client.db.litellm_prompttable.update.call_args.kwargs["data"]["litellm_params"])
    assert written_params["prompt_data"]["content"] == "Begin every reply with HOWDY"
    reloaded_spec = mock_registry.reload_prompt.call_args.kwargs["prompt"]
    assert reloaded_spec.prompt_id == "test_prompt.v1"
    assert reloaded_spec.litellm_params.prompt_data["content"] == "Begin every reply with HOWDY"
    assert response.litellm_params.prompt_data["content"] == "Begin every reply with HOWDY"


def test_is_ambiguous_keyed_prompt_data_shapes():
    from litellm.proxy.prompts.prompt_endpoints import is_ambiguous_keyed_prompt_data

    keyed_with_id = PromptLiteLLMParams(
        prompt_id="agent-prompt",
        prompt_integration="dotprompt",
        prompt_data={"json_prompt": {"content": "AHOY", "metadata": {}}},
    )
    flat_with_id = PromptLiteLLMParams(
        prompt_id="agent-prompt",
        prompt_integration="dotprompt",
        prompt_data={"content": "AHOY", "metadata": {}},
    )
    keyed_without_id = PromptLiteLLMParams(
        prompt_integration="dotprompt",
        prompt_data={"json_prompt": {"content": "AHOY", "metadata": {}}},
    )
    no_prompt_data = PromptLiteLLMParams(
        prompt_id="agent-prompt", prompt_integration="dotprompt"
    )
    empty_prompt_data = PromptLiteLLMParams(
        prompt_id="agent-prompt", prompt_integration="dotprompt", prompt_data={}
    )

    assert is_ambiguous_keyed_prompt_data(keyed_with_id) is True
    assert is_ambiguous_keyed_prompt_data(flat_with_id) is False
    assert is_ambiguous_keyed_prompt_data(keyed_without_id) is False
    assert is_ambiguous_keyed_prompt_data(no_prompt_data) is False
    assert is_ambiguous_keyed_prompt_data(empty_prompt_data) is False


@pytest.mark.asyncio
async def test_create_prompt_rejects_keyed_prompt_data_with_prompt_id():
    from fastapi import HTTPException

    from litellm.proxy.prompts.prompt_endpoints import (
        AMBIGUOUS_PROMPT_DATA_ERROR,
        Prompt,
        create_prompt,
    )

    mock_user_auth = UserAPIKeyAuth(
        api_key="sk-1234", user_role=LitellmUserRoles.PROXY_ADMIN
    )
    request = Prompt(
        prompt_id="agent-prompt",
        litellm_params=PromptLiteLLMParams(
            prompt_id="agent-prompt",
            prompt_integration="dotprompt",
            prompt_data={"json_prompt": {"content": "AHOY", "metadata": {}}},
        ),
    )

    with patch("litellm.proxy.proxy_server.prisma_client", MagicMock()):  # test-quality-ok: proxy_server module global is the endpoint's only injection point
        with pytest.raises(HTTPException) as exc_info:
            await create_prompt(request=request, user_api_key_dict=mock_user_auth)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == AMBIGUOUS_PROMPT_DATA_ERROR


@pytest.mark.asyncio
async def test_patch_prompt_rejects_keyed_prompt_data_with_prompt_id():
    from fastapi import HTTPException

    from litellm.proxy.prompts.prompt_endpoints import (
        AMBIGUOUS_PROMPT_DATA_ERROR,
        PatchPromptRequest,
        patch_prompt,
    )

    mock_user_auth = UserAPIKeyAuth(
        api_key="sk-1234", user_role=LitellmUserRoles.PROXY_ADMIN
    )
    request = PatchPromptRequest(
        litellm_params=PromptLiteLLMParams(
            prompt_id="agent-prompt",
            prompt_integration="dotprompt",
            prompt_data={"json_prompt": {"content": "AHOY", "metadata": {}}},
        ),
    )

    with patch("litellm.proxy.proxy_server.prisma_client", MagicMock()):  # test-quality-ok: proxy_server module global is the endpoint's only injection point
        with pytest.raises(HTTPException) as exc_info:
            await patch_prompt(
                prompt_id="agent-prompt",
                request=request,
                user_api_key_dict=mock_user_auth,
            )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == AMBIGUOUS_PROMPT_DATA_ERROR


@pytest.mark.asyncio
async def test_patch_prompt_info_only_keeps_legacy_keyed_row_patchable():
    from litellm.proxy.prompts.prompt_endpoints import PatchPromptRequest, patch_prompt

    mock_user_auth = UserAPIKeyAuth(
        api_key="sk-1234", user_role=LitellmUserRoles.PROXY_ADMIN
    )
    legacy_params = PromptLiteLLMParams(
        prompt_id="agent-prompt",
        prompt_integration="dotprompt",
        prompt_data={"json_prompt": {"content": "AHOY", "metadata": {}}},
    )
    target_row = MagicMock()
    target_row.id = "row-1"
    target_row.version = 1
    target_row.model_dump.return_value = {
        "prompt_id": "agent-prompt",
        "version": 1,
        "environment": "production",
        "created_by": None,
        "litellm_params": legacy_params.model_dump_json(),
        "prompt_info": PromptInfo(prompt_type="db", environment="production").model_dump_json(),
    }
    updated_row = MagicMock()
    updated_row.model_dump.return_value = {
        "prompt_id": "agent-prompt",
        "version": 1,
        "environment": "production",
        "created_by": None,
        "litellm_params": legacy_params.model_dump_json(),
        "prompt_info": PromptInfo(prompt_type="db", environment="production").model_dump_json(),
    }

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_prompttable.find_many = AsyncMock(
        return_value=[target_row]
    )
    mock_prisma_client.db.litellm_prompttable.update = AsyncMock(return_value=updated_row)

    existing_prompt = PromptSpec(
        prompt_id="agent-prompt.v1",
        litellm_params=legacy_params,
        prompt_info=PromptInfo(prompt_type="db"),
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client),  # test-quality-ok: proxy_server module global is the endpoint's only injection point
        patch(  # test-quality-ok: keeps the registry reload from touching global callback state
            "litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY"
        ) as mock_registry,
    ):
        mock_registry.get_prompt_by_id.return_value = existing_prompt

        await patch_prompt(
            prompt_id="agent-prompt",
            request=PatchPromptRequest(prompt_info=PromptInfo(prompt_type="db", environment="production")),
            user_api_key_dict=mock_user_auth,
        )

    update_kwargs = mock_prisma_client.db.litellm_prompttable.update.await_args.kwargs
    assert update_kwargs["where"] == {"id": "row-1"}
    assert json.loads(update_kwargs["data"]["prompt_info"])["environment"] == "production"
    assert json.loads(update_kwargs["data"]["litellm_params"])["prompt_data"] == {
        "json_prompt": {"content": "AHOY", "metadata": {}}
    }


@pytest.mark.asyncio
async def test_update_prompt_rejects_keyed_prompt_data_with_prompt_id():
    from fastapi import HTTPException

    from litellm.proxy.prompts.prompt_endpoints import (
        AMBIGUOUS_PROMPT_DATA_ERROR,
        Prompt,
        update_prompt,
    )

    mock_user_auth = UserAPIKeyAuth(
        api_key="sk-1234", user_role=LitellmUserRoles.PROXY_ADMIN
    )
    request = Prompt(
        prompt_id="agent-prompt",
        litellm_params=PromptLiteLLMParams(
            prompt_id="agent-prompt",
            prompt_integration="dotprompt",
            prompt_data={"json_prompt": {"content": "AHOY", "metadata": {}}},
        ),
    )

    with patch("litellm.proxy.proxy_server.prisma_client", MagicMock()):  # test-quality-ok: proxy_server module global is the endpoint's only injection point
        with pytest.raises(HTTPException) as exc_info:
            await update_prompt(
                prompt_id="agent-prompt",
                request=request,
                user_api_key_dict=mock_user_auth,
            )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == AMBIGUOUS_PROMPT_DATA_ERROR


def test_create_versioned_prompt_spec_populates_version():
    from litellm.proxy.prompts.prompt_endpoints import create_versioned_prompt_spec

    db_prompt = MagicMock()
    db_prompt.model_dump.return_value = {
        "prompt_id": "agent-prompt",
        "version": 3,
        "environment": "development",
        "created_by": "user-1",
        "litellm_params": {
            "prompt_id": "agent-prompt",
            "prompt_integration": "dotprompt",
        },
        "prompt_info": {"prompt_type": "db"},
        "created_at": None,
        "updated_at": None,
    }

    prompt_spec = create_versioned_prompt_spec(db_prompt=db_prompt)

    assert prompt_spec.prompt_id == "agent-prompt.v3"
    assert prompt_spec.version == 3


def test_initialize_prompt_keeps_version_and_created_by():
    import litellm
    from litellm.proxy.prompts.prompt_registry import InMemoryPromptRegistry

    registry = InMemoryPromptRegistry()
    prompt_spec = PromptSpec(
        prompt_id="agent-prompt.v3",
        litellm_params=PromptLiteLLMParams(
            prompt_id="agent-prompt",
            prompt_integration="dotprompt",
            prompt_data={"content": "AHOY", "metadata": {}},
        ),
        prompt_info=PromptInfo(prompt_type="db"),
        version=3,
        environment="development",
        created_by="user-1",
    )

    with patch.object(litellm.logging_callback_manager, "add_litellm_callback"):  # test-quality-ok: keeps initialize_prompt from registering a global callback that would leak across tests
        initialized_prompt = registry.initialize_prompt(prompt=prompt_spec)

    assert initialized_prompt is not None
    assert initialized_prompt.version == 3
    assert initialized_prompt.created_by == "user-1"
    assert initialized_prompt.environment == "development"
