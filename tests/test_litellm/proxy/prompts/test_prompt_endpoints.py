"""
Test prompt endpoints for version filtering and history
"""

from unittest.mock import MagicMock

import pytest

from litellm.types.prompts.init_prompts import (
    PromptInfo,
    PromptLiteLLMParams,
    PromptSpec,
)


class TestPromptVersioning:
    """
    Test prompt versioning functionality
    """

    def test_get_latest_prompt_versions(self):
        """
        Test that get_latest_prompt_versions returns only the latest version of each prompt
        """
        from litellm.proxy.prompts.prompt_endpoints import get_latest_prompt_versions

        # Create mock prompts with different versions
        prompts = [
            PromptSpec(
                prompt_id="jack.v1",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jack",
                    prompt_integration="dotprompt",
                    dotprompt_content="v1 content",
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
            PromptSpec(
                prompt_id="jack.v2",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jack",
                    prompt_integration="dotprompt",
                    dotprompt_content="v2 content",
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
            PromptSpec(
                prompt_id="jane.v1",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jane",
                    prompt_integration="dotprompt",
                    dotprompt_content="jane v1",
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
            PromptSpec(
                prompt_id="jack.v3",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jack",
                    prompt_integration="dotprompt",
                    dotprompt_content="v3 content",
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
        ]

        # Get latest versions
        latest = get_latest_prompt_versions(prompts=prompts)

        # Should return 2 prompts (jack.v3 and jane.v1)
        assert len(latest) == 2

        # Find jack and jane in results
        jack_prompt = next((p for p in latest if "jack" in p.prompt_id), None)
        jane_prompt = next((p for p in latest if "jane" in p.prompt_id), None)

        assert jack_prompt is not None
        assert jack_prompt.prompt_id == "jack.v3"
        assert jack_prompt.litellm_params.dotprompt_content == "v3 content"

        assert jane_prompt is not None
        assert jane_prompt.prompt_id == "jane.v1"

    def test_get_version_number(self):
        """
        Test that get_version_number correctly extracts version numbers
        """
        from litellm.proxy.prompts.prompt_endpoints import get_version_number

        assert get_version_number(prompt_id="jack.v1") == 1
        assert get_version_number(prompt_id="jack.v2") == 2
        assert get_version_number(prompt_id="jack.v10") == 10
        assert get_version_number(prompt_id="jack") == 1
        assert get_version_number(prompt_id="jack.vinvalid") == 1

    def test_get_base_prompt_id(self):
        """
        Test that get_base_prompt_id correctly strips version suffixes
        """
        from litellm.proxy.prompts.prompt_endpoints import get_base_prompt_id

        assert get_base_prompt_id(prompt_id="jack.v1") == "jack"
        assert get_base_prompt_id(prompt_id="jack.v2") == "jack"
        assert get_base_prompt_id(prompt_id="jack") == "jack"
        assert get_base_prompt_id(prompt_id="my_prompt.v10") == "my_prompt"


class TestPromptVersionsEndpoint:
    """
    Test the /prompts/{prompt_id}/versions endpoint
    """

    @pytest.mark.asyncio
    async def test_get_prompt_versions_returns_all_versions(self):
        """
        Test that get_prompt_versions returns all versions of a prompt sorted by version number
        """
        from unittest.mock import patch

        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.prompts.prompt_endpoints import get_prompt_versions

        # Mock user with admin role
        mock_user = UserAPIKeyAuth(
            api_key="test_key", user_role=LitellmUserRoles.PROXY_ADMIN
        )

        # Create mock prompt registry with multiple versions
        mock_prompts = {
            "jack.v1": PromptSpec(
                prompt_id="jack.v1",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jack",
                    prompt_integration="dotprompt",
                    dotprompt_content="v1",
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
            "jack.v2": PromptSpec(
                prompt_id="jack.v2",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jack",
                    prompt_integration="dotprompt",
                    dotprompt_content="v2",
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
            "jack.v3": PromptSpec(
                prompt_id="jack.v3",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jack",
                    prompt_integration="dotprompt",
                    dotprompt_content="v3",
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
            "jane.v1": PromptSpec(
                prompt_id="jane.v1",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jane",
                    prompt_integration="dotprompt",
                    dotprompt_content="jane",
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
        }

        # Force the in-memory path so this test is isolated from any leaked prisma mocks.
        with (
            patch("litellm.proxy.proxy_server.prisma_client", None),
            patch(
                "litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY"
            ) as mock_registry,
        ):
            mock_registry.IN_MEMORY_PROMPTS = mock_prompts

            # Test with base prompt ID
            response = await get_prompt_versions(
                prompt_id="jack", user_api_key_dict=mock_user
            )

            # Should return 3 versions of jack, sorted newest first
            assert len(response.prompts) == 3
            assert response.prompts[0].prompt_id == "jack"
            assert response.prompts[0].version == 3
            assert response.prompts[1].prompt_id == "jack"
            assert response.prompts[1].version == 2
            assert response.prompts[2].prompt_id == "jack"
            assert response.prompts[2].version == 1

            # Test with versioned prompt ID (should strip version)
            response = await get_prompt_versions(
                prompt_id="jack.v1", user_api_key_dict=mock_user
            )

            assert len(response.prompts) == 3
            assert response.prompts[0].prompt_id == "jack"
            assert response.prompts[0].version == 3

    @pytest.mark.asyncio
    async def test_get_prompt_versions_not_found(self):
        """
        Test that get_prompt_versions raises 404 when prompt doesn't exist
        """
        from unittest.mock import patch

        from fastapi import HTTPException

        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.prompts.prompt_endpoints import get_prompt_versions

        mock_user = UserAPIKeyAuth(
            api_key="test_key", user_role=LitellmUserRoles.PROXY_ADMIN
        )

        with (
            patch("litellm.proxy.proxy_server.prisma_client", None),
            patch(
                "litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY"
            ) as mock_registry,
        ):
            mock_registry.IN_MEMORY_PROMPTS = {}

            with pytest.raises(HTTPException) as exc_info:
                await get_prompt_versions(
                    prompt_id="nonexistent", user_api_key_dict=mock_user
                )

            assert exc_info.value.status_code == 404
            assert "No versions found" in exc_info.value.detail


class TestAdminViewerReadAccess:
    """
    proxy_admin_viewer has READ parity with proxy_admin on the prompt read endpoints
    """

    @pytest.mark.asyncio
    async def test_list_prompts_returns_all_prompts_for_admin_viewer(self):
        """A role without admin view falls through to the empty-list branch here."""
        from unittest.mock import patch

        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.prompts.prompt_endpoints import list_prompts

        viewer = UserAPIKeyAuth(
            api_key="test_key", user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
        )

        mock_prompts = {
            "jack.v1": PromptSpec(
                prompt_id="jack.v1",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jack",
                    prompt_integration="dotprompt",
                    dotprompt_content="v1",
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
            "jack.v2": PromptSpec(
                prompt_id="jack.v2",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jack",
                    prompt_integration="dotprompt",
                    dotprompt_content="v2",
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
            "jane.v1": PromptSpec(
                prompt_id="jane.v1",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jane",
                    prompt_integration="dotprompt",
                    dotprompt_content="jane",
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
        }

        with patch(
            "litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY"
        ) as mock_registry:
            mock_registry.IN_MEMORY_PROMPTS = mock_prompts

            response = await list_prompts(user_api_key_dict=viewer)

        assert sorted(p.prompt_id for p in response.prompts) == ["jack", "jane"]
        jack = next(p for p in response.prompts if p.prompt_id == "jack")
        assert jack.litellm_params.dotprompt_content == "v2"

    @pytest.mark.asyncio
    async def test_get_prompt_versions_allows_admin_viewer(self):
        """Version history used to 403 anyone who was not exactly proxy_admin."""
        from unittest.mock import patch

        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.prompts.prompt_endpoints import get_prompt_versions

        viewer = UserAPIKeyAuth(
            api_key="test_key", user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
        )

        mock_prompts = {
            "jack.v1": PromptSpec(
                prompt_id="jack.v1",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jack",
                    prompt_integration="dotprompt",
                    dotprompt_content="v1",
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
            "jack.v2": PromptSpec(
                prompt_id="jack.v2",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jack",
                    prompt_integration="dotprompt",
                    dotprompt_content="v2",
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
        }

        with (
            patch("litellm.proxy.proxy_server.prisma_client", None),
            patch(
                "litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY"
            ) as mock_registry,
        ):
            mock_registry.IN_MEMORY_PROMPTS = mock_prompts

            response = await get_prompt_versions(
                prompt_id="jack", user_api_key_dict=viewer
            )

        assert [p.version for p in response.prompts] == [2, 1]

    @pytest.mark.asyncio
    async def test_get_prompt_info_allows_admin_viewer(self):
        """Prompt info used to 403 anyone who was not exactly proxy_admin."""
        from unittest.mock import patch

        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.prompts.prompt_endpoints import get_prompt_info

        viewer = UserAPIKeyAuth(
            api_key="test_key", user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
        )

        with (
            patch("litellm.proxy.proxy_server.prisma_client", None),
            patch(
                "litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY"
            ) as mock_registry,
        ):
            mock_registry.resolve_prompt_spec.return_value = PromptSpec(
                prompt_id="jack.v2",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jack",
                    prompt_integration="dotprompt",
                    dotprompt_content="v2",
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            )
            mock_registry.get_prompt_callback_for_prompt.return_value = None

            response = await get_prompt_info(prompt_id="jack", user_api_key_dict=viewer)

        assert response.prompt_spec.prompt_id == "jack"
        assert response.prompt_spec.version == 2
