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
                    dotprompt_content="v1 content"
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
            PromptSpec(
                prompt_id="jack.v2",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jack",
                    prompt_integration="dotprompt",
                    dotprompt_content="v2 content"
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
            PromptSpec(
                prompt_id="jane.v1",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jane",
                    prompt_integration="dotprompt",
                    dotprompt_content="jane v1"
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
            PromptSpec(
                prompt_id="jack.v3",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jack",
                    prompt_integration="dotprompt",
                    dotprompt_content="v3 content"
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

    def test_get_latest_version_prompt_id(self):
        """
        Test that get_latest_version_prompt_id returns the highest version
        """
        from litellm.proxy.prompts.prompt_endpoints import get_latest_version_prompt_id

        # Mock prompt IDs dictionary
        all_prompt_ids = {
            "jack.v1": {},
            "jack.v2": {},
            "jack.v3": {},
            "jane.v1": {},
            "simple_prompt": {},
        }

        # Test with base prompt ID - should return latest version
        assert get_latest_version_prompt_id(
            prompt_id="jack",
            all_prompt_ids=all_prompt_ids
        ) == "jack.v3"

        # Test with versioned prompt ID - should still return latest version
        assert get_latest_version_prompt_id(
            prompt_id="jack.v1",
            all_prompt_ids=all_prompt_ids
        ) == "jack.v3"

        # Test with single version
        assert get_latest_version_prompt_id(
            prompt_id="jane",
            all_prompt_ids=all_prompt_ids
        ) == "jane.v1"

        # Test with non-versioned prompt
        assert get_latest_version_prompt_id(
            prompt_id="simple_prompt",
            all_prompt_ids=all_prompt_ids
        ) == "simple_prompt"

        # Test with non-existent prompt
        assert get_latest_version_prompt_id(
            prompt_id="nonexistent",
            all_prompt_ids=all_prompt_ids
        ) == "nonexistent"

    def test_construct_versioned_prompt_id(self):
        """
        Test that construct_versioned_prompt_id correctly builds versioned IDs
        """
        from litellm.proxy.prompts.prompt_endpoints import construct_versioned_prompt_id

        # Test with base prompt ID and version
        assert construct_versioned_prompt_id(
            prompt_id="jack_success",
            version=4
        ) == "jack_success.v4"

        # Test with None version - should return base ID unchanged
        assert construct_versioned_prompt_id(
            prompt_id="jack_success",
            version=None
        ) == "jack_success"

        # Test with existing versioned ID - should replace version
        assert construct_versioned_prompt_id(
            prompt_id="jack_success.v2",
            version=4
        ) == "jack_success.v4"

        # Test with hyphenated prompt ID
        assert construct_versioned_prompt_id(
            prompt_id="my-prompt",
            version=1
        ) == "my-prompt.v1"

        # Test with double-digit version
        assert construct_versioned_prompt_id(
            prompt_id="test_prompt",
            version=10
        ) == "test_prompt.v10"


class TestPromptVersionsEndpoint:
    """
    Test the /prompts/{prompt_id}/versions endpoint
    """

    @pytest.mark.asyncio
    async def test_get_prompt_versions_returns_all_versions(self):
        """
        Test that get_prompt_versions returns all versions of a prompt sorted by version number
        """
        from unittest.mock import MagicMock, patch

        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.prompts.prompt_endpoints import get_prompt_versions

        # Mock user with admin role
        mock_user = UserAPIKeyAuth(
            api_key="test_key",
            user_role=LitellmUserRoles.PROXY_ADMIN
        )

        # Create mock prompt registry with multiple versions
        mock_prompts = {
            "jack.v1": PromptSpec(
                prompt_id="jack.v1",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jack",
                    prompt_integration="dotprompt",
                    dotprompt_content="v1"
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
            "jack.v2": PromptSpec(
                prompt_id="jack.v2",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jack",
                    prompt_integration="dotprompt",
                    dotprompt_content="v2"
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
            "jack.v3": PromptSpec(
                prompt_id="jack.v3",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jack",
                    prompt_integration="dotprompt",
                    dotprompt_content="v3"
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
            "jane.v1": PromptSpec(
                prompt_id="jane.v1",
                litellm_params=PromptLiteLLMParams(
                    prompt_id="jane",
                    prompt_integration="dotprompt",
                    dotprompt_content="jane"
                ),
                prompt_info=PromptInfo(prompt_type="db"),
            ),
        }

        # Mock the IN_MEMORY_PROMPT_REGISTRY at the import location
        with patch("litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY") as mock_registry:
            mock_registry.IN_MEMORY_PROMPTS = mock_prompts

            # Test with base prompt ID
            response = await get_prompt_versions(
                prompt_id="jack",
                user_api_key_dict=mock_user
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
                prompt_id="jack.v1",
                user_api_key_dict=mock_user
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
            api_key="test_key",
            user_role=LitellmUserRoles.PROXY_ADMIN
        )

        with patch("litellm.proxy.prompts.prompt_registry.IN_MEMORY_PROMPT_REGISTRY") as mock_registry:
            mock_registry.IN_MEMORY_PROMPTS = {}

            with pytest.raises(HTTPException) as exc_info:
                await get_prompt_versions(
                    prompt_id="nonexistent",
                    user_api_key_dict=mock_user
                )

            assert exc_info.value.status_code == 404
            assert "No versions found" in exc_info.value.detail

