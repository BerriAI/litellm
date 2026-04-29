"""
Tests for Skills mode switching (litellm vs passthrough).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestGetSkillsMode:
    """Tests for get_skills_mode function."""

    def test_default_mode_is_passthrough(self):
        """Test that default skills_mode is passthrough."""
        with patch.dict("litellm.proxy.proxy_server.general_settings", {}, clear=True):
            from litellm.proxy.anthropic_endpoints.skills_endpoints import (
                get_skills_mode,
            )

            mode = get_skills_mode()
            assert mode == "passthrough"

    def test_litellm_mode(self):
        """Test skills_mode='litellm' is recognized."""
        with patch.dict(
            "litellm.proxy.proxy_server.general_settings",
            {"skills_mode": "litellm"},
            clear=True,
        ):
            from litellm.proxy.anthropic_endpoints.skills_endpoints import (
                get_skills_mode,
            )

            mode = get_skills_mode()
            assert mode == "litellm"

    def test_passthrough_mode_explicit(self):
        """Test explicit skills_mode='passthrough' is recognized."""
        with patch.dict(
            "litellm.proxy.proxy_server.general_settings",
            {"skills_mode": "passthrough"},
            clear=True,
        ):
            from litellm.proxy.anthropic_endpoints.skills_endpoints import (
                get_skills_mode,
            )

            mode = get_skills_mode()
            assert mode == "passthrough"

    def test_invalid_mode_defaults_to_passthrough(self):
        """Test that invalid skills_mode defaults to passthrough."""
        with patch.dict(
            "litellm.proxy.proxy_server.general_settings",
            {"skills_mode": "invalid_mode"},
            clear=True,
        ):
            from litellm.proxy.anthropic_endpoints.skills_endpoints import (
                get_skills_mode,
            )

            mode = get_skills_mode()
            assert mode == "passthrough"

    def test_none_mode_defaults_to_passthrough(self):
        """Test that skills_mode=None defaults to passthrough."""
        with patch.dict(
            "litellm.proxy.proxy_server.general_settings",
            {"skills_mode": None},
            clear=True,
        ):
            from litellm.proxy.anthropic_endpoints.skills_endpoints import (
                get_skills_mode,
            )

            mode = get_skills_mode()
            assert mode == "passthrough"


class TestSkillsEndpointsModeRouting:
    """Tests for endpoint routing based on skills_mode."""

    @pytest.mark.asyncio
    async def test_create_skill_litellm_mode_routes_to_handler(self):
        """Test that create_skill in litellm mode calls _handle_litellm_create_skill."""
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.types.llms.anthropic_skills import Skill

        mock_skill = Skill(
            id="litellm_skill_test123",
            display_title="Test Skill",
            source="litellm",
            created_at="2026-03-21T00:00:00Z",
            updated_at="2026-03-21T00:00:00Z",
        )

        with patch.dict(
            "litellm.proxy.proxy_server.general_settings",
            {"skills_mode": "litellm"},
            clear=True,
        ):
            with patch(
                "litellm.proxy.anthropic_endpoints.skills_endpoints._handle_litellm_create_skill",
                new_callable=AsyncMock,
                return_value=mock_skill,
            ) as mock_handler:
                from litellm.proxy.anthropic_endpoints.skills_endpoints import (
                    create_skill,
                )

                mock_request = MagicMock()
                mock_response = MagicMock()
                mock_user = UserAPIKeyAuth(api_key="test-key")

                result = await create_skill(
                    fastapi_response=mock_response,
                    request=mock_request,
                    user_api_key_dict=mock_user,
                )

                mock_handler.assert_called_once()
                assert result == mock_skill

    @pytest.mark.asyncio
    async def test_list_skills_litellm_mode_routes_to_handler(self):
        """Test that list_skills in litellm mode calls _handle_litellm_list_skills."""
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.types.llms.anthropic_skills import ListSkillsResponse

        mock_response = ListSkillsResponse(data=[], has_more=False)

        with patch.dict(
            "litellm.proxy.proxy_server.general_settings",
            {"skills_mode": "litellm"},
            clear=True,
        ):
            with patch(
                "litellm.proxy.anthropic_endpoints.skills_endpoints._handle_litellm_list_skills",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_handler:
                from litellm.proxy.anthropic_endpoints.skills_endpoints import (
                    list_skills,
                )

                mock_request = MagicMock()
                mock_fastapi_response = MagicMock()
                mock_user = UserAPIKeyAuth(api_key="test-key")

                result = await list_skills(
                    fastapi_response=mock_fastapi_response,
                    request=mock_request,
                    limit=20,
                    user_api_key_dict=mock_user,
                )

                mock_handler.assert_called_once_with(limit=20, page=None)
                assert result == mock_response

    @pytest.mark.asyncio
    async def test_get_skill_litellm_mode_routes_to_handler(self):
        """Test that get_skill in litellm mode calls _handle_litellm_get_skill."""
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.types.llms.anthropic_skills import Skill

        mock_skill = Skill(
            id="litellm_skill_123",
            display_title="Test",
            source="litellm",
            created_at="2026-03-21T00:00:00Z",
            updated_at="2026-03-21T00:00:00Z",
        )

        with patch.dict(
            "litellm.proxy.proxy_server.general_settings",
            {"skills_mode": "litellm"},
            clear=True,
        ):
            with patch(
                "litellm.proxy.anthropic_endpoints.skills_endpoints._handle_litellm_get_skill",
                new_callable=AsyncMock,
                return_value=mock_skill,
            ) as mock_handler:
                from litellm.proxy.anthropic_endpoints.skills_endpoints import (
                    get_skill,
                )

                mock_request = MagicMock()
                mock_response = MagicMock()
                mock_user = UserAPIKeyAuth(api_key="test-key")

                result = await get_skill(
                    skill_id="litellm_skill_123",
                    fastapi_response=mock_response,
                    request=mock_request,
                    user_api_key_dict=mock_user,
                )

                mock_handler.assert_called_once_with("litellm_skill_123")
                assert result == mock_skill

    @pytest.mark.asyncio
    async def test_delete_skill_litellm_mode_routes_to_handler(self):
        """Test that delete_skill in litellm mode calls _handle_litellm_delete_skill."""
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.types.llms.anthropic_skills import DeleteSkillResponse

        mock_response = DeleteSkillResponse(id="litellm_skill_123", type="skill_deleted")

        with patch.dict(
            "litellm.proxy.proxy_server.general_settings",
            {"skills_mode": "litellm"},
            clear=True,
        ):
            with patch(
                "litellm.proxy.anthropic_endpoints.skills_endpoints._handle_litellm_delete_skill",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_handler:
                from litellm.proxy.anthropic_endpoints.skills_endpoints import (
                    delete_skill,
                )

                mock_request = MagicMock()
                mock_fastapi_response = MagicMock()
                mock_user = UserAPIKeyAuth(api_key="test-key")

                result = await delete_skill(
                    skill_id="litellm_skill_123",
                    fastapi_response=mock_fastapi_response,
                    request=mock_request,
                    user_api_key_dict=mock_user,
                )

                mock_handler.assert_called_once_with("litellm_skill_123")
                assert result == mock_response
