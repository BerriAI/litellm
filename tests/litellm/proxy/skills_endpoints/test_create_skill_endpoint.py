"""
Tests for the create skill endpoint form data parsing.

Simulates actual curl/multipart uploads to verify the endpoint
correctly handles file uploads in litellm mode.
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from litellm.proxy._types import LiteLLM_SkillsTable, UserAPIKeyAuth


def _make_upload_file(filename: str, content: bytes) -> UploadFile:
    """Create a FastAPI UploadFile matching what multipart form parsing produces."""
    return UploadFile(
        filename=filename,
        file=io.BytesIO(content),
        headers=Headers({"content-type": "application/octet-stream"}),
    )


def _make_db_skill(**overrides) -> LiteLLM_SkillsTable:
    """Create a mock DB skill record."""
    from datetime import datetime

    defaults = {
        "skill_id": "litellm_skill_test123",
        "display_title": "Test Skill",
        "description": None,
        "instructions": "Test instructions",
        "source": "custom",
        "file_content": b"fake-zip",
        "file_name": "skill.zip",
        "file_type": "application/zip",
        "created_at": datetime(2026, 3, 21),
        "updated_at": datetime(2026, 3, 21),
    }
    defaults.update(overrides)
    return LiteLLM_SkillsTable(**defaults)


SKILL_MD_CONTENT = b"""---
name: test-skill
description: A test skill
---

Test instructions here.
"""


class TestCreateSkillFormParsing:
    """Tests that simulate actual curl multipart uploads."""

    @pytest.mark.asyncio
    async def test_files_bracket_key_single_upload(self):
        """
        Simulate: curl -F "files[]=@SKILL.md;filename=skill/SKILL.md"

        get_form_data strips [] so key becomes "files" with value in a list.
        """
        from litellm.proxy.anthropic_endpoints.skills_endpoints import (
            _handle_litellm_create_skill,
        )

        upload = _make_upload_file("skill/SKILL.md", SKILL_MD_CONTENT)

        mock_request = MagicMock()
        # get_form_data strips [] and appends to list
        mock_request.form = AsyncMock(return_value={"files[]": upload})

        mock_db_skill = _make_db_skill(display_title="test-skill")

        with patch(
            "litellm.proxy.anthropic_endpoints.skills_endpoints.get_form_data",
            new_callable=AsyncMock,
            return_value={"files": [upload], "display_title": "Test Skill"},
        ):
            with patch(
                "litellm.llms.litellm_proxy.skills.handler.LiteLLMSkillsHandler.create_skill",
                new_callable=AsyncMock,
                return_value=mock_db_skill,
            ):
                user = UserAPIKeyAuth(api_key="test-key")
                result = await _handle_litellm_create_skill(mock_request, user)

                assert result.id == "litellm_skill_test123"

    @pytest.mark.asyncio
    async def test_files_key_without_brackets(self):
        """
        Test that files under plain "files" key also works.
        """
        from litellm.proxy.anthropic_endpoints.skills_endpoints import (
            _handle_litellm_create_skill,
        )

        upload = _make_upload_file("skill/SKILL.md", SKILL_MD_CONTENT)
        mock_request = MagicMock()

        mock_db_skill = _make_db_skill()

        with patch(
            "litellm.proxy.anthropic_endpoints.skills_endpoints.get_form_data",
            new_callable=AsyncMock,
            return_value={"files": [upload], "display_title": "Test"},
        ):
            with patch(
                "litellm.llms.litellm_proxy.skills.handler.LiteLLMSkillsHandler.create_skill",
                new_callable=AsyncMock,
                return_value=mock_db_skill,
            ):
                user = UserAPIKeyAuth(api_key="test-key")
                result = await _handle_litellm_create_skill(mock_request, user)

                assert result.id == "litellm_skill_test123"

    @pytest.mark.asyncio
    async def test_single_upload_file_not_in_list(self):
        """
        Test that a single UploadFile (not wrapped in list) is handled.
        """
        from litellm.proxy.anthropic_endpoints.skills_endpoints import (
            _handle_litellm_create_skill,
        )

        upload = _make_upload_file("skill/SKILL.md", SKILL_MD_CONTENT)
        mock_request = MagicMock()

        mock_db_skill = _make_db_skill()

        # Single UploadFile, not in a list
        with patch(
            "litellm.proxy.anthropic_endpoints.skills_endpoints.get_form_data",
            new_callable=AsyncMock,
            return_value={"files": upload, "display_title": "Test"},
        ):
            with patch(
                "litellm.llms.litellm_proxy.skills.handler.LiteLLMSkillsHandler.create_skill",
                new_callable=AsyncMock,
                return_value=mock_db_skill,
            ):
                user = UserAPIKeyAuth(api_key="test-key")
                result = await _handle_litellm_create_skill(mock_request, user)

                assert result.id == "litellm_skill_test123"

    @pytest.mark.asyncio
    async def test_no_files_returns_400(self):
        """Test that missing files returns 400."""
        from fastapi import HTTPException

        from litellm.proxy.anthropic_endpoints.skills_endpoints import (
            _handle_litellm_create_skill,
        )

        mock_request = MagicMock()

        with patch(
            "litellm.proxy.anthropic_endpoints.skills_endpoints.get_form_data",
            new_callable=AsyncMock,
            return_value={"display_title": "Test"},
        ):
            user = UserAPIKeyAuth(api_key="test-key")
            with pytest.raises(HTTPException) as exc_info:
                await _handle_litellm_create_skill(mock_request, user)

            assert exc_info.value.status_code == 400
            assert "No files provided" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_empty_files_list_returns_400(self):
        """Test that empty files list returns 400."""
        from fastapi import HTTPException

        from litellm.proxy.anthropic_endpoints.skills_endpoints import (
            _handle_litellm_create_skill,
        )

        mock_request = MagicMock()

        with patch(
            "litellm.proxy.anthropic_endpoints.skills_endpoints.get_form_data",
            new_callable=AsyncMock,
            return_value={"files": [], "display_title": "Test"},
        ):
            user = UserAPIKeyAuth(api_key="test-key")
            with pytest.raises(HTTPException) as exc_info:
                await _handle_litellm_create_skill(mock_request, user)

            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_tuple_format_files(self):
        """Test that (filename, content) tuple format works."""
        from litellm.proxy.anthropic_endpoints.skills_endpoints import (
            _handle_litellm_create_skill,
        )

        mock_request = MagicMock()
        mock_db_skill = _make_db_skill()

        with patch(
            "litellm.proxy.anthropic_endpoints.skills_endpoints.get_form_data",
            new_callable=AsyncMock,
            return_value={
                "files": [("skill/SKILL.md", SKILL_MD_CONTENT)],
                "display_title": "Test",
            },
        ):
            with patch(
                "litellm.llms.litellm_proxy.skills.handler.LiteLLMSkillsHandler.create_skill",
                new_callable=AsyncMock,
                return_value=mock_db_skill,
            ):
                user = UserAPIKeyAuth(api_key="test-key")
                result = await _handle_litellm_create_skill(mock_request, user)

                assert result.id == "litellm_skill_test123"

    @pytest.mark.asyncio
    async def test_multiple_files(self):
        """Test uploading SKILL.md plus additional files."""
        from litellm.proxy.anthropic_endpoints.skills_endpoints import (
            _handle_litellm_create_skill,
        )

        skill_md = _make_upload_file("skill/SKILL.md", SKILL_MD_CONTENT)
        helper_py = _make_upload_file("skill/helper.py", b"def helper(): return 42")
        mock_request = MagicMock()
        mock_db_skill = _make_db_skill()

        with patch(
            "litellm.proxy.anthropic_endpoints.skills_endpoints.get_form_data",
            new_callable=AsyncMock,
            return_value={
                "files": [skill_md, helper_py],
                "display_title": "Multi File Skill",
            },
        ):
            with patch(
                "litellm.llms.litellm_proxy.skills.handler.LiteLLMSkillsHandler.create_skill",
                new_callable=AsyncMock,
                return_value=mock_db_skill,
            ) as mock_create:
                user = UserAPIKeyAuth(api_key="test-key")
                result = await _handle_litellm_create_skill(mock_request, user)

                assert result.id == "litellm_skill_test123"
                # Verify the create was called with file content
                call_args = mock_create.call_args
                data = call_args[1]["data"]
                assert data.file_content is not None

    @pytest.mark.asyncio
    async def test_invalid_frontmatter_returns_400(self):
        """Test that SKILL.md with invalid frontmatter returns 400."""
        from fastapi import HTTPException

        from litellm.proxy.anthropic_endpoints.skills_endpoints import (
            _handle_litellm_create_skill,
        )

        bad_skill_md = b"""---
description: Missing required name field
---

Body content.
"""
        upload = _make_upload_file("skill/SKILL.md", bad_skill_md)
        mock_request = MagicMock()

        with patch(
            "litellm.proxy.anthropic_endpoints.skills_endpoints.get_form_data",
            new_callable=AsyncMock,
            return_value={"files": [upload], "display_title": "Test"},
        ):
            user = UserAPIKeyAuth(api_key="test-key")
            with pytest.raises(HTTPException) as exc_info:
                await _handle_litellm_create_skill(mock_request, user)

            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_display_title_override(self):
        """Test that display_title from form overrides frontmatter name."""
        from litellm.proxy.anthropic_endpoints.skills_endpoints import (
            _handle_litellm_create_skill,
        )

        upload = _make_upload_file("skill/SKILL.md", SKILL_MD_CONTENT)
        mock_request = MagicMock()
        mock_db_skill = _make_db_skill(display_title="Custom Title")

        with patch(
            "litellm.proxy.anthropic_endpoints.skills_endpoints.get_form_data",
            new_callable=AsyncMock,
            return_value={
                "files": [upload],
                "display_title": "Custom Title",
            },
        ):
            with patch(
                "litellm.llms.litellm_proxy.skills.handler.LiteLLMSkillsHandler.create_skill",
                new_callable=AsyncMock,
                return_value=mock_db_skill,
            ) as mock_create:
                user = UserAPIKeyAuth(api_key="test-key")
                await _handle_litellm_create_skill(mock_request, user)

                call_args = mock_create.call_args
                data = call_args[1]["data"]
                assert data.display_title == "Custom Title"

    @pytest.mark.asyncio
    async def test_display_title_falls_back_to_frontmatter_name(self):
        """Test that without display_title override, frontmatter name is used."""
        from litellm.proxy.anthropic_endpoints.skills_endpoints import (
            _handle_litellm_create_skill,
        )

        upload = _make_upload_file("skill/SKILL.md", SKILL_MD_CONTENT)
        mock_request = MagicMock()
        mock_db_skill = _make_db_skill(display_title="test-skill")

        with patch(
            "litellm.proxy.anthropic_endpoints.skills_endpoints.get_form_data",
            new_callable=AsyncMock,
            return_value={"files": [upload]},
        ):
            with patch(
                "litellm.llms.litellm_proxy.skills.handler.LiteLLMSkillsHandler.create_skill",
                new_callable=AsyncMock,
                return_value=mock_db_skill,
            ) as mock_create:
                user = UserAPIKeyAuth(api_key="test-key")
                await _handle_litellm_create_skill(mock_request, user)

                call_args = mock_create.call_args
                data = call_args[1]["data"]
                # Should fall back to frontmatter name "test-skill"
                assert data.display_title == "test-skill"
