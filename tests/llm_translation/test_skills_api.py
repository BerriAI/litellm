"""
Tests for Skills API operations across providers.

All tests use mocked HTTP responses so no real network calls are made,
keeping this folder CI-safe and free of provider-credential requirements.
"""

import io
import os
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.types.llms.anthropic_skills import (
    DeleteSkillResponse,
    ListSkillsResponse,
    Skill,
)

# ---------------------------------------------------------------------------
# Shared mock response payloads
# ---------------------------------------------------------------------------

MOCK_SKILL = {
    "id": "skill_mock_001",
    "created_at": "2025-01-01T00:00:00Z",
    "display_title": "Mock Skill",
    "latest_version": "v1",
    "source": "custom",
    "type": "skill",
    "updated_at": "2025-01-01T00:00:00Z",
}

MOCK_LIST_RESPONSE = {
    "data": [MOCK_SKILL],
    "has_more": False,
}

MOCK_DELETE_RESPONSE = {
    "id": "skill_mock_001",
    "type": "skill_deleted",
}


def _mock_httpx_response(json_body: dict, status_code: int = 200) -> httpx.Response:
    """Build a minimal ``httpx.Response`` that behaves like a real one."""
    import json as _json

    return httpx.Response(
        status_code=status_code,
        content=_json.dumps(json_body).encode(),
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", "https://mock.test"),
    )


@contextmanager
def create_skill_zip(skill_name: str):
    """
    Helper context manager to create a zip file for a skill.

    Args:
        skill_name: Name of the skill directory in test_skills_data/

    Yields:
        File handle to the zip file

    The zip file is automatically cleaned up after use.
    """
    test_dir = Path(__file__).parent / "test_skills_data"
    skill_dir = test_dir / skill_name

    zip_path = test_dir / f"{skill_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(skill_dir, arcname=skill_name)
        zip_file.write(skill_dir / "SKILL.md", arcname=f"{skill_name}/SKILL.md")

    try:
        with open(zip_path, "rb") as f:
            yield f
    finally:
        if zip_path.exists():
            zip_path.unlink()


class TestAnthropicSkillsAPI:
    """Mock-based tests for the Anthropic Skills API translation layer."""

    provider = "anthropic"

    # -- create --------------------------------------------------------

    @patch("litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post")
    def test_create_skill(self, mock_post):
        mock_post.return_value = _mock_httpx_response(MOCK_SKILL)

        skill_name = "test-skill-litellm"
        with create_skill_zip(skill_name) as zip_file:
            response = litellm.create_skill(
                display_title="Mock Skill",
                files=[zip_file],
                custom_llm_provider=self.provider,
                api_key="sk-mock-key",
            )

        assert isinstance(response, Skill)
        assert response.id == "skill_mock_001"
        mock_post.assert_called_once()

    # -- list ----------------------------------------------------------

    @patch("litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.get")
    def test_list_skills(self, mock_get):
        mock_get.return_value = _mock_httpx_response(MOCK_LIST_RESPONSE)

        response = litellm.list_skills(
            limit=10,
            custom_llm_provider=self.provider,
            api_key="sk-mock-key",
        )

        assert isinstance(response, ListSkillsResponse)
        assert len(response.data) == 1
        assert response.data[0].id == "skill_mock_001"
        mock_get.assert_called_once()

    # -- get -----------------------------------------------------------

    @patch("litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.get")
    def test_get_skill(self, mock_get):
        mock_get.return_value = _mock_httpx_response(MOCK_SKILL)

        response = litellm.get_skill(
            skill_id="skill_mock_001",
            custom_llm_provider=self.provider,
            api_key="sk-mock-key",
        )

        assert isinstance(response, Skill)
        assert response.id == "skill_mock_001"
        mock_get.assert_called_once()

    # -- delete --------------------------------------------------------

    @patch("litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.delete")
    def test_delete_skill(self, mock_delete):
        mock_delete.return_value = _mock_httpx_response(MOCK_DELETE_RESPONSE)

        response = litellm.delete_skill(
            skill_id="skill_mock_001",
            custom_llm_provider=self.provider,
            api_key="sk-mock-key",
        )

        assert isinstance(response, DeleteSkillResponse)
        assert response.type == "skill_deleted"
        mock_delete.assert_called_once()

