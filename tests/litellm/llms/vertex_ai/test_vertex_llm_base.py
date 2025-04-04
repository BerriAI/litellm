import os
import sys
from unittest.mock import MagicMock, call, patch

import pytest

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase


class TestVertexBase:
    @pytest.mark.asyncio
    async def test_credential_project_validation(self):
        vertex_base = VertexBase()

        # Mock credentials with project_id "project-1"
        mock_creds = MagicMock()
        mock_creds.project_id = "project-1"
        mock_creds.token = "fake-token-1"
        mock_creds.expired = False
        mock_creds.quota_project_id = "project-1"

        # Test case 1: Ensure credentials match project
        with patch.object(
            vertex_base, "load_auth", return_value=(mock_creds, "project-1")
        ):
            token, project = await vertex_base._ensure_access_token_async(
                credentials={"type": "service_account", "project_id": "project-1"},
                project_id="project-1",
                custom_llm_provider="vertex_ai",
            )
            assert project == "project-1"
            assert token == "fake-token-1"

        # Test case 2: Prevent using credentials from different project
        with patch.object(
            vertex_base, "load_auth", return_value=(mock_creds, "project-1")
        ):
            with pytest.raises(ValueError, match="Could not resolve project_id"):
                result = await vertex_base._ensure_access_token_async(
                    credentials={"type": "service_account"},
                    project_id="different-project",
                    custom_llm_provider="vertex_ai",
                )
                print(f"result: {result}")

    @pytest.mark.asyncio
    async def test_dynamic_credential_updates(self):
        vertex_base = VertexBase()

        # Initial credentials
        mock_creds_1 = MagicMock()
        mock_creds_1.token = "token-1"
        mock_creds_1.expired = False
        mock_creds_1.project_id = "project-1"

        # Updated credentials
        mock_creds_2 = MagicMock()
        mock_creds_2.token = "token-2"
        mock_creds_2.expired = False
        mock_creds_2.project_id = "project-1"

        # Test case 1: Initial credential load
        with patch.object(
            vertex_base, "load_auth", return_value=(mock_creds_1, "project-1")
        ):
            token, project = await vertex_base._ensure_access_token_async(
                credentials={"type": "service_account"},
                project_id="project-1",
                custom_llm_provider="vertex_ai",
            )
            assert token == "token-1"

        # Test case 2: Credential refresh when expired
        mock_creds_1.expired = True
        mock_creds_1.token = None

        with patch.object(vertex_base, "refresh_auth") as mock_refresh:

            async def mock_refresh_impl(creds):
                creds.token = "refreshed-token"
                creds.expired = False

            mock_refresh.side_effect = mock_refresh_impl

            token, project = await vertex_base._ensure_access_token_async(
                credentials=None,  # Use existing credentials
                project_id="project-1",
                custom_llm_provider="vertex_ai",
            )
            assert token == "refreshed-token"
            assert mock_refresh.called

    @pytest.mark.asyncio
    async def test_gemini_credentials(self):
        vertex_base = VertexBase()

        # Test that Gemini requests bypass credential checks
        token, project = await vertex_base._ensure_access_token_async(
            credentials=None, project_id=None, custom_llm_provider="gemini"
        )
        assert token == ""
        assert project == ""
