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


def run_sync(coro):
    """Helper to run coroutine synchronously for testing"""
    import asyncio

    return asyncio.run(coro)


class TestVertexBase:
    @pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
    @pytest.mark.asyncio
    async def test_credential_project_validation(self, is_async):
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
            if is_async:
                token, project = await vertex_base._ensure_access_token_async(
                    credentials={"type": "service_account", "project_id": "project-1"},
                    project_id="project-1",
                    custom_llm_provider="vertex_ai",
                )
            else:
                token, project = vertex_base._ensure_access_token(
                    credentials={"type": "service_account", "project_id": "project-1"},
                    project_id="project-1",
                    custom_llm_provider="vertex_ai",
                )
            assert project == "project-1"
            assert token == "fake-token-1"

        # Test case 2: Allow using credentials from different project
        with patch.object(
            vertex_base, "load_auth", return_value=(mock_creds, "project-1")
        ):
            if is_async:
                result = await vertex_base._ensure_access_token_async(
                    credentials={"type": "service_account"},
                    project_id="different-project",
                    custom_llm_provider="vertex_ai",
                )
            else:
                result = vertex_base._ensure_access_token(
                    credentials={"type": "service_account"},
                    project_id="different-project",
                    custom_llm_provider="vertex_ai",
                )
            print(f"result: {result}")

    @pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
    @pytest.mark.asyncio
    async def test_cached_credentials(self, is_async):
        vertex_base = VertexBase()

        # Initial credentials
        mock_creds = MagicMock()
        mock_creds.token = "token-1"
        mock_creds.expired = False
        mock_creds.project_id = "project-1"
        mock_creds.quota_project_id = "project-1"

        # Test initial credential load and caching
        with patch.object(
            vertex_base, "load_auth", return_value=(mock_creds, "project-1")
        ):
            # First call should load credentials
            if is_async:
                token, project = await vertex_base._ensure_access_token_async(
                    credentials={"type": "service_account"},
                    project_id="project-1",
                    custom_llm_provider="vertex_ai",
                )
            else:
                token, project = vertex_base._ensure_access_token(
                    credentials={"type": "service_account"},
                    project_id="project-1",
                    custom_llm_provider="vertex_ai",
                )
            assert token == "token-1"

            # Second call should use cached credentials
            if is_async:
                token2, project2 = await vertex_base._ensure_access_token_async(
                    credentials={"type": "service_account"},
                    project_id="project-1",
                    custom_llm_provider="vertex_ai",
                )
            else:
                token2, project2 = vertex_base._ensure_access_token(
                    credentials={"type": "service_account"},
                    project_id="project-1",
                    custom_llm_provider="vertex_ai",
                )
            assert token2 == "token-1"
            assert project2 == "project-1"

    @pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
    @pytest.mark.asyncio
    async def test_credential_refresh(self, is_async):
        vertex_base = VertexBase()

        # Create expired credentials
        mock_creds = MagicMock()
        mock_creds.token = "my-token"
        mock_creds.expired = True
        mock_creds.project_id = "project-1"
        mock_creds.quota_project_id = "project-1"

        with patch.object(
            vertex_base, "load_auth", return_value=(mock_creds, "project-1")
        ), patch.object(vertex_base, "refresh_auth") as mock_refresh:

            def mock_refresh_impl(creds):
                creds.token = "refreshed-token"
                creds.expired = False

            mock_refresh.side_effect = mock_refresh_impl

            if is_async:
                token, project = await vertex_base._ensure_access_token_async(
                    credentials={"type": "service_account"},
                    project_id="project-1",
                    custom_llm_provider="vertex_ai",
                )
            else:
                token, project = vertex_base._ensure_access_token(
                    credentials={"type": "service_account"},
                    project_id="project-1",
                    custom_llm_provider="vertex_ai",
                )

            assert mock_refresh.called
            assert token == "refreshed-token"
            assert not mock_creds.expired

    @pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
    @pytest.mark.asyncio
    async def test_gemini_credentials(self, is_async):
        vertex_base = VertexBase()

        # Test that Gemini requests bypass credential checks
        if is_async:
            token, project = await vertex_base._ensure_access_token_async(
                credentials=None, project_id=None, custom_llm_provider="gemini"
            )
        else:
            token, project = vertex_base._ensure_access_token(
                credentials=None, project_id=None, custom_llm_provider="gemini"
            )
        assert token == ""
        assert project == ""

    @pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
    @pytest.mark.asyncio
    async def test_authorized_user_credentials(self, is_async):
        vertex_base = VertexBase()

        quota_project_id = "test-project"

        credentials = {
            "account": "",
            "client_id": "fake-client-id",
            "client_secret": "fake-secret",
            "quota_project_id": "test-project",
            "refresh_token": "fake-refresh-token",
            "type": "authorized_user",
            "universe_domain": "googleapis.com",
        }

        mock_creds = MagicMock()
        mock_creds.token = "token-1"
        mock_creds.expired = False
        mock_creds.quota_project_id = quota_project_id

        with patch.object(
            vertex_base, "_credentials_from_authorized_user", return_value=mock_creds
        ) as mock_credentials_from_authorized_user, patch.object(
            vertex_base, "refresh_auth"
        ) as mock_refresh:

            def mock_refresh_impl(creds):
                creds.token = "refreshed-token"

            mock_refresh.side_effect = mock_refresh_impl

            # 1. Test that authorized_user-style credentials are correctly handled and uses quota_project_id
            if is_async:
                token, project = await vertex_base._ensure_access_token_async(
                    credentials=credentials,
                    project_id=None,
                    custom_llm_provider="vertex_ai",
                )
            else:
                token, project = vertex_base._ensure_access_token(
                    credentials=credentials,
                    project_id=None,
                    custom_llm_provider="vertex_ai",
                )

            assert mock_credentials_from_authorized_user.called
            assert token == "refreshed-token"
            assert project == quota_project_id

            # 2. Test that authorized_user-style credentials are correctly handled and uses passed in project_id
            not_quota_project_id = "new-project"
            if is_async:
                token, project = await vertex_base._ensure_access_token_async(
                    credentials=credentials,
                    project_id=not_quota_project_id,
                    custom_llm_provider="vertex_ai",
                )
            else:
                token, project = vertex_base._ensure_access_token(
                    credentials=credentials,
                    project_id=not_quota_project_id,
                    custom_llm_provider="vertex_ai",
                )

            assert token == "refreshed-token"
            assert project == not_quota_project_id

    @pytest.mark.parametrize(
        "api_base, vertex_location, expected",
        [
            (None, "us-central1", "https://us-central1-aiplatform.googleapis.com"),
            (None, "global", "https://aiplatform.googleapis.com"),
            (
                "https://us-central1-aiplatform.googleapis.com",
                "us-central1",
                "https://us-central1-aiplatform.googleapis.com",
            ),
            (
                "https://aiplatform.googleapis.com",
                "global",
                "https://aiplatform.googleapis.com",
            ),
            (
                "https://us-central1-aiplatform.googleapis.com",
                "global",
                "https://us-central1-aiplatform.googleapis.com",
            ),
            (
                "https://aiplatform.googleapis.com",
                "us-central1",
                "https://aiplatform.googleapis.com",
            ),
        ],
    )
    def test_get_api_base(self, api_base, vertex_location, expected):
        vertex_base = VertexBase()
        assert (
            vertex_base.get_api_base(api_base=api_base, vertex_location=vertex_location)
            == expected
        ), f"Expected {expected} with api_base {api_base} and vertex_location {vertex_location}"
