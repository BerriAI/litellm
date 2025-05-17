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

        # Test case 2: Prevent using credentials from different project
        with patch.object(
            vertex_base, "load_auth", return_value=(mock_creds, "project-1")
        ):
            with pytest.raises(ValueError, match="Could not resolve project_id"):
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

    def test_load_auth_wif(self):
        vertex_base = VertexBase()
        input_project_id = "some_project_id"

        # Test case 1: Using Workload Identity Federation for Microsoft Azure and
        # OIDC identity providers (default behavior)
        json_obj_1  = {
            "type": "external_account",
        }
        mock_auth_1 = MagicMock()
        mock_creds_1 = MagicMock()
        mock_request_1 = MagicMock()
        mock_creds_1 = mock_auth_1.identity_pool.Credentials.from_info.return_value
        with patch.dict(sys.modules, {"google.auth": mock_auth_1,
                                      "google.auth.transport.requests": mock_request_1}):
            
            creds_1, project_id = vertex_base.load_auth(
                credentials=json_obj_1, project_id=input_project_id
            )

        mock_auth_1.identity_pool.Credentials.from_info.assert_called_once_with(
            json_obj_1
        )
        mock_creds_1.refresh.assert_called_once_with(
            mock_request_1.Request.return_value
        )
        assert creds_1 == mock_creds_1
        assert project_id == input_project_id

        # Test case 2: Using Workload Identity Federation for AWS
        json_obj_2  = {
            "type": "external_account",
            "credential_source": {"environment_id": "aws1"}
        }
        mock_auth_2 = MagicMock()
        mock_creds_2 = MagicMock()
        mock_request_2 = MagicMock()
        mock_creds_2 = mock_auth_2.aws.Credentials.from_info.return_value
        with patch.dict(sys.modules, {"google.auth": mock_auth_2,
                                      "google.auth.transport.requests": mock_request_2}):
            
            creds_2, project_id = vertex_base.load_auth(
                credentials=json_obj_2, project_id=input_project_id
            )

        mock_auth_2.aws.Credentials.from_info.assert_called_once_with(
            json_obj_2
        )
        mock_creds_2.refresh.assert_called_once_with(
            mock_request_2.Request.return_value
        )
        assert creds_2 == mock_creds_2
        assert project_id == input_project_id
