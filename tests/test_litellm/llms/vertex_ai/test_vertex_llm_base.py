import json
import os
import sys
from unittest.mock import MagicMock, call, patch

import pytest

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.vertex_ai.aws_credentials_supplier import (
    Boto3AwsSecurityCredentialsSupplier,
)
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

    @pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
    @pytest.mark.asyncio
    async def test_identity_pool_credentials(self, is_async):
        vertex_base = VertexBase()

        # Test case: Using Workload Identity Federation for Microsoft Azure and
        # OIDC identity providers (default behavior)
        credentials = {
            "project_id": "test-project",
            "refresh_token": "fake-refresh-token",
            "type": "external_account",
        }
        mock_creds = MagicMock()
        mock_creds.token = "token-1"
        mock_creds.expired = False
        mock_creds.project_id = "test-project"

        with patch.object(
            vertex_base, "_credentials_from_identity_pool", return_value=mock_creds
        ) as mock_credentials_from_identity_pool, patch.object(
            vertex_base, "refresh_auth"
        ) as mock_refresh:

            def mock_refresh_impl(creds):
                creds.token = "refreshed-token"

            mock_refresh.side_effect = mock_refresh_impl

            if is_async:
                token, _ = await vertex_base._ensure_access_token_async(
                    credentials=credentials,
                    project_id=None,
                    custom_llm_provider="vertex_ai",
                )
            else:
                token, _ = vertex_base._ensure_access_token(
                    credentials=credentials,
                    project_id=None,
                    custom_llm_provider="vertex_ai",
                )

            assert mock_credentials_from_identity_pool.called
            assert token == "refreshed-token"

    @pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
    @pytest.mark.asyncio
    async def test_identity_pool_credentials_with_aws(self, is_async):
        vertex_base = VertexBase()

        # Test case: Using Workload Identity Federation for Microsoft Azure and
        # OIDC identity providers (default behavior)
        credentials = {
            "project_id": "test-project",
            "refresh_token": "fake-refresh-token",
            "type": "external_account",
            "credential_source": {
                "environment_id": "aws1"
            }
        }
        mock_creds = MagicMock()
        mock_creds.token = "token-1"
        mock_creds.expired = False
        mock_creds.project_id = "test-project"

        with patch.object(
            vertex_base, "_credentials_from_identity_pool_with_aws", return_value=mock_creds
        ) as mock_credentials_from_identity_pool_with_aws, patch.object(
            vertex_base, "refresh_auth"
        ) as mock_refresh:

            def mock_refresh_impl(creds):
                creds.token = "refreshed-token"

            mock_refresh.side_effect = mock_refresh_impl

            if is_async:
                token, _ = await vertex_base._ensure_access_token_async(
                    credentials=credentials,
                    project_id=None,
                    custom_llm_provider="vertex_ai",
                )
            else:
                token, _ = vertex_base._ensure_access_token(
                    credentials=credentials,
                    project_id=None,
                    custom_llm_provider="vertex_ai",
                )

            assert mock_credentials_from_identity_pool_with_aws.called
            assert token == "refreshed-token"

    @pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
    @pytest.mark.asyncio
    async def test_new_cache_format_tuple_storage(self, is_async):
        """Test that new cache format stores (credentials, project_id) tuples"""
        vertex_base = VertexBase()

        mock_creds = MagicMock()
        mock_creds.token = "token-1"
        mock_creds.expired = False
        mock_creds.project_id = "project-1"
        mock_creds.quota_project_id = "project-1"

        credentials = {"type": "service_account", "project_id": "project-1"}

        with patch.object(
            vertex_base, "load_auth", return_value=(mock_creds, "project-1")
        ):
            if is_async:
                token, project = await vertex_base._ensure_access_token_async(
                    credentials=credentials,
                    project_id="project-1",
                    custom_llm_provider="vertex_ai",
                )
            else:
                token, project = vertex_base._ensure_access_token(
                    credentials=credentials,
                    project_id="project-1",
                    custom_llm_provider="vertex_ai",
                )

            assert token == "token-1"
            assert project == "project-1"

            # Verify cache stores tuple format
            cache_key = (json.dumps(credentials), "project-1")
            assert cache_key in vertex_base._credentials_project_mapping
            cached_entry = vertex_base._credentials_project_mapping[cache_key]
            assert isinstance(cached_entry, tuple)
            assert len(cached_entry) == 2
            cached_creds, cached_project = cached_entry
            assert cached_creds == mock_creds
            assert cached_project == "project-1"

    @pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
    @pytest.mark.asyncio
    async def test_backward_compatibility_old_cache_format(self, is_async):
        """Test backward compatibility with old cache format (just credentials)"""
        vertex_base = VertexBase()

        mock_creds = MagicMock()
        mock_creds.token = "token-1"
        mock_creds.expired = False
        mock_creds.project_id = "project-1"
        mock_creds.quota_project_id = "project-1"

        credentials = {"type": "service_account", "project_id": "project-1"}

        # Simulate old cache format by manually adding just credentials (not tuple)
        cache_key = (json.dumps(credentials), "project-1")
        vertex_base._credentials_project_mapping[cache_key] = mock_creds

        # Should handle old format gracefully
        if is_async:
            token, project = await vertex_base._ensure_access_token_async(
                credentials=credentials,
                project_id="project-1",
                custom_llm_provider="vertex_ai",
            )
        else:
            token, project = vertex_base._ensure_access_token(
                credentials=credentials,
                project_id="project-1",
                custom_llm_provider="vertex_ai",
            )

        assert token == "token-1"
        assert project == "project-1"

    @pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
    @pytest.mark.asyncio
    async def test_resolved_project_id_cache_optimization(self, is_async):
        """Test that resolved project_id creates additional cache entries for optimization"""
        vertex_base = VertexBase()

        mock_creds = MagicMock()
        mock_creds.token = "token-1"
        mock_creds.expired = False
        mock_creds.project_id = "resolved-project"
        mock_creds.quota_project_id = "resolved-project"

        credentials = {"type": "service_account"}

        with patch.object(
            vertex_base, "load_auth", return_value=(mock_creds, "resolved-project")
        ):
            # Call without project_id, should use resolved project from credentials
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

            assert token == "token-1"
            assert project == "resolved-project"

                        # Verify both cache entries exist
            original_cache_key = (json.dumps(credentials), None)
            resolved_cache_key = (json.dumps(credentials), "resolved-project")

            assert original_cache_key in vertex_base._credentials_project_mapping
            assert resolved_cache_key in vertex_base._credentials_project_mapping

            # Both should contain the same tuple
            original_entry = vertex_base._credentials_project_mapping[original_cache_key]
            resolved_entry = vertex_base._credentials_project_mapping[resolved_cache_key]

            assert isinstance(original_entry, tuple)
            assert isinstance(resolved_entry, tuple)
            assert original_entry[0] == mock_creds
            assert original_entry[1] == "resolved-project"
            assert resolved_entry[0] == mock_creds
            assert resolved_entry[1] == "resolved-project"

    @pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
    @pytest.mark.asyncio
    async def test_cache_update_on_credential_refresh(self, is_async):
        """Test that cache is updated when credentials are refreshed"""
        vertex_base = VertexBase()

        mock_creds = MagicMock()
        mock_creds.token = "original-token"
        mock_creds.expired = True  # Start with expired credentials
        mock_creds.project_id = "project-1"
        mock_creds.quota_project_id = "project-1"

        credentials = {"type": "service_account", "project_id": "project-1"}

        with patch.object(
            vertex_base, "load_auth", return_value=(mock_creds, "project-1")
        ), patch.object(vertex_base, "refresh_auth") as mock_refresh:

            def mock_refresh_impl(creds):
                creds.token = "refreshed-token"
                creds.expired = False

            mock_refresh.side_effect = mock_refresh_impl

            if is_async:
                token, project = await vertex_base._ensure_access_token_async(
                    credentials=credentials,
                    project_id="project-1",
                    custom_llm_provider="vertex_ai",
                )
            else:
                token, project = vertex_base._ensure_access_token(
                    credentials=credentials,
                    project_id="project-1",
                    custom_llm_provider="vertex_ai",
                )

            assert mock_refresh.called
            assert token == "refreshed-token"
            assert project == "project-1"

            # Verify cache was updated with refreshed credentials
            cache_key = (json.dumps(credentials), "project-1")
            assert cache_key in vertex_base._credentials_project_mapping
            cached_entry = vertex_base._credentials_project_mapping[cache_key]
            assert isinstance(cached_entry, tuple)
            cached_creds, cached_project = cached_entry
            assert cached_creds.token == "refreshed-token"
            assert not cached_creds.expired
            assert cached_project == "project-1"

    @pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
    @pytest.mark.asyncio
    async def test_cache_with_different_project_id_combinations(self, is_async):
        """Test caching behavior with different project_id parameter combinations"""
        vertex_base = VertexBase()

        mock_creds = MagicMock()
        mock_creds.token = "token-1"
        mock_creds.expired = False
        mock_creds.project_id = "cred-project"
        mock_creds.quota_project_id = "cred-project"

        credentials = {"type": "service_account", "project_id": "cred-project"}

        with patch.object(
            vertex_base, "load_auth", return_value=(mock_creds, "cred-project")
        ):
            # First call with explicit project_id
            if is_async:
                token1, project1 = await vertex_base._ensure_access_token_async(
                    credentials=credentials,
                    project_id="explicit-project",
                    custom_llm_provider="vertex_ai",
                )
            else:
                token1, project1 = vertex_base._ensure_access_token(
                    credentials=credentials,
                    project_id="explicit-project",
                    custom_llm_provider="vertex_ai",
                )

            # Second call with None project_id (should use credential project)
            if is_async:
                token2, project2 = await vertex_base._ensure_access_token_async(
                    credentials=credentials,
                    project_id=None,
                    custom_llm_provider="vertex_ai",
                )
            else:
                token2, project2 = vertex_base._ensure_access_token(
                    credentials=credentials,
                    project_id=None,
                    custom_llm_provider="vertex_ai",
                )

            assert token1 == "token-1"
            assert project1 == "explicit-project"  # Should use explicit project_id
            assert token2 == "token-1"
            assert project2 == "cred-project"  # Should use credential project_id

            # Verify separate cache entries
            explicit_cache_key = (json.dumps(credentials), "explicit-project")
            none_cache_key = (json.dumps(credentials), None)
            resolved_cache_key = (json.dumps(credentials), "cred-project")

            assert explicit_cache_key in vertex_base._credentials_project_mapping
            assert none_cache_key in vertex_base._credentials_project_mapping
            assert resolved_cache_key in vertex_base._credentials_project_mapping

    @pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
    @pytest.mark.asyncio
    async def test_project_id_resolution_and_caching_core_issue(self, is_async):
        """
        When user doesn't provide project_id, system should resolve it from credentials
        and cache the resolved project_id for future calls without calling load_auth again.
        """
        vertex_base = VertexBase()

        mock_creds = MagicMock()
        mock_creds.token = "token-from-creds"
        mock_creds.expired = False
        mock_creds.project_id = "resolved-from-credentials"
        mock_creds.quota_project_id = "resolved-from-credentials"

        # User provides credentials but NO project_id (this is the key scenario)
        credentials = {"type": "service_account"}

        with patch.object(
            vertex_base, "load_auth", return_value=(mock_creds, "resolved-from-credentials")
        ) as mock_load_auth:

            # First call: User provides NO project_id, should resolve from credentials
            if is_async:
                token1, project1 = await vertex_base._ensure_access_token_async(
                    credentials=credentials,
                    project_id=None,  # Key: user doesn't provide project_id
                    custom_llm_provider="vertex_ai",
                )
            else:
                token1, project1 = vertex_base._ensure_access_token(
                    credentials=credentials,
                    project_id=None,  # Key: user doesn't provide project_id
                    custom_llm_provider="vertex_ai",
                )

            # Should have called load_auth once to resolve project_id
            assert mock_load_auth.call_count == 1
            assert token1 == "token-from-creds"
            assert project1 == "resolved-from-credentials"

            # Verify cache contains both the original key and resolved key
            original_cache_key = (json.dumps(credentials), None)
            resolved_cache_key = (json.dumps(credentials), "resolved-from-credentials")

            assert original_cache_key in vertex_base._credentials_project_mapping
            assert resolved_cache_key in vertex_base._credentials_project_mapping

            # Both should contain the tuple with resolved project_id
            original_entry = vertex_base._credentials_project_mapping[original_cache_key]
            resolved_entry = vertex_base._credentials_project_mapping[resolved_cache_key]

            assert isinstance(original_entry, tuple)
            assert isinstance(resolved_entry, tuple)
            assert original_entry[1] == "resolved-from-credentials"
            assert resolved_entry[1] == "resolved-from-credentials"

            # Second call: Same scenario - should use cache and NOT call load_auth again
            if is_async:
                token2, project2 = await vertex_base._ensure_access_token_async(
                    credentials=credentials,
                    project_id=None,  # Still no project_id provided
                    custom_llm_provider="vertex_ai",
                )
            else:
                token2, project2 = vertex_base._ensure_access_token(
                    credentials=credentials,
                    project_id=None,  # Still no project_id provided
                    custom_llm_provider="vertex_ai",
                )

            # Should NOT have called load_auth again (still 1 call total)
            assert mock_load_auth.call_count == 1
            assert token2 == "token-from-creds"
            assert project2 == "resolved-from-credentials"

            # Third call: Now user provides the resolved project_id explicitly
            # This should also use cache (the resolved_cache_key)
            if is_async:
                token3, project3 = await vertex_base._ensure_access_token_async(
                    credentials=credentials,
                    project_id="resolved-from-credentials",  # Explicit resolved project_id
                    custom_llm_provider="vertex_ai",
                )
            else:
                token3, project3 = vertex_base._ensure_access_token(
                    credentials=credentials,
                    project_id="resolved-from-credentials",  # Explicit resolved project_id
                    custom_llm_provider="vertex_ai",
                )

            # Should still NOT have called load_auth again (cache hit)
            assert mock_load_auth.call_count == 1
            assert token3 == "token-from-creds"
            assert project3 == "resolved-from-credentials"

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

    @pytest.mark.parametrize(
        "api_base, custom_llm_provider, gemini_api_key, endpoint, stream, auth_header, url, model, expected_auth_header, expected_url",
        [
            # Test case 1: Gemini with custom API base
            (
                "https://proxy.example.com/generativelanguage.googleapis.com/v1beta",
                "gemini",
                "test-api-key",
                "generateContent",
                False,
                None,
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent",
                "gemini-2.5-flash-lite",
                "test-api-key",
                "https://proxy.example.com/generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
            ),
            # Test case 2: Gemini with custom API base and streaming
            (
                "https://proxy.example.com/generativelanguage.googleapis.com/v1beta",
                "gemini",
                "test-api-key",
                "generateContent",
                True,
                None,
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent",
                "gemini-2.5-flash-lite",
                "test-api-key",
                "https://proxy.example.com/generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?alt=sse"
            ),
            # Test case 3: Non-Gemini provider with custom API base
            (
                "https://custom-vertex-api.com",
                "vertex_ai",
                None,
                "generateContent",
                False,
                "Bearer token123",
                "https://aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-pro:generateContent",
                "gemini-pro",
                "Bearer token123",
                "https://custom-vertex-api.com:generateContent"
            ),
            # Test case 4: No API base provided (should return original values)
            (
                None,
                "gemini",
                "test-api-key",
                "generateContent",
                False,
                "Bearer token123",
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent",
                "gemini-2.5-flash-lite",
                "Bearer token123",
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
            ),
            # Test case 5: Gemini without API key (should raise ValueError)
            (
                "https://proxy.example.com/generativelanguage.googleapis.com/v1beta",
                "gemini",
                None,
                "generateContent",
                False,
                None,
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent",
                "gemini-2.5-flash-lite",
                None,  # This should raise an exception
                None
            ),
        ],
    )
    def test_check_custom_proxy(
        self, 
        api_base, 
        custom_llm_provider, 
        gemini_api_key, 
        endpoint, 
        stream, 
        auth_header, 
        url, 
        model, 
        expected_auth_header, 
        expected_url
    ):
        """Test the _check_custom_proxy method for handling custom API base URLs"""
        vertex_base = VertexBase()
        
        if custom_llm_provider == "gemini" and api_base and gemini_api_key is None:
            # Test case 5: Should raise ValueError for Gemini without API key
            with pytest.raises(ValueError, match="Missing gemini_api_key"):
                vertex_base._check_custom_proxy(
                    api_base=api_base,
                    custom_llm_provider=custom_llm_provider,
                    gemini_api_key=gemini_api_key,
                    endpoint=endpoint,
                    stream=stream,
                    auth_header=auth_header,
                    url=url,
                    model=model,
                )
        else:
            # Test cases 1-4: Should work correctly
            result_auth_header, result_url = vertex_base._check_custom_proxy(
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                gemini_api_key=gemini_api_key,
                endpoint=endpoint,
                stream=stream,
                auth_header=auth_header,
                url=url,
                model=model,
            )
            
            assert result_auth_header == expected_auth_header, f"Expected auth_header {expected_auth_header}, got {result_auth_header}"
            assert result_url == expected_url, f"Expected URL {expected_url}, got {result_url}"

    def test_check_custom_proxy_gemini_url_construction(self):
        """Test that Gemini URLs are constructed correctly with custom API base"""
        vertex_base = VertexBase()
        
        # Test various Gemini models with custom API base
        test_cases = [
            ("gemini-2.5-flash-lite", "generateContent", "https://proxy.example.com/generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"),
            ("gemini-2.5-pro", "generateContent", "https://proxy.example.com/generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"),
            ("gemini-1.5-flash", "streamGenerateContent", "https://proxy.example.com/generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent"),
        ]
        
        for model, endpoint, expected_url in test_cases:
            _, result_url = vertex_base._check_custom_proxy(
                api_base="https://proxy.example.com/generativelanguage.googleapis.com/v1beta",
                custom_llm_provider="gemini",
                gemini_api_key="test-api-key",
                endpoint=endpoint,
                stream=False,
                auth_header=None,
                url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:{endpoint}",
                model=model,
            )
            
            assert result_url == expected_url, f"Expected {expected_url}, got {result_url} for model {model}"

    def test_check_custom_proxy_streaming_parameter(self):
        """Test that streaming parameter correctly adds ?alt=sse to URLs"""
        vertex_base = VertexBase()
        
        # Test with streaming enabled
        _, result_url_streaming = vertex_base._check_custom_proxy(
            api_base="https://proxy.example.com/generativelanguage.googleapis.com/v1beta",
            custom_llm_provider="gemini",
            gemini_api_key="test-api-key",
            endpoint="generateContent",
            stream=True,
            auth_header=None,
            url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent",
            model="gemini-2.5-flash-lite",
        )
        
        expected_streaming_url = "https://proxy.example.com/generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?alt=sse"
        assert result_url_streaming == expected_streaming_url, f"Expected {expected_streaming_url}, got {result_url_streaming}"
        
        # Test with streaming disabled
        _, result_url_no_streaming = vertex_base._check_custom_proxy(
            api_base="https://proxy.example.com/generativelanguage.googleapis.com/v1beta",
            custom_llm_provider="gemini",
            gemini_api_key="test-api-key",
            endpoint="generateContent",
            stream=False,
            auth_header=None,
            url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent",
            model="gemini-2.5-flash-lite",
        )
        
        expected_no_streaming_url = "https://proxy.example.com/generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
        assert result_url_no_streaming == expected_no_streaming_url, f"Expected {expected_no_streaming_url}, got {result_url_no_streaming}"

    def test_aws_federation_with_explicit_credentials(self):
        """
        Test AWS to GCP federation using explicit AWS credentials (no metadata endpoints).
        
        This test verifies that when AWS auth parameters are provided in the credential config,
        LiteLLM uses BaseAWSLLM to get AWS credentials and creates a custom supplier for GCP.
        """
        vertex_base = VertexBase()

        credentials = {
            "type": "external_account",
            "audience": "//iam.googleapis.com/projects/123456/locations/global/workloadIdentityPools/my-pool/providers/aws-provider",
            "subject_token_type": "urn:ietf:params:aws:token-type:aws4_request",
            "token_url": "https://sts.googleapis.com/v1/token",
            "service_account_impersonation_url": "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/my-sa@project.iam.gserviceaccount.com:generateAccessToken",
            "credential_source": {
                "environment_id": "aws1",
                "regional_cred_verification_url": "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15"
            },
            "aws_role_name": "arn:aws:iam::123456789012:role/MyRole",
            "aws_session_name": "litellm-test",
            "aws_region_name": "us-east-1"
        }

        # Mock boto3 credentials
        mock_boto3_creds = MagicMock()
        mock_boto3_creds.access_key = "AKIAIOSFODNN7EXAMPLE"
        mock_boto3_creds.secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        mock_boto3_creds.token = "fake-session-token"
        
        def mock_get_frozen_credentials():
            frozen = MagicMock()
            frozen.access_key = "AKIAIOSFODNN7EXAMPLE"
            frozen.secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
            frozen.token = "fake-session-token"
            return frozen
        
        mock_boto3_creds.get_frozen_credentials = mock_get_frozen_credentials

        # Mock GCP credentials
        mock_gcp_creds = MagicMock()
        mock_gcp_creds.token = "gcp-token"
        mock_gcp_creds.expired = False
        mock_gcp_creds.project_id = "test-project"

        with patch("litellm.llms.bedrock.base_aws_llm.BaseAWSLLM.get_credentials", return_value=mock_boto3_creds) as mock_aws_creds, \
             patch("google.auth.aws.Credentials", return_value=mock_gcp_creds) as mock_gcp_aws_creds, \
             patch.object(vertex_base, "refresh_auth") as mock_refresh:

            def mock_refresh_impl(creds):
                creds.token = "refreshed-gcp-token"

            mock_refresh.side_effect = mock_refresh_impl

            # Call load_auth
            creds, project_id = vertex_base.load_auth(
                credentials=credentials,
                project_id=None
            )

            # Verify BaseAWSLLM.get_credentials was called with correct params
            mock_aws_creds.assert_called_once()
            call_kwargs = mock_aws_creds.call_args[1]
            assert call_kwargs["aws_role_name"] == "arn:aws:iam::123456789012:role/MyRole"
            assert call_kwargs["aws_session_name"] == "litellm-test"
            assert call_kwargs["aws_region_name"] == "us-east-1"

            # Verify google.auth.aws.Credentials was called
            assert mock_gcp_aws_creds.called
            call_kwargs = mock_gcp_aws_creds.call_args[1]
            
            # Verify custom supplier was used
            assert call_kwargs["aws_security_credentials_supplier"] is not None
            assert call_kwargs["credential_source"] is None  # Not using metadata

            # Verify credentials were refreshed
            assert mock_refresh.called
            assert creds.token == "refreshed-gcp-token"

    def test_boto3_aws_security_credentials_supplier(self):
        """
        Test Boto3AwsSecurityCredentialsSupplier correctly wraps boto3 credentials.
        """
        # Mock boto3 credentials with get_frozen_credentials
        mock_boto3_creds = MagicMock()
        
        frozen_creds = MagicMock()
        frozen_creds.access_key = "AKIATEST123"
        frozen_creds.secret_key = "secretkey123"
        frozen_creds.token = "session-token-123"
        
        mock_boto3_creds.get_frozen_credentials = MagicMock(return_value=frozen_creds)

        # Create supplier
        supplier = Boto3AwsSecurityCredentialsSupplier(
            boto3_credentials=mock_boto3_creds,
            aws_region="us-west-2"
        )

        # Test get_aws_security_credentials
        creds = supplier.get_aws_security_credentials(context=None, request=None)
        
        assert creds["access_key_id"] == "AKIATEST123"
        assert creds["secret_access_key"] == "secretkey123"
        assert creds["security_token"] == "session-token-123"

        # Test get_aws_region
        region = supplier.get_aws_region(context=None, request=None)
        assert region == "us-west-2"

    def test_extract_aws_params(self):
        """
        Test _extract_aws_params correctly identifies AWS auth parameters.
        """
        vertex_base = VertexBase()

        # Test with AWS role params
        json_obj = {
            "type": "external_account",
            "audience": "//iam.googleapis.com/projects/123/locations/global/workloadIdentityPools/pool/providers/aws",
            "aws_role_name": "arn:aws:iam::123456789012:role/MyRole",
            "aws_session_name": "my-session",
            "aws_region_name": "us-east-1"
        }

        aws_params = vertex_base._extract_aws_params(json_obj)
        
        assert aws_params is not None
        assert aws_params["aws_role_name"] == "arn:aws:iam::123456789012:role/MyRole"
        assert aws_params["aws_session_name"] == "my-session"
        assert aws_params["aws_region_name"] == "us-east-1"

        # Test with no AWS params
        json_obj_no_aws = {
            "type": "external_account",
            "audience": "//iam.googleapis.com/projects/123/locations/global/workloadIdentityPools/pool/providers/aws"
        }

        aws_params_none = vertex_base._extract_aws_params(json_obj_no_aws)
        assert aws_params_none is None

    def test_aws_federation_fallback_to_metadata(self):
        """
        Test that when no AWS auth params are provided, it falls back to metadata-based flow.
        """
        vertex_base = VertexBase()

        credentials = {
            "type": "external_account",
            "audience": "//iam.googleapis.com/projects/123456/locations/global/workloadIdentityPools/my-pool/providers/aws-provider",
            "subject_token_type": "urn:ietf:params:aws:token-type:aws4_request",
            "token_url": "https://sts.googleapis.com/v1/token",
            "credential_source": {
                "environment_id": "aws1",
                "regional_cred_verification_url": "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15"
            }
        }

        mock_creds = MagicMock()
        mock_creds.token = "token-from-metadata"
        mock_creds.expired = False
        mock_creds.project_id = "test-project"

        with patch.object(
            vertex_base, "_credentials_from_identity_pool_with_aws", return_value=mock_creds
        ) as mock_metadata_auth, \
             patch.object(
            vertex_base, "_credentials_from_aws_with_explicit_auth"
        ) as mock_explicit_auth, \
             patch.object(vertex_base, "refresh_auth") as mock_refresh:

            def mock_refresh_impl(creds):
                creds.token = "refreshed-token"

            mock_refresh.side_effect = mock_refresh_impl

            # Call load_auth - should use metadata-based flow
            creds, project_id = vertex_base.load_auth(
                credentials=credentials,
                project_id=None
            )

            # Verify metadata-based auth was used
            assert mock_metadata_auth.called
            # Verify explicit auth was NOT used
            assert not mock_explicit_auth.called
            # Verify credentials were returned
            assert creds.token == "refreshed-token"
