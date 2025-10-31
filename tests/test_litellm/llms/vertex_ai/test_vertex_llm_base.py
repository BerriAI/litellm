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
        ), patch.object(
            vertex_base, "load_auth_async", return_value=(mock_creds, "project-1")
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
        ), patch.object(
            vertex_base, "load_auth_async", return_value=(mock_creds, "project-1")
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
        ), patch.object(
            vertex_base, "load_auth_async", return_value=(mock_creds, "project-1")
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
        ), patch.object(
            vertex_base, "load_auth_async", return_value=(mock_creds, "project-1")
        ), patch.object(vertex_base, "refresh_auth") as mock_refresh, patch.object(
            vertex_base, "refresh_auth_async"
        ) as mock_refresh_async:

            def mock_refresh_impl(creds):
                creds.token = "refreshed-token"
                creds.expired = False

            mock_refresh.side_effect = mock_refresh_impl
            mock_refresh_async.side_effect = mock_refresh_impl

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

            # Check the appropriate mock based on async/sync
            if is_async:
                assert mock_refresh_async.called
            else:
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
            vertex_base, "_credentials_from_authorized_user_async", return_value=mock_creds
        ) as mock_credentials_from_authorized_user_async, patch.object(vertex_base, "refresh_auth") as mock_refresh, patch.object(
            vertex_base, "refresh_auth_async"
        ) as mock_refresh_async:

            def mock_refresh_impl(creds):
                creds.token = "refreshed-token"

            mock_refresh.side_effect = mock_refresh_impl
            mock_refresh_async.side_effect = mock_refresh_impl

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

            # Verify the appropriate method was called
            if is_async:
                assert mock_credentials_from_authorized_user_async.called
            else:
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
        ) as mock_credentials_from_identity_pool, patch.object(vertex_base, "refresh_auth") as mock_refresh, patch.object(
            vertex_base, "refresh_auth_async"
        ) as mock_refresh_async:

            def mock_refresh_impl(creds):
                creds.token = "refreshed-token"

            mock_refresh.side_effect = mock_refresh_impl
            mock_refresh_async.side_effect = mock_refresh_impl

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
        ) as mock_credentials_from_identity_pool_with_aws, patch.object(vertex_base, "refresh_auth") as mock_refresh, patch.object(
            vertex_base, "refresh_auth_async"
        ) as mock_refresh_async:

            def mock_refresh_impl(creds):
                creds.token = "refreshed-token"

            mock_refresh.side_effect = mock_refresh_impl
            mock_refresh_async.side_effect = mock_refresh_impl

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
        ), patch.object(
            vertex_base, "load_auth_async", return_value=(mock_creds, "project-1")
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
        ), patch.object(
            vertex_base, "load_auth_async", return_value=(mock_creds, "resolved-project")
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
        ), patch.object(
            vertex_base, "load_auth_async", return_value=(mock_creds, "project-1")
        ), patch.object(vertex_base, "refresh_auth") as mock_refresh, patch.object(
            vertex_base, "refresh_auth_async"
        ) as mock_refresh_async:

            def mock_refresh_impl(creds):
                creds.token = "refreshed-token"
                creds.expired = False

            mock_refresh.side_effect = mock_refresh_impl
            mock_refresh_async.side_effect = mock_refresh_impl

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

            # Check the appropriate mock based on async/sync
            if is_async:
                assert mock_refresh_async.called
            else:
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
        ), patch.object(
            vertex_base, "load_auth_async", return_value=(mock_creds, "cred-project")
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
        ) as mock_load_auth_sync, patch.object(
            vertex_base, "load_auth_async", return_value=(mock_creds, "resolved-from-credentials")
        ) as mock_load_auth_async:

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
            mock_load_auth = mock_load_auth_async if is_async else mock_load_auth_sync
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

    @pytest.mark.asyncio
    async def test_async_auth_uses_async_methods(self):
        """Test that async auth uses load_auth_async and refresh_auth_async (now the default)"""
        vertex_base = VertexBase()

        mock_creds = MagicMock()
        mock_creds.token = "async-token"
        mock_creds.expired = False
        mock_creds.project_id = "async-project"
        mock_creds.quota_project_id = "async-project"

        credentials = {"type": "service_account", "project_id": "async-project"}

        with patch.object(
            vertex_base, "load_auth_async", return_value=(mock_creds, "async-project")
        ) as mock_load_auth_async, patch.object(
            vertex_base, "load_auth"
        ) as mock_load_auth_sync:

            token, project = await vertex_base.get_access_token_async(
                credentials=credentials,
                project_id="async-project",
            )

            # Verify async method was called
            assert mock_load_auth_async.called
            # Verify sync method was NOT called
            assert not mock_load_auth_sync.called
            assert token == "async-token"
            assert project == "async-project"

    @pytest.mark.asyncio
    async def test_refresh_auth_async_with_aiohttp(self):
        """Test that refresh_auth_async uses aiohttp when available"""
        vertex_base = VertexBase()

        mock_creds = MagicMock()
        mock_creds.expired = True
        mock_creds.token = None

        async def mock_refresh(request):
            # Simulate successful token refresh (ASYNC function)
            mock_creds.token = "refreshed-async-token"
            mock_creds.expired = False

        # Make refresh an async coroutine (simulating async credentials)
        mock_creds.refresh = mock_refresh

        # Call refresh_auth_async
        await vertex_base.refresh_auth_async(mock_creds)

        # Verify credentials were refreshed
        assert mock_creds.token == "refreshed-async-token"
        assert not mock_creds.expired

    @pytest.mark.asyncio
    async def test_load_auth_async_service_account(self):
        """Test load_auth_async with service account credentials creates async credentials"""
        vertex_base = VertexBase()

        credentials = {
            "type": "service_account",
            "project_id": "test-project",
            "client_email": "test@test-project.iam.gserviceaccount.com",
        }

        mock_creds = MagicMock()
        mock_creds.token = "loaded-token"
        mock_creds.expired = False
        mock_creds.project_id = "test-project"

        # Patch the ASYNC credential creation method
        with patch.object(
            vertex_base, "_credentials_from_service_account_async", return_value=mock_creds
        ) as mock_service_account_async, patch.object(
            vertex_base, "refresh_auth_async"
        ) as mock_refresh_async:

            async def mock_refresh_impl(creds):
                creds.token = "async-refreshed-token"
                creds.expired = False

            mock_refresh_async.side_effect = mock_refresh_impl

            creds, project = await vertex_base.load_auth_async(
                credentials=credentials,
                project_id="test-project"
            )

            # Verify ASYNC service account method was called
            assert mock_service_account_async.called
            # Verify async refresh was called
            assert mock_refresh_async.called
            assert creds.token == "async-refreshed-token"
            assert project == "test-project"

    @pytest.mark.asyncio
    async def test_async_token_refresh_when_expired(self):
        """Test that expired tokens are refreshed using async method"""
        vertex_base = VertexBase()

        # Create expired credentials
        mock_creds = MagicMock()
        mock_creds.token = "old-token"
        mock_creds.expired = True
        mock_creds.project_id = "test-project"
        mock_creds.quota_project_id = "test-project"

        credentials = {"type": "service_account", "project_id": "test-project"}

        with patch.object(
            vertex_base, "load_auth_async", return_value=(mock_creds, "test-project")
        ) as mock_load_auth_async, patch.object(
            vertex_base, "refresh_auth_async"
        ) as mock_refresh_async:

            async def mock_refresh_impl(creds):
                creds.token = "refreshed-async-token"
                creds.expired = False

            mock_refresh_async.side_effect = mock_refresh_impl

            token, project = await vertex_base.get_access_token_async(
                credentials=credentials,
                project_id="test-project",
            )

            # Verify refresh_auth_async was called for expired credentials
            assert mock_refresh_async.called
            assert token == "refreshed-async-token"
            assert not mock_creds.expired
            assert project == "test-project"

    @pytest.mark.asyncio
    async def test_async_caching_with_new_implementation(self):
        """Test that credential caching works correctly with async implementation"""
        vertex_base = VertexBase()

        mock_creds = MagicMock()
        mock_creds.token = "cached-async-token"
        mock_creds.expired = False
        mock_creds.project_id = "cached-project"
        mock_creds.quota_project_id = "cached-project"

        credentials = {"type": "service_account", "project_id": "cached-project"}

        with patch.object(
            vertex_base, "load_auth_async", return_value=(mock_creds, "cached-project")
        ) as mock_load_auth_async:

            # First call - should load credentials
            token1, project1 = await vertex_base.get_access_token_async(
                credentials=credentials,
                project_id="cached-project",
            )

            assert mock_load_auth_async.call_count == 1
            assert token1 == "cached-async-token"

            # Second call - should use cached credentials
            token2, project2 = await vertex_base.get_access_token_async(
                credentials=credentials,
                project_id="cached-project",
            )

            # Should still be only 1 call (used cache)
            assert mock_load_auth_async.call_count == 1
            assert token2 == "cached-async-token"
            assert project2 == "cached-project"

            # Verify cache entry exists
            cache_key = (json.dumps(credentials), "cached-project")
            assert cache_key in vertex_base._credentials_project_mapping

    @pytest.mark.asyncio
    async def test_async_and_sync_share_same_cache(self):
        """Test that async and sync implementations share the same credential cache"""
        vertex_base = VertexBase()

        mock_creds = MagicMock()
        mock_creds.token = "shared-cache-token"
        mock_creds.expired = False
        mock_creds.project_id = "shared-project"
        mock_creds.quota_project_id = "shared-project"

        credentials = {"type": "service_account", "project_id": "shared-project"}

        with patch.object(
            vertex_base, "load_auth_async", return_value=(mock_creds, "shared-project")
        ) as mock_load_auth_async, patch.object(
            vertex_base, "load_auth", return_value=(mock_creds, "shared-project")
        ) as mock_load_auth_sync:

            # First call with async
            token1, project1 = await vertex_base.get_access_token_async(
                credentials=credentials,
                project_id="shared-project",
            )

            assert mock_load_auth_async.call_count == 1
            assert token1 == "shared-cache-token"

            # Second call with sync (should use same cache)
            token2, project2 = vertex_base.get_access_token(
                credentials=credentials,
                project_id="shared-project",
            )

            # Should NOT call load_auth because cache was populated by async call
            assert mock_load_auth_sync.call_count == 0
            assert token2 == "shared-cache-token"
            assert project2 == "shared-project"

    @pytest.mark.asyncio
    async def test_load_auth_async_authorized_user(self):
        """Test load_auth_async with authorized user credentials creates async credentials"""
        vertex_base = VertexBase()

        credentials = {
            "type": "authorized_user",
            "client_id": "test-client-id",
            "client_secret": "test-secret",
            "refresh_token": "test-refresh-token",
            "quota_project_id": "test-quota-project",
        }

        mock_creds = MagicMock()
        mock_creds.token = "authorized-user-token"
        mock_creds.expired = False
        mock_creds.quota_project_id = "test-quota-project"

        # Patch the ASYNC credential creation method
        with patch.object(
            vertex_base, "_credentials_from_authorized_user_async", return_value=mock_creds
        ) as mock_authorized_user_async, patch.object(
            vertex_base, "refresh_auth_async"
        ) as mock_refresh_async:

            async def mock_refresh_impl(creds):
                creds.token = "refreshed-authorized-token"

            mock_refresh_async.side_effect = mock_refresh_impl

            creds, project = await vertex_base.load_auth_async(
                credentials=credentials,
                project_id=None
            )

            # Verify ASYNC authorized user method was called
            assert mock_authorized_user_async.called
            # Verify async refresh was called
            assert mock_refresh_async.called
            assert creds.token == "refreshed-authorized-token"
            # Should use quota_project_id when project_id is None
            assert project == "test-quota-project"

    @pytest.mark.asyncio
    async def test_async_credentials_with_old_transport(self):
        """
        Test that async credentials use OLD transport for TRUE async refresh.
        This verifies the implementation: OLD async credentials + OLD transport (compatible).
        """
        vertex_base = VertexBase()

        # Create mock async credentials (simulating _credentials_async or _service_account_async)
        mock_creds = MagicMock()
        mock_creds.token = "initial-token"
        mock_creds.expired = True
        
        # Track whether refresh was called and what request type was used
        refresh_called = []
        
        async def async_refresh(request):
            """Simulates async refresh method from google.oauth2._credentials_async"""
            refresh_called.append({
                'request_type': type(request).__name__,
                'has_session': hasattr(request, 'session') or hasattr(request, '_session')
            })
            mock_creds.token = "async-refreshed-token"
            mock_creds.expired = False
        
        # Make refresh a coroutine to simulate async credentials
        mock_creds.refresh = async_refresh
        
        # Call refresh_auth_async
        await vertex_base.refresh_auth_async(mock_creds)
        
        # Verify async refresh was called
        assert len(refresh_called) == 1, "Async refresh should be called once"
        assert refresh_called[0]['request_type'] == 'Request', "Should use Request from OLD transport"
        assert refresh_called[0]['has_session'], "Request should have session"
        assert mock_creds.token == "async-refreshed-token"
        assert not mock_creds.expired
        
        print(f"✅ Async credentials used with OLD transport (google.auth.transport._aiohttp_requests)")
        
        # Cleanup
        await VertexBase.close_token_refresh_session()

    @pytest.mark.asyncio
    async def test_persistent_session_reuse_across_multiple_refreshes(self):
        """
        Test that the same aiohttp session is reused across multiple token refreshes.
        This verifies the session pooling optimization.
        """
        import aiohttp
        from unittest.mock import MagicMock
        
        # Close any existing session to start fresh
        await VertexBase.close_token_refresh_session()
        
        # Track session IDs captured during refresh
        session_ids = []
        
        # Create a mock credentials object with ASYNC refresh
        mock_creds = MagicMock()
        mock_creds.token = "test-token"
        mock_creds.expired = True
        
        async def mock_refresh(request):
            # Capture the session ID each time refresh is called (ASYNC function)
            # Note: OLD transport google.auth.transport._aiohttp_requests.Request uses 'session' attribute
            session = getattr(request, 'session', None) or getattr(request, '_session', None)
            if session:
                session_ids.append(id(session))
            mock_creds.token = f"refreshed-token-{len(session_ids)}"
        
        # Make refresh an async coroutine (simulating async credentials)
        mock_creds.refresh = mock_refresh
        
        vertex_base = VertexBase()
        
        # Perform multiple token refreshes
        await vertex_base.refresh_auth_async(mock_creds)
        await vertex_base.refresh_auth_async(mock_creds)
        await vertex_base.refresh_auth_async(mock_creds)
        
        # Verify all refreshes used the SAME session instance
        assert len(session_ids) == 3, f"Expected 3 refreshes, got {len(session_ids)}"
        assert session_ids[0] == session_ids[1] == session_ids[2], \
            f"Session IDs should be identical, got: {session_ids}"
        
        print(f"✅ All 3 refreshes used the same session (ID: {session_ids[0]})")
        
        # Verify the session has the correct settings
        session = await VertexBase._get_or_create_token_refresh_session()
        assert isinstance(session, aiohttp.ClientSession)
        assert session._auto_decompress is False, "Session should have auto_decompress=False"
        assert not session.closed, "Session should still be open"
        
        print(f"✅ Session has auto_decompress=False: {session._auto_decompress is False}")
        
        # Test cleanup
        await VertexBase.close_token_refresh_session()
        assert session.closed, "Session should be closed after cleanup"
        
        print(f"✅ Session properly closed after cleanup")
        
        # Verify new session is created after cleanup
        new_session = await VertexBase._get_or_create_token_refresh_session()
        assert id(new_session) != id(session), "New session should be created after cleanup"
        assert not new_session.closed, "New session should be open"
        
        print(f"✅ New session created after cleanup (old ID: {id(session)}, new ID: {id(new_session)})")
        
        # Cleanup for next test
        await VertexBase.close_token_refresh_session()

    @pytest.mark.asyncio
    async def test_concurrent_token_refresh_uses_same_session(self):
        """
        Test that concurrent token refreshes all use the same session.
        This verifies thread-safety of session creation.
        """
        import asyncio
        import time
        from unittest.mock import MagicMock
        
        # Close any existing session to start fresh
        await VertexBase.close_token_refresh_session()
        
        # Track session IDs from concurrent requests
        session_ids = []
        
        async def refresh_and_track(vertex_base, creds, index):
            await vertex_base.refresh_auth_async(creds)
            session = await VertexBase._get_or_create_token_refresh_session()
            session_ids.append((index, id(session)))
        
        # Create multiple vertex base instances (simulating multiple concurrent requests)
        vertex_bases = [VertexBase() for _ in range(5)]
        
        # Create mock credentials for each
        mock_creds_list = []
        for i in range(5):
            mock_creds = MagicMock()
            mock_creds.token = f"test-token-{i}"
            mock_creds.expired = True
            
            async def mock_refresh(request):
                # Simulate network delay (ASYNC function)
                await asyncio.sleep(0.01)
                mock_creds.token = "refreshed"
            
            # Make refresh an async coroutine (simulating async credentials)
            mock_creds.refresh = mock_refresh
            mock_creds_list.append(mock_creds)
        
        # Run all refreshes concurrently
        await asyncio.gather(*[
            refresh_and_track(vb, creds, i) 
            for i, (vb, creds) in enumerate(zip(vertex_bases, mock_creds_list))
        ])
        
        # All concurrent requests should have used the SAME session
        assert len(session_ids) == 5, f"Expected 5 concurrent requests, got {len(session_ids)}"
        unique_sessions = set(sid for _, sid in session_ids)
        assert len(unique_sessions) == 1, \
            f"All concurrent requests should use same session, got {len(unique_sessions)} unique sessions"
        
        print(f"✅ All 5 concurrent requests used the same session (ID: {list(unique_sessions)[0]})")
        
        # Cleanup
        await VertexBase.close_token_refresh_session()
