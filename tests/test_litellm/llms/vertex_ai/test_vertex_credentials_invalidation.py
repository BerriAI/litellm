"""
Tests for VertexBase.invalidate_credentials() — credential cache eviction
on 401/UNAUTHENTICATED errors from the Vertex AI API.

Fixes: https://github.com/BerriAI/litellm/issues/23512
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("."))

from litellm.llms.vertex_ai.vertex_llm_base import VertexBase


class TestVertexBaseInvalidateCredentials:
    """Test suite for VertexBase.invalidate_credentials()."""

    def _make_base_with_cached_creds(self, credentials, project_id):
        """Helper: create a VertexBase instance with a pre-populated cache entry."""
        base = VertexBase()
        mock_creds = MagicMock()
        mock_creds.token = "valid-looking-but-actually-revoked-token"
        mock_creds.expired = False

        import json

        cache_key_creds = (
            json.dumps(credentials) if isinstance(credentials, dict) else credentials
        )
        cache_key = (cache_key_creds, project_id)
        base._credentials_project_mapping[cache_key] = (mock_creds, project_id)
        return base, cache_key

    def test_invalidate_credentials_removes_cache_entry(self):
        """invalidate_credentials should delete the matching cache entry."""
        creds_str = "/path/to/service_account.json"
        project_id = "my-project"
        base, cache_key = self._make_base_with_cached_creds(creds_str, project_id)

        assert cache_key in base._credentials_project_mapping
        base.invalidate_credentials(credentials=creds_str, project_id=project_id)
        assert cache_key not in base._credentials_project_mapping

    def test_invalidate_credentials_with_dict_credentials(self):
        """invalidate_credentials should work with dict credentials (JSON-serialized key)."""
        creds_dict = {"type": "service_account", "project_id": "my-project"}
        project_id = "my-project"
        base, cache_key = self._make_base_with_cached_creds(creds_dict, project_id)

        assert cache_key in base._credentials_project_mapping
        base.invalidate_credentials(credentials=creds_dict, project_id=project_id)
        assert cache_key not in base._credentials_project_mapping

    def test_invalidate_credentials_noop_when_not_cached(self):
        """invalidate_credentials should not raise when the key is not in the cache."""
        base = VertexBase()
        # Should not raise
        base.invalidate_credentials(
            credentials="/path/to/nonexistent.json", project_id="no-project"
        )

    def test_invalidate_credentials_preserves_other_entries(self):
        """invalidate_credentials should only remove the targeted entry."""
        base = VertexBase()
        mock_creds_a = MagicMock()
        mock_creds_a.token = "token-a"
        mock_creds_a.expired = False
        mock_creds_b = MagicMock()
        mock_creds_b.token = "token-b"
        mock_creds_b.expired = False

        key_a = ("creds-a", "project-a")
        key_b = ("creds-b", "project-b")
        base._credentials_project_mapping[key_a] = (mock_creds_a, "project-a")
        base._credentials_project_mapping[key_b] = (mock_creds_b, "project-b")

        base.invalidate_credentials(credentials="creds-a", project_id="project-a")

        assert key_a not in base._credentials_project_mapping
        assert key_b in base._credentials_project_mapping

    def test_get_access_token_reloads_after_invalidation(self):
        """
        After invalidate_credentials, the next get_access_token call should
        reload credentials via load_auth instead of returning the stale token.
        """
        creds_str = "/path/to/sa.json"
        project_id = "my-project"
        base, cache_key = self._make_base_with_cached_creds(creds_str, project_id)

        # Invalidate
        base.invalidate_credentials(credentials=creds_str, project_id=project_id)

        # Now get_access_token should call load_auth since cache is empty
        fresh_creds = MagicMock()
        fresh_creds.token = "fresh-valid-token"
        fresh_creds.expired = False

        with patch.object(
            base, "load_auth", return_value=(fresh_creds, project_id)
        ) as mock_load:
            token, returned_project = base.get_access_token(
                credentials=creds_str, project_id=project_id
            )

        mock_load.assert_called_once_with(
            credentials=creds_str, project_id=project_id
        )
        assert token == "fresh-valid-token"
        assert returned_project == project_id
