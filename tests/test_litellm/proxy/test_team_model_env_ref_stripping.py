"""Regression tests for team-scoped DB model os.environ/ isolation.

Issue #31052: team admins must not be able to reference proxy environment
variables via os.environ/ in DB-stored model params.

Fix: strip_env_refs_for_team_model() blanks os.environ/ values in
litellm_params before they enter the router, but only for team-scoped
DB models (team_id != None).  Proxy-admin-created DB models (team_id is
None) retain the ability to use os.environ/ references.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.common_utils.db_model_utils import strip_env_refs_for_team_model


class TestStripEnvRefsForTeamModel:
    def test_team_model_os_environ_api_key_is_blanked(self) -> None:
        """Team-scoped models must not retain os.environ/ api_key references."""
        params = {"api_key": "os.environ/LITELLM_MASTER_KEY", "model": "openai/gpt-4"}
        result = strip_env_refs_for_team_model(
            litellm_params=params, model_info={"team_id": "team-abc"}
        )
        assert result["api_key"] == "", "os.environ/ api_key must be blanked for team models"
        assert result["model"] == "openai/gpt-4"

    def test_team_model_os_environ_api_base_is_blanked(self) -> None:
        params = {
            "api_key": "sk-real-key",
            "api_base": "os.environ/INTERNAL_API_BASE",
            "model": "openai/gpt-4",
        }
        result = strip_env_refs_for_team_model(
            litellm_params=params, model_info={"team_id": "team-abc"}
        )
        assert result["api_base"] == ""
        assert result["api_key"] == "sk-real-key"

    def test_proxy_admin_model_os_environ_preserved(self) -> None:
        """Proxy-admin models (team_id=None) may use os.environ/ references."""
        params = {"api_key": "os.environ/MY_PROVIDER_KEY", "model": "openai/gpt-4"}
        result = strip_env_refs_for_team_model(
            litellm_params=params, model_info={"team_id": None}
        )
        assert result["api_key"] == "os.environ/MY_PROVIDER_KEY"

    def test_no_model_info_os_environ_preserved(self) -> None:
        """With no model_info, assume proxy-admin context and preserve env refs."""
        params = {"api_key": "os.environ/MY_PROVIDER_KEY", "model": "openai/gpt-4"}
        result = strip_env_refs_for_team_model(litellm_params=params, model_info=None)
        assert result["api_key"] == "os.environ/MY_PROVIDER_KEY"

    def test_multiple_env_refs_all_blanked(self) -> None:
        params = {
            "api_key": "os.environ/LITELLM_MASTER_KEY",
            "api_base": "os.environ/INTERNAL_BASE",
            "model": "openai/gpt-4",
            "api_version": "2024-02-01",
        }
        result = strip_env_refs_for_team_model(
            litellm_params=params, model_info={"team_id": "team-xyz"}
        )
        assert result["api_key"] == ""
        assert result["api_base"] == ""
        assert result["model"] == "openai/gpt-4"
        assert result["api_version"] == "2024-02-01"

    def test_non_env_values_unaffected_for_team_model(self) -> None:
        params = {
            "api_key": "sk-actual-key",
            "api_base": "https://api.openai.com",
            "model": "openai/gpt-4",
        }
        result = strip_env_refs_for_team_model(
            litellm_params=params, model_info={"team_id": "team-abc"}
        )
        assert result == params

    def test_nested_dict_env_refs_blanked(self) -> None:
        """Nested dicts (e.g. extra_headers) must also have os.environ/ stripped."""
        params = {
            "api_key": "sk-real",
            "extra_headers": {"Authorization": "os.environ/INTERNAL_TOKEN"},
        }
        result = strip_env_refs_for_team_model(
            litellm_params=params, model_info={"team_id": "team-abc"}
        )
        assert result["extra_headers"]["Authorization"] == ""
        assert result["api_key"] == "sk-real"

    def test_nested_list_env_refs_blanked(self) -> None:
        """Lists of dicts (e.g. dataSources) must also have os.environ/ stripped."""
        params = {
            "dataSources": [{"parameters": {"key": "os.environ/AZURE_SEARCH_KEY"}}],
        }
        result = strip_env_refs_for_team_model(
            litellm_params=params, model_info={"team_id": "team-abc"}
        )
        assert result["dataSources"][0]["parameters"]["key"] == ""

    def test_pydantic_model_info_team_id_respected(self) -> None:
        """model_info as Pydantic obj with team_id must still trigger stripping."""
        from unittest.mock import MagicMock

        pydantic_info = MagicMock()
        pydantic_info.model_dump.return_value = {"team_id": "team-pydantic"}

        params = {"api_key": "os.environ/SECRET"}
        # simulate the callsite: model_dump() called before passing to strip_env_refs_for_team_model
        result = strip_env_refs_for_team_model(
            litellm_params=params,
            model_info=pydantic_info.model_dump(),
        )
        assert result["api_key"] == ""
