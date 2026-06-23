import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

from litellm.llms.databricks.common_utils import DatabricksBase


def test_databricks_validate_environment():
    databricks_base = DatabricksBase()

    with patch.object(
        databricks_base, "_get_databricks_credentials"
    ) as mock_get_credentials:
        try:
            databricks_base.databricks_validate_environment(
                api_key=None,
                api_base="my_api_base",
                endpoint_type="chat_completions",
                custom_endpoint=False,
                headers=None,
            )
        except Exception:
            pass
        mock_get_credentials.assert_called_once()


class TestDatabricksProfileAuth:
    """PROFILE auth strategy (named ~/.databrickscfg profile)."""

    def _clean_env(self, monkeypatch):
        for var in (
            "DATABRICKS_CLIENT_ID",
            "DATABRICKS_CLIENT_SECRET",
            "DATABRICKS_CONFIG_PROFILE",
            "DATABRICKS_API_BASE",
            "DATABRICKS_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_profile_param_routes_to_sdk_with_profile(self, monkeypatch):
        self._clean_env(monkeypatch)
        base = DatabricksBase()
        with patch.object(
            base,
            "_get_databricks_credentials",
            return_value=("https://h/serving-endpoints", {"Authorization": "Bearer t"}),
        ) as mock_creds:
            base.databricks_resolve_auth(
                api_key=None,
                api_base="https://h",
                custom_endpoint=False,
                headers=None,
                databricks_profile="myprofile",
            )
        assert mock_creds.call_args.kwargs.get("profile") == "myprofile"

    def test_profile_env_var_routes_to_sdk_with_profile(self, monkeypatch):
        self._clean_env(monkeypatch)
        monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "envprofile")
        base = DatabricksBase()
        with patch.object(
            base,
            "_get_databricks_credentials",
            return_value=("https://h/serving-endpoints", {}),
        ) as mock_creds:
            base.databricks_resolve_auth(
                api_key=None,
                api_base="https://h",
                custom_endpoint=False,
                headers=None,
            )
        assert mock_creds.call_args.kwargs.get("profile") == "envprofile"

    def test_explicit_profile_param_overrides_env(self, monkeypatch):
        self._clean_env(monkeypatch)
        monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "envprofile")
        base = DatabricksBase()
        with patch.object(
            base,
            "_get_databricks_credentials",
            return_value=("https://h/serving-endpoints", {}),
        ) as mock_creds:
            base.databricks_resolve_auth(
                api_key=None,
                api_base="https://h",
                custom_endpoint=False,
                headers=None,
                databricks_profile="paramprofile",
            )
        assert mock_creds.call_args.kwargs.get("profile") == "paramprofile"

    def test_pat_takes_priority_over_profile(self, monkeypatch):
        self._clean_env(monkeypatch)
        base = DatabricksBase()
        with patch.object(base, "_get_databricks_credentials") as mock_creds:
            _, headers = base.databricks_resolve_auth(
                api_key="dapiSECRET",
                api_base="https://h",
                custom_endpoint=False,
                headers=None,
                databricks_profile="myprofile",
            )
        mock_creds.assert_not_called()
        assert headers["Authorization"] == "Bearer dapiSECRET"

    def test_resolve_databricks_profile_from_sources(self):
        from types import SimpleNamespace

        assert (
            DatabricksBase.resolve_databricks_profile({"databricks_profile": "p1"})
            == "p1"
        )
        # optional_params (first source) wins over litellm_params (second)
        assert (
            DatabricksBase.resolve_databricks_profile(
                {"databricks_profile": "p1"}, SimpleNamespace(databricks_profile="p2")
            )
            == "p1"
        )
        assert (
            DatabricksBase.resolve_databricks_profile(
                {}, SimpleNamespace(databricks_profile="p2")
            )
            == "p2"
        )
        assert DatabricksBase.resolve_databricks_profile({}, SimpleNamespace()) is None
        assert DatabricksBase.resolve_databricks_profile(None) is None


class TestDatabricksRequestTags:
    """Phase 6 — Databricks-Ai-Gateway-Request-Tags header."""

    def test_explicit_tags_dict(self):
        val = DatabricksBase.build_request_tags_header(
            {"databricks_ai_gateway_request_tags": {"team": "fe", "env": "prod"}},
            None,
        )
        assert json.loads(val) == {"team": "fe", "env": "prod"}

    def test_explicit_tags_popped_from_optional_params(self):
        optional_params = {
            "databricks_ai_gateway_request_tags": {"team": "fe"},
            "temperature": 0.5,
        }
        DatabricksBase.build_request_tags_header(optional_params, None)
        # consumed so it never leaks into the request body
        assert "databricks_ai_gateway_request_tags" not in optional_params
        assert optional_params["temperature"] == 0.5

    def test_tags_list_joined(self):
        val = DatabricksBase.build_request_tags_header(None, {"tags": ["a", "b", "c"]})
        assert json.loads(val) == {"tags": "a,b,c"}

    def test_explicit_dict_and_tags_list_merged(self):
        val = DatabricksBase.build_request_tags_header(
            {"databricks_ai_gateway_request_tags": {"team": "fe"}},
            {"tags": ["x", "y"]},
        )
        assert json.loads(val) == {"team": "fe", "tags": "x,y"}

    def test_values_coerced_to_strings(self):
        val = DatabricksBase.build_request_tags_header(
            {"databricks_ai_gateway_request_tags": {"n": 3, "flag": True}}, None
        )
        assert json.loads(val) == {"n": "3", "flag": "True"}

    def test_none_when_empty(self):
        assert DatabricksBase.build_request_tags_header(None, None) is None
        assert DatabricksBase.build_request_tags_header({}, {"tags": []}) is None

    def test_apply_request_tags_header_sets_header(self):
        base = DatabricksBase()
        headers = {"Content-Type": "application/json"}
        base.apply_request_tags_header(
            headers, litellm_params={"tags": ["t1"]}
        )
        assert headers["Databricks-Ai-Gateway-Request-Tags"] == json.dumps(
            {"tags": "t1"}, separators=(",", ":"), sort_keys=True
        )

    def test_apply_request_tags_header_noop_when_empty(self):
        base = DatabricksBase()
        headers = {"Content-Type": "application/json"}
        base.apply_request_tags_header(headers, optional_params={}, litellm_params={})
        assert "Databricks-Ai-Gateway-Request-Tags" not in headers
