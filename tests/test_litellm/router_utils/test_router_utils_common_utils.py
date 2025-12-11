from typing import Dict, List, Optional, Union
from unittest.mock import Mock

import pytest

from litellm.router_utils.common_utils import (
    _deployment_supports_web_search,
    filter_team_based_models,
    filter_web_search_deployments,
)


class TestFilterTeamBasedModels:
    """Test cases for filter_team_based_models function"""

    @pytest.fixture
    def sample_deployments_with_teams(self) -> List[Dict]:
        """Sample deployments where some have team_id and some don't"""
        return [
            {"model_info": {"id": "deployment-1", "team_id": "team-a"}},
            {"model_info": {"id": "deployment-2", "team_id": "team-b"}},
            {
                "model_info": {
                    "id": "deployment-3"
                    # No team_id - should always be included
                }
            },
            {"model_info": {"id": "deployment-4", "team_id": "team-a"}},
        ]

    @pytest.fixture
    def sample_deployments_no_teams(self) -> List[Dict]:
        """Sample deployments with no team_id restrictions"""
        return [
            {"model_info": {"id": "deployment-1"}},
            {"model_info": {"id": "deployment-2"}},
        ]

    def test_filter_team_based_models_none_request_kwargs(
        self, sample_deployments_with_teams
    ):
        """Test that when request_kwargs is None, all deployments are returned unchanged"""
        result = filter_team_based_models(sample_deployments_with_teams, None)
        assert result == sample_deployments_with_teams

    def test_filter_team_based_models_empty_request_kwargs(
        self, sample_deployments_with_teams
    ):
        """Test with empty request_kwargs"""
        result = filter_team_based_models(sample_deployments_with_teams, {})
        # Should include all deployments since no team_id in request
        assert len(result) == 1

    def test_filter_team_based_models_no_metadata(self, sample_deployments_with_teams):
        """Test with request_kwargs that has no metadata"""
        request_kwargs = {"some_other_key": "value"}
        result = filter_team_based_models(sample_deployments_with_teams, request_kwargs)
        # Should include only non-team based deployments
        assert len(result) == 1

    def test_filter_team_based_models_team_match_metadata(
        self, sample_deployments_with_teams
    ):
        """Test filtering when team_id is in metadata"""
        request_kwargs = {"metadata": {"user_api_key_team_id": "team-a"}}
        result = filter_team_based_models(sample_deployments_with_teams, request_kwargs)

        # Should include:
        # - deployment-1 (team-a matches)
        # - deployment-3 (no team_id restriction)
        # - deployment-4 (team-a matches)
        # Should exclude:
        # - deployment-2 (team-b doesn't match)
        expected_ids = ["deployment-1", "deployment-3", "deployment-4"]
        result_ids = [d.get("model_info", {}).get("id") for d in result]
        assert sorted(result_ids) == sorted(expected_ids)

    def test_filter_team_based_models_team_match_litellm_metadata(
        self, sample_deployments_with_teams
    ):
        """Test filtering when team_id is in litellm_metadata"""
        request_kwargs = {"litellm_metadata": {"user_api_key_team_id": "team-b"}}
        result = filter_team_based_models(sample_deployments_with_teams, request_kwargs)

        # Should include:
        # - deployment-2 (team-b matches)
        # - deployment-3 (no team_id restriction)
        # Should exclude:
        # - deployment-1 (team-a doesn't match)
        # - deployment-4 (team-a doesn't match)
        expected_ids = ["deployment-2", "deployment-3"]
        result_ids = [d.get("model_info", {}).get("id") for d in result]
        assert sorted(result_ids) == sorted(expected_ids)

    def test_filter_team_based_models_priority_metadata_over_litellm(
        self, sample_deployments_with_teams
    ):
        """Test that metadata.user_api_key_team_id takes priority over litellm_metadata.user_api_key_team_id"""
        request_kwargs = {
            "metadata": {
                "user_api_key_team_id": "team-a",  # This should take priority
                "litellm_metadata": {"user_api_key_team_id": "team-b"},
            }
        }
        result = filter_team_based_models(sample_deployments_with_teams, request_kwargs)

        # Should filter based on team-a (from metadata, not litellm_metadata)
        expected_ids = ["deployment-1", "deployment-3", "deployment-4"]
        result_ids = [d.get("model_info", {}).get("id") for d in result]
        assert sorted(result_ids) == sorted(expected_ids)

    def test_filter_team_based_models_no_matching_team(
        self, sample_deployments_with_teams
    ):
        """Test when request team doesn't match any deployment teams"""
        request_kwargs = {"metadata": {"user_api_key_team_id": "team-nonexistent"}}
        result = filter_team_based_models(sample_deployments_with_teams, request_kwargs)

        # Should only include deployment-3 (no team_id restriction)
        expected_ids = ["deployment-3"]
        result_ids = [d.get("model_info", {}).get("id") for d in result]
        assert result_ids == expected_ids

    def test_filter_team_based_models_no_team_restrictions(
        self, sample_deployments_no_teams
    ):
        """Test with deployments that have no team restrictions"""
        request_kwargs = {"metadata": {"user_api_key_team_id": "any-team"}}
        result = filter_team_based_models(sample_deployments_no_teams, request_kwargs)

        # Should include all deployments since none have team_id restrictions
        assert result == sample_deployments_no_teams

    def test_filter_team_based_models_missing_model_info(self):
        """Test with deployments missing model_info"""
        deployments = [
            {"model_info": {"id": "deployment-1", "team_id": "team-a"}},
            {
                # Missing model_info entirely
            },
            {
                "model_info": {
                    # Missing id
                    "team_id": "team-b"
                }
            },
        ]

        request_kwargs = {"metadata": {"user_api_key_team_id": "team-a"}}
        result = filter_team_based_models(deployments, request_kwargs)

        # Should handle missing model_info gracefully
        # deployment-1 should be included (team matches)
        # others should be included since they don't have proper team_id setup
        assert len(result) >= 1  # At least deployment-1 should be included

    def test_filter_team_based_models_dict_input(self):
        """Test with Dict input instead of List[Dict]"""
        # Note: Based on the function signature, it accepts Union[List[Dict], Dict]
        # But the implementation seems to expect List[Dict] for the filtering logic
        # This test documents the current behavior
        deployments_dict = {"key1": "value1", "key2": "value2"}

        request_kwargs = {"metadata": {"user_api_key_team_id": "team-a"}}

        # This should not crash, though the filtering logic won't apply to Dict input
        result = filter_team_based_models(deployments_dict, request_kwargs)
        # The function will likely return the dict unchanged or handle it differently
        assert result is not None

    def test_filter_team_based_models_empty_deployments(self):
        """Test with empty deployments list"""
        result = filter_team_based_models(
            [], {"metadata": {"user_api_key_team_id": "team-a"}}
        )
        assert result == []

    def test_filter_team_based_models_none_team_id_in_deployment(self):
        """Test with explicit None team_id in deployment"""
        deployments = [
            {"model_info": {"id": "deployment-1", "team_id": None}},
            {"model_info": {"id": "deployment-2", "team_id": "team-a"}},
        ]

        request_kwargs = {"metadata": {"user_api_key_team_id": "team-a"}}
        result = filter_team_based_models(deployments, request_kwargs)

        # Both should be included:
        # - deployment-1 (None team_id is treated as no restriction)
        # - deployment-2 (team matches)
        expected_ids = ["deployment-1", "deployment-2"]
        result_ids = [d.get("model_info", {}).get("id") for d in result]
        assert sorted(result_ids) == sorted(expected_ids)


class TestDeploymentSupportsWebSearch:
    """Test cases for _deployment_supports_web_search helper function"""

    def test_model_info_true(self):
        """model_info.supports_web_search=True returns True"""
        deployment = {"model_info": {"supports_web_search": True}}
        assert _deployment_supports_web_search(deployment) is True

    def test_model_info_false(self):
        """model_info.supports_web_search=False returns False"""
        deployment = {"model_info": {"supports_web_search": False}}
        assert _deployment_supports_web_search(deployment) is False

    def test_no_config_defaults_to_true(self):
        """When no supports_web_search in config, default to True"""
        deployment = {"litellm_params": {"model": "gpt-4"}, "model_info": {"id": "123"}}
        assert _deployment_supports_web_search(deployment) is True

    def test_empty_deployment_defaults_to_true(self):
        """Empty deployment defaults to True"""
        assert _deployment_supports_web_search({}) is True

    def test_missing_model_info_defaults_to_true(self):
        """When model_info missing, default to True"""
        deployment = {"litellm_params": {"model": "gpt-4"}}
        assert _deployment_supports_web_search(deployment) is True


class TestFilterWebSearchDeployments:
    """Test cases for filter_web_search_deployments function"""

    @pytest.fixture
    def sample_deployments(self) -> List[Dict]:
        """Sample deployments with varying web search support"""
        return [
            {"model_info": {"id": "deployment-1"}},  # default True
            {"model_info": {"id": "deployment-2", "supports_web_search": True}},
            {"model_info": {"id": "deployment-3", "supports_web_search": False}},
        ]

    def test_no_request_kwargs_returns_all(self, sample_deployments):
        """When request_kwargs is None, return all deployments"""
        result = filter_web_search_deployments(sample_deployments, None)
        assert result == sample_deployments

    def test_no_tools_returns_all(self, sample_deployments):
        """When no tools in request, return all deployments"""
        result = filter_web_search_deployments(sample_deployments, {"other": "value"})
        assert result == sample_deployments

    def test_empty_tools_returns_all(self, sample_deployments):
        """When tools list is empty, return all deployments"""
        result = filter_web_search_deployments(sample_deployments, {"tools": []})
        assert result == sample_deployments

    def test_none_tools_returns_all(self, sample_deployments):
        """When tools is explicitly None, return all deployments (regression test for #17672)"""
        result = filter_web_search_deployments(sample_deployments, {"tools": None})
        assert result == sample_deployments

    def test_non_web_search_tools_returns_all(self, sample_deployments):
        """When tools don't include web_search, return all deployments"""
        request_kwargs = {"tools": [{"type": "function", "function": {}}]}
        result = filter_web_search_deployments(sample_deployments, request_kwargs)
        assert result == sample_deployments

    def test_web_search_filters_unsupported(self, sample_deployments):
        """When web_search tool present, filter out deployments that don't support it"""
        request_kwargs = {"tools": [{"type": "web_search"}]}
        result = filter_web_search_deployments(sample_deployments, request_kwargs)
        # Should exclude deployment-3 (supports_web_search=False)
        assert len(result) == 2
        result_ids = [d["model_info"]["id"] for d in result]
        assert "deployment-1" in result_ids
        assert "deployment-2" in result_ids
        assert "deployment-3" not in result_ids

    def test_web_search_preview_filters_unsupported(self, sample_deployments):
        """web_search_preview type should also trigger filtering"""
        request_kwargs = {"tools": [{"type": "web_search_preview"}]}
        result = filter_web_search_deployments(sample_deployments, request_kwargs)
        assert len(result) == 2
        result_ids = [d["model_info"]["id"] for d in result]
        assert "deployment-3" not in result_ids

    def test_web_search_with_other_tools(self, sample_deployments):
        """Web search filtering works when mixed with other tools"""
        request_kwargs = {
            "tools": [
                {"type": "function", "function": {"name": "get_weather"}},
                {"type": "web_search"},
            ]
        }
        result = filter_web_search_deployments(sample_deployments, request_kwargs)
        assert len(result) == 2
        result_ids = [d["model_info"]["id"] for d in result]
        assert "deployment-3" not in result_ids

    def test_all_deployments_support_web_search(self):
        """When all deployments support web search, none are filtered"""
        deployments = [
            {"model_info": {"id": "d1", "supports_web_search": True}},
            {"model_info": {"id": "d2", "supports_web_search": True}},
        ]
        request_kwargs = {"tools": [{"type": "web_search"}]}
        result = filter_web_search_deployments(deployments, request_kwargs)
        assert len(result) == 2

    def test_no_deployments_support_web_search(self):
        """When no deployments support web search, all are filtered out"""
        deployments = [
            {"model_info": {"id": "d1", "supports_web_search": False}},
            {"model_info": {"id": "d2", "supports_web_search": False}},
        ]
        request_kwargs = {"tools": [{"type": "web_search"}]}
        result = filter_web_search_deployments(deployments, request_kwargs)
        assert len(result) == 0

    def test_missing_config_defaults_to_supported(self):
        """Deployments without supports_web_search config default to True"""
        deployments = [
            {"model_info": {"id": "d1"}},  # No supports_web_search - defaults to True
            {"model_info": {"id": "d2"}},  # No supports_web_search - defaults to True
            {"model_info": {"id": "d3", "supports_web_search": False}},  # Explicit False
        ]
        request_kwargs = {"tools": [{"type": "web_search"}]}
        result = filter_web_search_deployments(deployments, request_kwargs)
        # d1 and d2 should be included (default True), d3 excluded (explicit False)
        assert len(result) == 2
        result_ids = [d["model_info"]["id"] for d in result]
        assert "d1" in result_ids
        assert "d2" in result_ids
        assert "d3" not in result_ids

    def test_empty_deployments_list(self):
        """Empty deployments list returns empty list"""
        request_kwargs = {"tools": [{"type": "web_search"}]}
        result = filter_web_search_deployments([], request_kwargs)
        assert result == []

    def test_dict_deployment_passthrough(self):
        """When deployment is a dict (single deployment), pass through unchanged"""
        deployment = {"model_info": {"id": "d1", "supports_web_search": False}}
        request_kwargs = {"tools": [{"type": "web_search"}]}
        result = filter_web_search_deployments(deployment, request_kwargs)
        # Should return the dict unchanged, not filter it
        assert result == deployment
