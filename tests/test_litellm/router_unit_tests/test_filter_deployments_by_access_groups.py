"""
Unit tests for filter_deployments_by_access_groups function.

Tests the fix for GitHub issue #18333: Models loadbalanced outside of Model Access Group.
"""

import pytest

from litellm.router_utils.common_utils import filter_deployments_by_access_groups


class TestFilterDeploymentsByAccessGroups:
    """Tests for the filter_deployments_by_access_groups function."""

    def test_no_filter_when_no_access_groups_in_metadata(self):
        """When no allowed_access_groups in metadata, return all deployments."""
        deployments = [
            {"model_info": {"id": "1", "access_groups": ["AG1"]}},
            {"model_info": {"id": "2", "access_groups": ["AG2"]}},
        ]
        request_kwargs = {"metadata": {"user_api_key_team_id": "team-1"}}

        result = filter_deployments_by_access_groups(
            healthy_deployments=deployments,
            request_kwargs=request_kwargs,
        )

        assert len(result) == 2  # All deployments returned

    def test_filter_to_single_access_group(self):
        """Filter to only deployments matching allowed access group."""
        deployments = [
            {"model_info": {"id": "1", "access_groups": ["AG1"]}},
            {"model_info": {"id": "2", "access_groups": ["AG2"]}},
        ]
        request_kwargs = {"metadata": {"user_api_key_allowed_access_groups": ["AG2"]}}

        result = filter_deployments_by_access_groups(
            healthy_deployments=deployments,
            request_kwargs=request_kwargs,
        )

        assert len(result) == 1
        assert result[0]["model_info"]["id"] == "2"

    def test_filter_with_multiple_allowed_groups(self):
        """Filter with multiple allowed access groups."""
        deployments = [
            {"model_info": {"id": "1", "access_groups": ["AG1"]}},
            {"model_info": {"id": "2", "access_groups": ["AG2"]}},
            {"model_info": {"id": "3", "access_groups": ["AG3"]}},
        ]
        request_kwargs = {
            "metadata": {"user_api_key_allowed_access_groups": ["AG1", "AG2"]}
        }

        result = filter_deployments_by_access_groups(
            healthy_deployments=deployments,
            request_kwargs=request_kwargs,
        )

        assert len(result) == 2
        ids = [d["model_info"]["id"] for d in result]
        assert "1" in ids
        assert "2" in ids
        assert "3" not in ids

    def test_deployment_with_multiple_access_groups(self):
        """Deployment with multiple access groups should match if any overlap."""
        deployments = [
            {"model_info": {"id": "1", "access_groups": ["AG1", "AG2"]}},
            {"model_info": {"id": "2", "access_groups": ["AG3"]}},
        ]
        request_kwargs = {"metadata": {"user_api_key_allowed_access_groups": ["AG2"]}}

        result = filter_deployments_by_access_groups(
            healthy_deployments=deployments,
            request_kwargs=request_kwargs,
        )

        assert len(result) == 1
        assert result[0]["model_info"]["id"] == "1"

    def test_deployment_without_access_groups_included(self):
        """Deployments without access groups should be included (not restricted)."""
        deployments = [
            {"model_info": {"id": "1", "access_groups": ["AG1"]}},
            {"model_info": {"id": "2"}},  # No access_groups
            {"model_info": {"id": "3", "access_groups": []}},  # Empty access_groups
        ]
        request_kwargs = {"metadata": {"user_api_key_allowed_access_groups": ["AG2"]}}

        result = filter_deployments_by_access_groups(
            healthy_deployments=deployments,
            request_kwargs=request_kwargs,
        )

        # Should include deployments 2 and 3 (no restrictions)
        assert len(result) == 2
        ids = [d["model_info"]["id"] for d in result]
        assert "2" in ids
        assert "3" in ids

    def test_dict_deployment_passes_through(self):
        """When deployment is a dict (specific deployment), pass through."""
        deployment = {"model_info": {"id": "1", "access_groups": ["AG1"]}}
        request_kwargs = {"metadata": {"user_api_key_allowed_access_groups": ["AG2"]}}

        result = filter_deployments_by_access_groups(
            healthy_deployments=deployment,
            request_kwargs=request_kwargs,
        )

        assert result == deployment  # Unchanged

    def test_none_request_kwargs_passes_through(self):
        """When request_kwargs is None, return deployments unchanged."""
        deployments = [
            {"model_info": {"id": "1", "access_groups": ["AG1"]}},
        ]

        result = filter_deployments_by_access_groups(
            healthy_deployments=deployments,
            request_kwargs=None,
        )

        assert result == deployments

    def test_litellm_metadata_fallback(self):
        """Should also check litellm_metadata for allowed access groups."""
        deployments = [
            {"model_info": {"id": "1", "access_groups": ["AG1"]}},
            {"model_info": {"id": "2", "access_groups": ["AG2"]}},
        ]
        request_kwargs = {
            "litellm_metadata": {"user_api_key_allowed_access_groups": ["AG1"]}
        }

        result = filter_deployments_by_access_groups(
            healthy_deployments=deployments,
            request_kwargs=request_kwargs,
        )

        assert len(result) == 1
        assert result[0]["model_info"]["id"] == "1"


def test_filter_deployments_by_access_groups_issue_18333():
    """
    Regression test for GitHub issue #18333.

    Scenario: Two models named 'gpt-5' in different access groups (AG1, AG2).
    Team2 has access to AG2 only. When Team2 requests 'gpt-5', only the AG2
    deployment should be available for load balancing.
    """
    deployments = [
        {
            "model_name": "gpt-5",
            "litellm_params": {"model": "gpt-4.1", "api_key": "key-1"},
            "model_info": {"id": "ag1-deployment", "access_groups": ["AG1"]},
        },
        {
            "model_name": "gpt-5",
            "litellm_params": {"model": "gpt-4o", "api_key": "key-2"},
            "model_info": {"id": "ag2-deployment", "access_groups": ["AG2"]},
        },
    ]

    # Team2's request with allowed access groups
    request_kwargs = {
        "metadata": {
            "user_api_key_team_id": "team-2",
            "user_api_key_allowed_access_groups": ["AG2"],
        }
    }

    result = filter_deployments_by_access_groups(
        healthy_deployments=deployments,
        request_kwargs=request_kwargs,
    )

    # Only AG2 deployment should be returned
    assert len(result) == 1
    assert result[0]["model_info"]["id"] == "ag2-deployment"
    assert result[0]["litellm_params"]["model"] == "gpt-4o"


def test_get_access_groups_from_models():
    """
    Test the helper function that extracts access group names from models list.
    This is used by the proxy to populate user_api_key_allowed_access_groups.
    """
    from litellm.proxy.auth.model_checks import get_access_groups_from_models

    # Setup: access groups definition
    model_access_groups = {
        "AG1": ["gpt-4", "gpt-5"],
        "AG2": ["claude-v1", "claude-v2"],
        "beta-models": ["gpt-5-turbo"],
    }

    # Test 1: Extract access groups from models list
    models = ["gpt-4", "AG1", "AG2", "some-other-model"]
    result = get_access_groups_from_models(
        model_access_groups=model_access_groups, models=models
    )
    assert set(result) == {"AG1", "AG2"}

    # Test 2: No access groups in models list
    models = ["gpt-4", "claude-v1", "some-model"]
    result = get_access_groups_from_models(
        model_access_groups=model_access_groups, models=models
    )
    assert result == []

    # Test 3: Empty models list
    result = get_access_groups_from_models(
        model_access_groups=model_access_groups, models=[]
    )
    assert result == []

    # Test 4: All access groups
    models = ["AG1", "AG2", "beta-models"]
    result = get_access_groups_from_models(
        model_access_groups=model_access_groups, models=models
    )
    assert set(result) == {"AG1", "AG2", "beta-models"}
