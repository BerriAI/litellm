"""
Tests for filter_deployments_by_access_groups function.

Verifies that when a virtual key accesses a model through an access group,
only deployments belonging to the key's access groups are routed to.

Relates to: https://github.com/BerriAI/litellm/issues/21935
"""

from typing import Dict, List, Optional
from unittest.mock import Mock

import pytest

from litellm.router_utils.common_utils import filter_deployments_by_access_groups


def _make_auth(
    models: Optional[List[str]] = None,
    access_group_ids: Optional[List[str]] = None,
) -> Mock:
    """Helper to create a mock UserAPIKeyAuth object."""
    auth = Mock()
    auth.models = models or []
    auth.access_group_ids = access_group_ids or []
    return auth


class TestFilterDeploymentsByAccessGroups:
    """Test cases for filter_deployments_by_access_groups function"""

    @pytest.fixture
    def deployments_with_access_groups(self) -> List[Dict]:
        """
        Two deployments with the same model name but different access groups.
        This is the exact scenario from Issue #21935.
        """
        return [
            {
                "model_info": {
                    "id": "deployment-dev",
                    "access_groups": ["dev_models"],
                },
                "litellm_params": {"model": "azure/gpt-4o-eastus"},
                "model_name": "gpt-4o",
            },
            {
                "model_info": {
                    "id": "deployment-prod",
                    "access_groups": ["prod_models"],
                },
                "litellm_params": {"model": "azure/gpt-4o-westus"},
                "model_name": "gpt-4o",
            },
        ]

    @pytest.fixture
    def deployments_mixed(self) -> List[Dict]:
        """Deployments with a mix of access groups and unrestricted."""
        return [
            {
                "model_info": {
                    "id": "deployment-dev",
                    "access_groups": ["dev_models"],
                },
                "model_name": "gpt-4o",
            },
            {
                "model_info": {
                    "id": "deployment-prod",
                    "access_groups": ["prod_models"],
                },
                "model_name": "gpt-4o",
            },
            {
                "model_info": {
                    "id": "deployment-unrestricted",
                    # No access_groups — should always be kept
                },
                "model_name": "gpt-4o",
            },
        ]

    # ── Core Issue #21935 scenario ──────────────────────────────────────

    def test_key_with_dev_access_group_only_gets_dev_deployment(
        self, deployments_with_access_groups
    ):
        """
        Key 1 has access_group_ids=["dev_models"].
        Should only route to deployment-dev, not deployment-prod.
        """
        request_kwargs = {
            "metadata": {
                "user_api_key_auth": _make_auth(access_group_ids=["dev_models"])
            }
        }
        result = filter_deployments_by_access_groups(
            model="gpt-4o",
            healthy_deployments=deployments_with_access_groups,
            request_kwargs=request_kwargs,
        )
        result_ids = [d["model_info"]["id"] for d in result]
        assert result_ids == ["deployment-dev"]

    def test_key_with_prod_access_group_only_gets_prod_deployment(
        self, deployments_with_access_groups
    ):
        """
        Key 2 has access_group_ids=["prod_models"].
        Should only route to deployment-prod, not deployment-dev.
        """
        request_kwargs = {
            "metadata": {
                "user_api_key_auth": _make_auth(access_group_ids=["prod_models"])
            }
        }
        result = filter_deployments_by_access_groups(
            model="gpt-4o",
            healthy_deployments=deployments_with_access_groups,
            request_kwargs=request_kwargs,
        )
        result_ids = [d["model_info"]["id"] for d in result]
        assert result_ids == ["deployment-prod"]

    # ── Access group in models list (alternative config) ────────────────

    def test_access_group_in_models_list(self, deployments_with_access_groups):
        """
        Key has models=["dev_models"] (access group name in models list).
        Should only route to deployment-dev.
        """
        request_kwargs = {
            "metadata": {"user_api_key_auth": _make_auth(models=["dev_models"])}
        }
        result = filter_deployments_by_access_groups(
            model="gpt-4o",
            healthy_deployments=deployments_with_access_groups,
            request_kwargs=request_kwargs,
        )
        result_ids = [d["model_info"]["id"] for d in result]
        assert result_ids == ["deployment-dev"]

    # ── Multiple access groups ──────────────────────────────────────────

    def test_key_with_multiple_access_groups(self, deployments_with_access_groups):
        """
        Key has access to both dev_models and prod_models.
        Should get both deployments.
        """
        request_kwargs = {
            "metadata": {
                "user_api_key_auth": _make_auth(
                    access_group_ids=["dev_models", "prod_models"]
                )
            }
        }
        result = filter_deployments_by_access_groups(
            model="gpt-4o",
            healthy_deployments=deployments_with_access_groups,
            request_kwargs=request_kwargs,
        )
        result_ids = sorted([d["model_info"]["id"] for d in result])
        assert result_ids == ["deployment-dev", "deployment-prod"]

    # ── Direct model access (no filtering) ──────────────────────────────

    def test_direct_model_access_no_filtering(self, deployments_with_access_groups):
        """
        Key has models=["gpt-4o"] (direct model name).
        Should get ALL deployments — no access group filtering.
        """
        request_kwargs = {
            "metadata": {"user_api_key_auth": _make_auth(models=["gpt-4o"])}
        }
        result = filter_deployments_by_access_groups(
            model="gpt-4o",
            healthy_deployments=deployments_with_access_groups,
            request_kwargs=request_kwargs,
        )
        assert len(result) == 2

    # ── Wildcard access (no filtering) ──────────────────────────────────

    def test_wildcard_star_no_filtering(self, deployments_with_access_groups):
        """Key with models=["*"] should get all deployments."""
        request_kwargs = {"metadata": {"user_api_key_auth": _make_auth(models=["*"])}}
        result = filter_deployments_by_access_groups(
            model="gpt-4o",
            healthy_deployments=deployments_with_access_groups,
            request_kwargs=request_kwargs,
        )
        assert len(result) == 2

    def test_all_proxy_models_no_filtering(self, deployments_with_access_groups):
        """Key with models=["all-proxy-models"] should get all deployments."""
        request_kwargs = {
            "metadata": {"user_api_key_auth": _make_auth(models=["all-proxy-models"])}
        }
        result = filter_deployments_by_access_groups(
            model="gpt-4o",
            healthy_deployments=deployments_with_access_groups,
            request_kwargs=request_kwargs,
        )
        assert len(result) == 2

    # ── Unrestricted deployments ────────────────────────────────────────

    def test_unrestricted_deployments_always_kept(self, deployments_mixed):
        """
        Deployments without access_groups should always be kept,
        even when key has access group restrictions.
        """
        request_kwargs = {
            "metadata": {
                "user_api_key_auth": _make_auth(access_group_ids=["dev_models"])
            }
        }
        result = filter_deployments_by_access_groups(
            model="gpt-4o",
            healthy_deployments=deployments_mixed,
            request_kwargs=request_kwargs,
        )
        result_ids = sorted([d["model_info"]["id"] for d in result])
        # dev deployment + unrestricted deployment
        assert result_ids == ["deployment-dev", "deployment-unrestricted"]

    # ── No access groups on any deployment ──────────────────────────────

    def test_no_access_groups_on_deployments(self):
        """When no deployments use access groups, return all."""
        deployments = [
            {"model_info": {"id": "d1"}, "model_name": "gpt-4o"},
            {"model_info": {"id": "d2"}, "model_name": "gpt-4o"},
        ]
        request_kwargs = {
            "metadata": {
                "user_api_key_auth": _make_auth(access_group_ids=["dev_models"])
            }
        }
        result = filter_deployments_by_access_groups(
            model="gpt-4o",
            healthy_deployments=deployments,
            request_kwargs=request_kwargs,
        )
        assert len(result) == 2

    # ── Edge cases ──────────────────────────────────────────────────────

    def test_none_request_kwargs(self, deployments_with_access_groups):
        """When request_kwargs is None, return all deployments unchanged."""
        result = filter_deployments_by_access_groups(
            model="gpt-4o",
            healthy_deployments=deployments_with_access_groups,
            request_kwargs=None,
        )
        assert result == deployments_with_access_groups

    def test_dict_deployment_passthrough(self):
        """When deployment is a dict (single chosen deployment), pass through."""
        deployment = {"model_info": {"id": "d1", "access_groups": ["dev_models"]}}
        result = filter_deployments_by_access_groups(
            model="gpt-4o",
            healthy_deployments=deployment,
            request_kwargs={
                "metadata": {
                    "user_api_key_auth": _make_auth(access_group_ids=["prod_models"])
                }
            },
        )
        assert result == deployment

    def test_empty_deployments(self):
        """Empty deployments list returns empty list."""
        result = filter_deployments_by_access_groups(
            model="gpt-4o",
            healthy_deployments=[],
            request_kwargs={
                "metadata": {
                    "user_api_key_auth": _make_auth(access_group_ids=["dev_models"])
                }
            },
        )
        assert result == []

    def test_no_auth_in_metadata(self, deployments_with_access_groups):
        """When user_api_key_auth is missing from metadata, return all."""
        result = filter_deployments_by_access_groups(
            model="gpt-4o",
            healthy_deployments=deployments_with_access_groups,
            request_kwargs={"metadata": {}},
        )
        assert len(result) == 2

    def test_empty_models_and_access_groups(self, deployments_with_access_groups):
        """Key with empty models and access_group_ids returns all."""
        request_kwargs = {
            "metadata": {
                "user_api_key_auth": _make_auth(models=[], access_group_ids=[])
            }
        }
        result = filter_deployments_by_access_groups(
            model="gpt-4o",
            healthy_deployments=deployments_with_access_groups,
            request_kwargs=request_kwargs,
        )
        assert len(result) == 2

    def test_litellm_metadata_fallback(self, deployments_with_access_groups):
        """Auth object in litellm_metadata (used for some endpoints) is also read."""
        request_kwargs = {
            "litellm_metadata": {
                "user_api_key_auth": _make_auth(access_group_ids=["dev_models"])
            }
        }
        result = filter_deployments_by_access_groups(
            model="gpt-4o",
            healthy_deployments=deployments_with_access_groups,
            request_kwargs=request_kwargs,
        )
        result_ids = [d["model_info"]["id"] for d in result]
        assert result_ids == ["deployment-dev"]

    def test_deployment_in_multiple_access_groups(self):
        """Deployment belonging to multiple groups matches if key has any of them."""
        deployments = [
            {
                "model_info": {
                    "id": "deployment-multi",
                    "access_groups": ["dev_models", "staging_models"],
                },
                "model_name": "gpt-4o",
            },
            {
                "model_info": {
                    "id": "deployment-prod",
                    "access_groups": ["prod_models"],
                },
                "model_name": "gpt-4o",
            },
        ]
        request_kwargs = {
            "metadata": {
                "user_api_key_auth": _make_auth(access_group_ids=["staging_models"])
            }
        }
        result = filter_deployments_by_access_groups(
            model="gpt-4o",
            healthy_deployments=deployments,
            request_kwargs=request_kwargs,
        )
        result_ids = [d["model_info"]["id"] for d in result]
        assert result_ids == ["deployment-multi"]

    def test_returns_empty_list_when_no_matching_deployments(self):
        """
        If filtering removes ALL deployments (key's access group doesn't match
        any deployment's access group), return empty list so downstream routing
        raises "no healthy deployments" — not silently bypass access control.
        """
        # Only a prod_models deployment exists
        deployments = [
            {
                "model_info": {
                    "id": "deployment-prod",
                    "access_groups": ["prod_models"],
                },
                "model_name": "gpt-4o",
            },
        ]
        # Key has access only to dev_models — no overlap with prod_models
        request_kwargs = {
            "metadata": {
                "user_api_key_auth": _make_auth(access_group_ids=["dev_models"])
            }
        }
        # dev_models is a known group (it's in key_access_groups via access_group_ids),
        # but no deployment belongs to it → filtered list is empty
        result = filter_deployments_by_access_groups(
            model="gpt-4o",
            healthy_deployments=deployments,
            request_kwargs=request_kwargs,
        )
        # Must return empty list — not the original list — to let routing fail
        # with "no healthy deployments" and surface the misconfiguration
        assert result == []

    def test_mixed_models_and_access_group_ids(self):
        """
        Key has both direct model names and access_group_ids.
        When calling a model that's NOT in direct list but IS via access group,
        filtering should apply.
        """
        deployments = [
            {
                "model_info": {
                    "id": "deployment-dev",
                    "access_groups": ["dev_models"],
                },
                "model_name": "gpt-4o",
            },
            {
                "model_info": {
                    "id": "deployment-prod",
                    "access_groups": ["prod_models"],
                },
                "model_name": "gpt-4o",
            },
        ]
        # Key has direct access to gpt-3.5-turbo and access to dev_models group
        request_kwargs = {
            "metadata": {
                "user_api_key_auth": _make_auth(
                    models=["gpt-3.5-turbo"],
                    access_group_ids=["dev_models"],
                )
            }
        }
        result = filter_deployments_by_access_groups(
            model="gpt-4o",  # not in direct models list → access group filter applies
            healthy_deployments=deployments,
            request_kwargs=request_kwargs,
        )
        result_ids = [d["model_info"]["id"] for d in result]
        assert result_ids == ["deployment-dev"]

    def test_missing_model_info(self):
        """Deployments with missing model_info are handled gracefully."""
        deployments = [
            {
                "model_info": {
                    "id": "deployment-dev",
                    "access_groups": ["dev_models"],
                },
                "model_name": "gpt-4o",
            },
            {
                # Missing model_info entirely
                "model_name": "gpt-4o",
            },
            {
                "model_info": {
                    "id": "deployment-prod",
                    "access_groups": ["prod_models"],
                },
                "model_name": "gpt-4o",
            },
        ]
        request_kwargs = {
            "metadata": {
                "user_api_key_auth": _make_auth(access_group_ids=["dev_models"])
            }
        }
        result = filter_deployments_by_access_groups(
            model="gpt-4o",
            healthy_deployments=deployments,
            request_kwargs=request_kwargs,
        )
        result_ids = [d.get("model_info", {}).get("id") for d in result]
        # deployment-dev (matches) + deployment without model_info (unrestricted)
        assert "deployment-dev" in result_ids
        assert None in result_ids  # the one without model_info
        assert "deployment-prod" not in result_ids
