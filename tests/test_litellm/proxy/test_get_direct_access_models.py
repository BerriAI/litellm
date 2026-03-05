"""
Tests for get_direct_access_models in proxy_server.py.

Verifies that users provisioned with "all-proxy-models" see all models
on the Models and Endpoints page. Fixes #22791.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._types import LiteLLM_UserTable, SpecialModelNames
from litellm.proxy.proxy_server import get_direct_access_models


def _make_mock_router(model_list):
    """
    Create a mock Router with the given model_list.

    get_model_list(model_name=name) filters by model_name.
    get_model_ids(exclude_team_models=True) returns all non-team model IDs.
    """
    router = MagicMock()
    router.model_list = model_list

    def _get_model_list(model_name=None, team_id=None):
        if model_name is None:
            return model_list
        return [
            m for m in model_list if m.get("model_name") == model_name
        ] or None

    router.get_model_list = MagicMock(side_effect=_get_model_list)

    def _get_model_ids(model_name=None, exclude_team_models=False):
        ids = []
        for m in model_list:
            mid = m.get("model_info", {}).get("id")
            if mid is None:
                continue
            if exclude_team_models and m.get("model_info", {}).get("team_id"):
                continue
            ids.append(mid)
        return ids

    router.get_model_ids = MagicMock(side_effect=_get_model_ids)
    return router


SAMPLE_MODELS = [
    {
        "model_name": "gpt-4",
        "litellm_params": {"model": "gpt-4"},
        "model_info": {"id": "model-id-1"},
    },
    {
        "model_name": "claude-3",
        "litellm_params": {"model": "claude-3-opus"},
        "model_info": {"id": "model-id-2"},
    },
    {
        "model_name": "codestral",
        "litellm_params": {"model": "codestral-latest"},
        "model_info": {"id": "model-id-3"},
    },
    {
        "model_name": "team-only-model",
        "litellm_params": {"model": "team-model"},
        "model_info": {"id": "model-id-4", "team_id": "team-abc"},
    },
]


class TestGetDirectAccessModels:
    """Test suite for get_direct_access_models."""

    def test_all_proxy_models_returns_all_non_team_model_ids(self):
        """
        When user has 'all-proxy-models' in their models list,
        get_direct_access_models should return all non-team model IDs.
        This is the fix for issue #22791.
        """
        user = LiteLLM_UserTable(
            user_id="user-1",
            models=[SpecialModelNames.all_proxy_models.value],
        )
        router = _make_mock_router(SAMPLE_MODELS)

        result = get_direct_access_models(user, router)

        # Should return all 3 non-team model IDs
        assert set(result) == {"model-id-1", "model-id-2", "model-id-3"}
        # Should NOT include the team-only model
        assert "model-id-4" not in result
        # Verify it called get_model_ids correctly
        router.get_model_ids.assert_called_once_with(exclude_team_models=True)

    def test_specific_models_returns_only_matching_ids(self):
        """
        When user has specific model names, only those models'
        deployment IDs should be returned.
        """
        user = LiteLLM_UserTable(
            user_id="user-2",
            models=["gpt-4", "codestral"],
        )
        router = _make_mock_router(SAMPLE_MODELS)

        result = get_direct_access_models(user, router)

        assert set(result) == {"model-id-1", "model-id-3"}
        # Should NOT have called get_model_ids
        router.get_model_ids.assert_not_called()

    def test_empty_models_list_returns_empty(self):
        """
        When user has no models, return empty list.
        """
        user = LiteLLM_UserTable(
            user_id="user-3",
            models=[],
        )
        router = _make_mock_router(SAMPLE_MODELS)

        result = get_direct_access_models(user, router)

        assert result == []

    def test_nonexistent_model_name_returns_empty(self):
        """
        When user has a model name that doesn't match any deployments,
        it should be skipped.
        """
        user = LiteLLM_UserTable(
            user_id="user-4",
            models=["nonexistent-model"],
        )
        router = _make_mock_router(SAMPLE_MODELS)

        result = get_direct_access_models(user, router)

        assert result == []

    def test_all_proxy_models_with_other_models(self):
        """
        When user has 'all-proxy-models' alongside other model names,
        'all-proxy-models' should take precedence and return all models.
        """
        user = LiteLLM_UserTable(
            user_id="user-5",
            models=[SpecialModelNames.all_proxy_models.value, "gpt-4"],
        )
        router = _make_mock_router(SAMPLE_MODELS)

        result = get_direct_access_models(user, router)

        # Should still return all non-team model IDs
        assert set(result) == {"model-id-1", "model-id-2", "model-id-3"}
