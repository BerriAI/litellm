"""
Unit tests for weighted routing with simple-shuffle strategy.

Regression test for: https://linear.app/litellm-ai/issue/LIT-1795
Bug: When model_info.id matches model_name, the router treats requests
as targeting a specific deployment instead of routing across all deployments.
"""

import os
import sys
from collections import Counter
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm import Router


def _make_model_list(with_model_info_id: bool, id_matches_model_name: bool = False):
    """Helper to build model_list with or without model_info.id."""
    model_list = [
        {
            "model_name": "test-model",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "fake-key-1",
                "weight": 2,
            },
        },
        {
            "model_name": "test-model",
            "litellm_params": {
                "model": "openai/gpt-3.5-turbo",
                "api_key": "fake-key-2",
                "weight": 1,
            },
        },
    ]
    if with_model_info_id:
        if id_matches_model_name:
            model_list[0]["model_info"] = {"id": "test-model"}
        else:
            model_list[0]["model_info"] = {"id": "deployment-1"}
        model_list[1]["model_info"] = {"id": "deployment-2"}
    return model_list


@pytest.mark.asyncio
async def test_weighted_routing_with_model_info_id_matching_model_name():
    """
    Regression test for LIT-1795: when model_info.id equals model_name,
    the router should still return multiple deployments for weighted routing,
    not treat the request as targeting a specific deployment.
    """
    router = Router(
        model_list=_make_model_list(
            with_model_info_id=True, id_matches_model_name=True
        ),
        routing_strategy="simple-shuffle",
    )

    # _common_checks_available_deployment should return a list (multiple deployments),
    # not a single dict (specific deployment)
    _, deployments = router._common_checks_available_deployment(model="test-model")
    assert isinstance(deployments, list), (
        "Expected list of deployments for weighted routing, "
        f"got {type(deployments).__name__} (single deployment selected). "
        "This means model_info.id matching model_name caused specific deployment selection."
    )
    assert len(deployments) == 2, (
        f"Expected 2 deployments, got {len(deployments)}"
    )


@pytest.mark.asyncio
async def test_weighted_routing_with_model_info_id_not_matching_model_name():
    """
    When model_info.id does NOT match model_name, routing should also
    return multiple deployments.
    """
    router = Router(
        model_list=_make_model_list(
            with_model_info_id=True, id_matches_model_name=False
        ),
        routing_strategy="simple-shuffle",
    )

    _, deployments = router._common_checks_available_deployment(model="test-model")
    assert isinstance(deployments, list)
    assert len(deployments) == 2


@pytest.mark.asyncio
async def test_weighted_routing_without_model_info_id():
    """
    Without model_info.id (auto-generated IDs), routing should work as expected.
    """
    router = Router(
        model_list=_make_model_list(with_model_info_id=False),
        routing_strategy="simple-shuffle",
    )

    _, deployments = router._common_checks_available_deployment(model="test-model")
    assert isinstance(deployments, list)
    assert len(deployments) == 2


@pytest.mark.asyncio
async def test_specific_deployment_by_id_still_works():
    """
    Requesting a model by its model_info.id (when that ID is NOT a model_name)
    should still return a single specific deployment.
    """
    router = Router(
        model_list=_make_model_list(
            with_model_info_id=True, id_matches_model_name=False
        ),
        routing_strategy="simple-shuffle",
    )

    # Request by deployment ID directly - should get single deployment
    _, deployment = router._common_checks_available_deployment(model="deployment-1")
    assert isinstance(deployment, dict), (
        "Requesting by deployment ID should return a single deployment dict"
    )
