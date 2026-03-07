"""
Tests for router.upsert_deployment() model name change detection.

Regression tests for https://github.com/BerriAI/litellm/issues/22190
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.types.router import Deployment, LiteLLM_Params


def test_upsert_deployment_detects_model_name_change():
    """
    Test that upsert_deployment updates a deployment when model_name changes
    but litellm_params stay the same.

    Regression test for https://github.com/BerriAI/litellm/issues/22190
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "original-name",
                "litellm_params": {
                    "model": "openai/fake-model",
                    "api_key": "fake-key",
                    "api_base": "https://example.com",
                },
            },
        ],
    )

    # Get the deployment to find its model_id
    deployment = router.get_deployment_by_model_group_name(
        model_group_name="original-name"
    )
    model_id = deployment.model_info.id

    # Upsert with same litellm_params but different model_name
    result = router.upsert_deployment(
        deployment=Deployment(
            model_name="updated-name",
            litellm_params=deployment.litellm_params,
            model_info={"id": model_id},
        )
    )

    # Should return the deployment (not None) since the name changed
    assert result is not None

    # Router should still have exactly 1 deployment
    assert len(router.model_list) == 1

    # The deployment should have the updated name
    updated = router.get_deployment(model_id=model_id)
    assert updated is not None
    assert updated.model_name == "updated-name"


def test_upsert_deployment_skips_when_nothing_changed():
    """
    Test that upsert_deployment returns None when neither model_name
    nor litellm_params have changed.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "my-model",
                "litellm_params": {
                    "model": "openai/fake-model",
                    "api_key": "fake-key",
                    "api_base": "https://example.com",
                },
            },
        ],
    )

    deployment = router.get_deployment_by_model_group_name(
        model_group_name="my-model"
    )
    model_id = deployment.model_info.id

    # Upsert with identical name and params
    result = router.upsert_deployment(
        deployment=Deployment(
            model_name="my-model",
            litellm_params=deployment.litellm_params,
            model_info={"id": model_id},
        )
    )

    # Should return None since nothing changed
    assert result is None
    assert len(router.model_list) == 1


def test_upsert_deployment_detects_litellm_params_change():
    """
    Test that upsert_deployment updates a deployment when litellm_params
    change but model_name stays the same.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "my-model",
                "litellm_params": {
                    "model": "openai/fake-model",
                    "api_key": "old-key",
                    "api_base": "https://example.com",
                },
            },
        ],
    )

    deployment = router.get_deployment_by_model_group_name(
        model_group_name="my-model"
    )
    model_id = deployment.model_info.id

    # Upsert with same name but different api_key
    result = router.upsert_deployment(
        deployment=Deployment(
            model_name="my-model",
            litellm_params=LiteLLM_Params(
                model="openai/fake-model",
                api_key="new-key",
                api_base="https://example.com",
            ),
            model_info={"id": model_id},
        )
    )

    # Should return the deployment since params changed
    assert result is not None
    assert len(router.model_list) == 1
