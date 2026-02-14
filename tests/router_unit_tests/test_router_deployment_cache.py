import sys
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
from litellm import Router
from litellm.types.router import Deployment, LiteLLM_Params


@pytest.fixture
def model_list():
    return [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": "sk-test1",
            },
            "model_info": {
                "id": "model-id-1",
            },
        },
        {
            "model_name": "gpt-4o",
            "litellm_params": {
                "model": "gpt-4o",
                "api_key": "sk-test2",
            },
            "model_info": {
                "id": "model-id-2",
            },
        },
    ]


@pytest.fixture
def router(model_list):
    return Router(model_list=model_list)


def test_get_deployment_returns_correct_object(router):
    """get_deployment returns correct Deployment object."""
    deployment = router.get_deployment("model-id-1")
    assert deployment is not None
    assert isinstance(deployment, Deployment)
    assert deployment.model_name == "gpt-3.5-turbo"
    assert deployment.litellm_params.model == "gpt-3.5-turbo"


def test_get_deployment_returns_cached_object(router):
    """get_deployment returns same cached object on second call."""
    result1 = router.get_deployment("model-id-1")
    result2 = router.get_deployment("model-id-1")
    assert result1 is result2  # same object reference = cache hit


def test_get_deployment_by_model_group_name_returns_correct_object(router):
    """get_deployment_by_model_group_name returns correct Deployment."""
    deployment = router.get_deployment_by_model_group_name("gpt-3.5-turbo")
    assert deployment is not None
    assert isinstance(deployment, Deployment)
    assert deployment.model_name == "gpt-3.5-turbo"


def test_add_deployment_invalidates_cache(router):
    """add_deployment invalidates cache so new deployments are found."""
    # Populate cache
    router.get_deployment("model-id-1")

    # Add a new deployment
    new_deployment = Deployment(
        model_name="gpt-4",
        litellm_params=LiteLLM_Params(model="gpt-4", api_key="sk-test3"),
        model_info={"id": "model-id-3"},
    )
    router.add_deployment(deployment=new_deployment)

    # New deployment should be found
    result = router.get_deployment("model-id-3")
    assert result is not None
    assert result.model_name == "gpt-4"


def test_delete_deployment_invalidates_cache(router):
    """delete_deployment invalidates cache so deleted deployments return None."""
    # Populate cache
    result = router.get_deployment("model-id-1")
    assert result is not None

    # Delete the deployment
    router.delete_deployment(id="model-id-1")

    # Should return None now
    result = router.get_deployment("model-id-1")
    assert result is None


def test_upsert_deployment_invalidates_cache(router):
    """upsert_deployment invalidates cache so updated deployments are returned."""
    # Populate cache
    original = router.get_deployment("model-id-1")
    assert original is not None
    assert original.litellm_params.model == "gpt-3.5-turbo"

    # Upsert with changed api_key
    updated_deployment = Deployment(
        model_name="gpt-3.5-turbo",
        litellm_params=LiteLLM_Params(
            model="gpt-3.5-turbo", api_key="sk-test-updated"
        ),
        model_info={"id": "model-id-1"},
    )
    router.upsert_deployment(deployment=updated_deployment)

    # Should return updated deployment
    result = router.get_deployment("model-id-1")
    assert result is not None
    assert result.litellm_params.api_key == "sk-test-updated"


def test_set_model_list_invalidates_cache(router):
    """set_model_list invalidates cache completely."""
    # Populate cache
    router.get_deployment("model-id-1")
    router.get_deployment("model-id-2")

    # Set completely new model list
    router.set_model_list(
        [
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                    "api_key": "sk-new",
                },
                "model_info": {
                    "id": "new-model-id",
                },
            }
        ]
    )

    # Old model IDs should return None
    assert router.get_deployment("model-id-1") is None
    assert router.get_deployment("model-id-2") is None

    # New model ID should work
    result = router.get_deployment("new-model-id")
    assert result is not None
    assert result.model_name == "gpt-4"


def test_get_deployment_invalid_model_id_returns_none(router):
    """get_deployment with invalid model_id returns None."""
    result = router.get_deployment("nonexistent-model-id")
    assert result is None
