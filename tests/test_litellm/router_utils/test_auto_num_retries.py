"""Tests that num_retries auto-matches deployment count."""
import pytest
import litellm


def test_num_retries_auto_set_to_deployment_count():
    """num_retries should default to len(deployments) - 1 for the model group."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": f"key{i}"},
                "model_info": {"id": f"dep-{i}"},
            }
            for i in range(5)
        ],
    )
    kwargs = {"model": "gpt-4"}
    router._update_kwargs_before_fallbacks(model="gpt-4", kwargs=kwargs)
    # 5 deployments → num_retries should be 4
    assert kwargs["num_retries"] == 4


def test_num_retries_explicit_override_preserved():
    """If user explicitly sets num_retries, don't override it."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": f"key{i}"},
                "model_info": {"id": f"dep-{i}"},
            }
            for i in range(5)
        ],
        num_retries=1,
    )
    kwargs = {"model": "gpt-4"}
    router._update_kwargs_before_fallbacks(model="gpt-4", kwargs=kwargs)
    # User set num_retries=1, should be preserved
    assert kwargs["num_retries"] == 1


def test_num_retries_single_deployment():
    """Single deployment: num_retries should be 0 (no point retrying same one)."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "key1"},
                "model_info": {"id": "dep-1"},
            },
        ],
    )
    kwargs = {"model": "gpt-4"}
    router._update_kwargs_before_fallbacks(model="gpt-4", kwargs=kwargs)
    assert kwargs["num_retries"] == 0
