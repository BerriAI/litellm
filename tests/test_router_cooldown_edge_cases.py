import pytest
import time
from litellm import Router

def test_router_model_list_initialization():
    model_list = [
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "azure/gpt-4-east",
                "api_key": "test-key-1",
                "api_base": "https://test.azure.com",
            },
        },
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "azure/gpt-4-west",
                "api_key": "test-key-2",
                "api_base": "https://test.azure.com",
            },
        },
    ]
    
    router = Router(model_list=model_list)
    assert len(router.model_list) == 2
    assert router.model_list[0]["model_name"] == "gpt-4"
    assert router.model_list[1]["model_name"] == "gpt-4"

def test_router_cooldown_time_assignment():
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "test-key"},
            }
        ],
        cooldown_time=45,
    )
    assert router.cooldown_time == 45

def test_router_get_model_group_names():
    router = Router(
        model_list=[
            {"model_name": "claude-3-haiku", "litellm_params": {"model": "bedrock/claude-3-haiku"}},
            {"model_name": "claude-3-sonnet", "litellm_params": {"model": "bedrock/claude-3-sonnet"}},
        ]
    )
    model_names = [m["model_name"] for m in router.model_list]
    assert "claude-3-haiku" in model_names
    assert "claude-3-sonnet" in model_names