import pytest
from litellm import Router

def test_router_fallbacks_initialization():
    fallbacks = [
        {"gpt-4": ["azure/gpt-4-east", "bedrock/claude-3-5-sonnet"]}
    ]
    router = Router(
        model_list=[
            {"model_name": "gpt-4", "litellm_params": {"model": "azure/gpt-4-east"}},
            {"model_name": "azure/gpt-4-east", "litellm_params": {"model": "azure/gpt-4-east"}},
            {"model_name": "bedrock/claude-3-5-sonnet", "litellm_params": {"model": "bedrock/claude-3-5-sonnet"}},
        ],
        fallbacks=fallbacks,
    )
    assert router.fallbacks == fallbacks
    assert len(router.fallbacks) == 1
    assert "gpt-4" in router.fallbacks[0]

def test_router_empty_fallbacks():
    router = Router(
        model_list=[
            {"model_name": "gpt-3.5-turbo", "litellm_params": {"model": "gpt-3.5-turbo"}}
        ]
    )
    assert router.fallbacks is None or router.fallbacks == []