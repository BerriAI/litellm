"""Unit tests verifying that Router._core_get_router_model_info does not crash with ModelNotMappedError

when a deployment specifies an unmapped/custom model name with user-defined model_info.
Regression test for https://github.com/BerriAI/litellm/issues/51452.
"""

from litellm import Router
from litellm.router_strategy.complexity_router.context_compaction import _budget


def test_router_model_info_unmapped_model_with_user_model_info():
    """Ensure router_model_info gracefully falls back to user-configured model_info

    when the model is not present in litellm's static pricing/context window catalog.
    """
    router = Router(
        model_list=[
            {
                "model_name": "custom-unmapped-model",
                "litellm_params": {
                    "model": "ollama_chat/my-completely-custom-qwen-variant:latest",
                    "api_base": "http://localhost:11434",
                },
                "model_info": {
                    "max_input_tokens": 16384,
                    "max_output_tokens": 4096,
                    "max_tokens": 4096,
                    "mode": "chat",
                    "input_cost_per_token": 0.0001,
                    "output_cost_per_token": 0.0002,
                },
            }
        ]
    )

    deployment = router.get_available_deployment("custom-unmapped-model")
    assert deployment is not None

    # router_model_info must succeed without raising ModelNotMappedError
    info = router.registry().router_model_info(deployment=deployment, received_model_name="custom-unmapped-model")

    assert info is not None
    assert info["max_input_tokens"] == 16384
    assert info["max_output_tokens"] == 4096
    assert info["mode"] == "chat"
    assert info["input_cost_per_token"] == 0.0001
    assert info["output_cost_per_token"] == 0.0002
    assert info["litellm_provider"] == "ollama_chat"

    # Context compaction _budget must also succeed using the user-defined limits
    payload = {
        "model": "custom-unmapped-model",
        "messages": [{"role": "user", "content": "test message"}],
    }
    budget = _budget(router._core_ctx, deployment, payload, 0.9)
    assert budget.window == 16384
    # available = int(16384 * 0.9) - 4096 = 14745 - 4096 = 10649
    assert budget.available == int(16384 * 0.9) - 4096


def test_router_model_info_unmapped_model_without_user_model_info():
    """Ensure router_model_info handles unmapped models without model_info without crashing."""
    router = Router(
        model_list=[
            {
                "model_name": "unmapped-no-info",
                "litellm_params": {
                    "model": "openai/custom-private-fine-tune-v99",
                    "api_key": "sk-mock",
                },
            }
        ]
    )

    deployment = router.get_available_deployment("unmapped-no-info")
    assert deployment is not None

    info = router.registry().router_model_info(deployment=deployment, received_model_name="unmapped-no-info")

    assert info is not None
    assert info["litellm_provider"] == "openai"
    assert info["mode"] == "chat"
