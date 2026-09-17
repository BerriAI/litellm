"""Router model-info lookup for provider-namespaced (double-prefixed) model ids.

OpenRouter-native models have ids that start with ``openrouter/`` (``openrouter/free``,
``openrouter/auto``, ``openrouter/bodybuilder``) and are configured as
``openrouter/openrouter/<id>`` - only the outer provider prefix is stripped, so the id
sent to the OpenRouter API keeps its namespace. The cost map keys those models under
that same double-prefixed form, so ``Router.get_router_model_info`` must look up
``"<provider>/<model>"`` instead of assuming that a model id which already starts with
the provider name is prefixed already.
"""

import pytest

from litellm import Router

NATIVE_OPENROUTER_MODELS = ["free", "auto", "bodybuilder"]


@pytest.mark.parametrize("native_model", NATIVE_OPENROUTER_MODELS)
def test_native_openrouter_model_resolves_model_info(native_model: str) -> None:
    model_name = f"openrouter/openrouter/{native_model}"
    router = Router(
        model_list=[
            {
                "model_name": model_name,
                "litellm_params": {"model": model_name, "api_key": "sk-test"},
            }
        ]
    )

    model_info = router.get_router_model_info(
        deployment=router.model_list[0],
        received_model_name=model_name,
    )

    assert model_info["litellm_provider"] == "openrouter"
    assert model_info["max_input_tokens"]


def test_azure_base_model_prefix_is_not_doubled() -> None:
    router = Router(
        model_list=[
            {
                "model_name": "azure-deployment",
                "litellm_params": {
                    "model": "azure/gpt-4",
                    "api_base": "https://example.openai.azure.com",
                    "api_key": "sk-test",
                },
            }
        ]
    )

    model_info = router.get_router_model_info(
        deployment=router.model_list[0],
        received_model_name="azure-deployment",
    )

    assert model_info["litellm_provider"] == "azure"
