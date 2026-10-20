import pytest

from litellm.router_utils.deployment_pricing import deployment_pricing_model, deployment_pricing_models


@pytest.mark.parametrize(
    "base_model, model, expected",
    [
        ("azure/gpt-4.1", "azure/my-gpt4-deploy", ("azure/gpt-4.1", "azure/my-gpt4-deploy")),
        (None, "azure/my-gpt4-deploy", ("azure/my-gpt4-deploy",)),
        ("", "azure/my-gpt4-deploy", ("azure/my-gpt4-deploy",)),
        ("azure/gpt-4.1", "azure/gpt-4.1", ("azure/gpt-4.1",)),
        (None, None, ()),
    ],
    ids=["base_model_first", "no_base_model", "blank_base_model", "same_name_once", "nothing_named"],
)
def test_deployment_pricing_models_walks_base_model_then_deployment_model(base_model, model, expected):
    assert deployment_pricing_models(base_model, model) == expected
    assert deployment_pricing_model(base_model, model) == (expected[0] if expected else None)
