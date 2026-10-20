import pytest

from litellm.router_utils.deployment_pricing import deployment_pricing_models, pricing_models


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
def test_pricing_models_walks_base_model_then_deployment_model(base_model, model, expected):
    assert pricing_models(base_model, model) == expected


@pytest.mark.parametrize(
    "deployment, expected",
    [
        (
            {"litellm_params": {"model": "azure/my-gpt4-deploy"}, "model_info": {"base_model": "azure/gpt-4.1"}},
            ("azure/gpt-4.1", "azure/my-gpt4-deploy"),
        ),
        (
            {"litellm_params": {"model": "azure/my-gpt4-deploy", "base_model": "azure/gpt-4.1"}},
            ("azure/gpt-4.1", "azure/my-gpt4-deploy"),
        ),
        (
            {
                "litellm_params": {"model": "azure/my-gpt4-deploy", "base_model": "azure/gpt-4o"},
                "model_info": {"base_model": "azure/gpt-4.1"},
            },
            ("azure/gpt-4.1", "azure/my-gpt4-deploy"),
        ),
        ({"litellm_params": {"model": "openai/gpt-4.1"}, "model_info": {}}, ("openai/gpt-4.1",)),
        ({"litellm_params": "not-a-dict", "model_info": None}, ()),
        ("not-a-deployment", ()),
    ],
    ids=[
        "model_info_base_model",
        "params_base_model",
        "model_info_wins",
        "no_base_model",
        "malformed_blocks",
        "not_a_dict",
    ],
)
def test_deployment_pricing_models_reads_the_deployment_dict(deployment, expected):
    assert deployment_pricing_models(deployment) == expected
