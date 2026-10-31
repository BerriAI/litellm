"""A cost-map row with a non-chat ``mode`` must not drop the deployment.

A Bedrock rerank deployment that uses an application inference profile and sets
``model_info.base_model`` is priced from the backend model's cost-map row, which
carries ``"mode": "rerank"``. ``_PricingModelInfo`` used to reject that literal,
so the deployment was skipped while the router was being built and every request
to the model group returned "no healthy deployments".
"""

import pytest

from litellm import Router
from litellm.utils import model_info_with_pricing_overrides

RERANK_DEPLOYMENT = {
    "model_name": "cohere-rerank-3-5",
    "litellm_params": {
        "model": "bedrock/arn:aws:bedrock:eu-central-1::foundation-model/cohere.rerank-v3-5:0",
        "model_id": "arn:aws:bedrock:eu-central-1:123456789012:application-inference-profile/abc123",
        "aws_region_name": "eu-central-1",
        "aws_access_key_id": "fake-key",
        "aws_secret_access_key": "fake-secret",
    },
    "model_info": {
        "base_model": "cohere.rerank-v3-5:0",
        "input_cost_per_query": 0.00175,
    },
}


@pytest.fixture(autouse=True)
def local_cost_map(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def test_rerank_cost_map_mode_does_not_drop_the_deployment():
    router = Router(model_list=[RERANK_DEPLOYMENT])

    assert len(router.model_list) == 1
    assert router.model_list[0]["model_info"]["input_cost_per_query"] == 0.00175


def test_unknown_cost_map_mode_falls_back_to_the_explicit_model_info():
    """An unrecognised mode must not raise while the router registers a deployment."""
    model_info = {
        "key": "some-model",
        "max_tokens": 10,
        "max_input_tokens": 10,
        "max_output_tokens": 10,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "openai",
        "mode": "a_mode_this_version_does_not_know",
        "supported_openai_params": [],
    }

    priced = model_info_with_pricing_overrides(model_info, {})

    assert priced["mode"] == "a_mode_this_version_does_not_know"
