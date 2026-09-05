from typing import Final

import pytest

from litellm.llms.bedrock.chat.invoke_transformations.amazon_openai_transformation import (
    AmazonBedrockOpenAIConfig,
)

IMPORTED_MODEL_ARN: Final = "arn:aws:bedrock:us-east-1:123456789012:imported-model/abc123"


@pytest.mark.parametrize(
    "model",
    [
        f"bedrock/openai/us-east-1/{IMPORTED_MODEL_ARN}",
        f"openai/us-east-1/{IMPORTED_MODEL_ARN}",
        f"bedrock/openai/{IMPORTED_MODEL_ARN}",
    ],
)
def test_region_prefixed_imported_model_invokes_the_bare_arn_in_the_prefix_region(monkeypatch, model):
    monkeypatch.setenv("AWS_REGION_NAME", "us-west-2")
    config: Final = AmazonBedrockOpenAIConfig()

    url: Final = config.get_complete_url(
        api_base=None, api_key=None, model=model, optional_params={}, litellm_params={}, stream=False
    )
    body: Final = config.transform_request(
        model=model,
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert url == (
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/"
        "arn:aws:bedrock:us-east-1:123456789012:imported-model%2Fabc123/invoke"
    )
    assert body["model"] == IMPORTED_MODEL_ARN
