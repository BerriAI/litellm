import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeConfig,
)
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)


@pytest.mark.parametrize(
    "config,model",
    [
        (AmazonInvokeConfig, "anthropic.claude-3-sonnet-20240229-v1:0"),
        (AmazonInvokeConfig, "amazon.titan-text-express-v1"),
        (AmazonInvokeConfig, "mistral.mistral-7b-instruct-v0:2"),
        (AmazonAnthropicClaudeConfig, "anthropic.claude-sonnet-4-6"),
    ],
)
def test_signed_invoke_body_drops_stream_chunk_size(config, model):
    """stream_chunk_size is a LiteLLM-internal knob for re-chunking the HTTP
    response stream. Leaking it into the provider request body makes Bedrock
    reject the whole request: ValidationException 'stream_chunk_size: Extra
    inputs are not permitted'. The two invoke transform entry points build the
    body differently, so this asserts on the actual signed wire bytes that both
    funnel through, regardless of which transform produced them."""
    cfg = config()
    request_body = cfg.transform_request(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"stream": True, "stream_chunk_size": 2048, "max_tokens": 10},
        litellm_params={},
        headers={},
    )

    _, signed_body = cfg.sign_request(
        headers={},
        optional_params={},
        request_data=request_body,
        api_base="https://bedrock-runtime.us-east-1.amazonaws.com/model/{}/invoke".format(
            model
        ),
        api_key="test-bearer-token",
        model=model,
        stream=True,
    )

    assert signed_body is not None
    assert "stream_chunk_size" not in signed_body.decode()
    assert "max_tokens" in signed_body.decode()


def test_extra_body_passthrough_by_default():
    """Unknown body keys are forwarded verbatim when drop_params is off, so the
    soft-allowlist escape hatch keeps working."""
    cfg = AmazonInvokeConfig()
    request_body = cfg.transform_request(
        model="mistral.mistral-7b-instruct-v0:2",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"temperature": 0.5, "made_up_param": "x"},
        litellm_params={},
        headers={},
    )

    assert request_body["temperature"] == 0.5
    assert request_body["made_up_param"] == "x"


def test_drop_params_strips_extra_body_but_keeps_known_params():
    """drop_params selects the strict body: typed provider keys survive, unknown
    passthrough keys are dropped instead of being shipped to the provider."""
    cfg = AmazonInvokeConfig()
    request_body = cfg.transform_request(
        model="mistral.mistral-7b-instruct-v0:2",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"temperature": 0.5, "made_up_param": "x"},
        litellm_params={"drop_params": True},
        headers={},
    )

    assert request_body["temperature"] == 0.5
    assert "made_up_param" not in request_body
