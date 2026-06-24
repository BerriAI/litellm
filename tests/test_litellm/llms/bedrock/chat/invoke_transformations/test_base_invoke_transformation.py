import json
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
from litellm.types.internal_params import LiteLLMInternalParam


@pytest.mark.parametrize(
    "config,model",
    [
        (AmazonInvokeConfig, "anthropic.claude-3-sonnet-20240229-v1:0"),
        (AmazonInvokeConfig, "amazon.titan-text-express-v1"),
        (AmazonInvokeConfig, "mistral.mistral-7b-instruct-v0:2"),
        (AmazonAnthropicClaudeConfig, "anthropic.claude-sonnet-4-6"),
    ],
)
def test_transform_request_drops_stream_chunk_size(config, model):
    """stream_chunk_size is a LiteLLM-internal knob for re-chunking the HTTP
    response stream. Leaking it into the provider request body makes Bedrock
    reject the whole request: ValidationException 'stream_chunk_size: Extra
    inputs are not permitted'."""
    request_body = config().transform_request(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"stream": True, "stream_chunk_size": 2048, "max_tokens": 10},
        litellm_params={},
        headers={},
    )

    assert "stream_chunk_size" not in json.dumps(request_body)


@pytest.mark.parametrize(
    "model",
    [
        "mistral.mistral-7b-instruct-v0:2",
        "cohere.command-text-v14",
        "amazon.titan-text-express-v1",
        "meta.llama3-8b-instruct-v1:0",
        "ai21.j2-ultra-v1",
    ],
)
def test_invoke_request_does_not_leak_internal_params(model):
    """Regression for #30371: the invoke path splats inference_params into the
    request body, so internal knobs (e.g. skip_mcp_handler) leaked and strict
    Bedrock models rejected the request. Real inference params must survive."""
    seeded = {param.value: "internal" for param in LiteLLMInternalParam}
    seeded.update({"max_tokens": 10, "temperature": 0.5})

    request_body = AmazonInvokeConfig().transform_request(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        optional_params=seeded,
        litellm_params={},
        headers={},
    )

    serialized = json.dumps(request_body)
    for param in LiteLLMInternalParam:
        assert param.value not in serialized, f"{param.value} leaked into {model} body"
    assert "max_tokens" in serialized and "temperature" in serialized


@pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "amazon.nova-micro-v1:0",
        "twelvelabs.pegasus-1-2-v1:0",
        "openai.gpt-oss-20b-1:0",
    ],
)
def test_invoke_delegate_paths_do_not_leak_internal_params(model):
    """The anthropic, nova, twelvelabs and openai invoke providers delegate to a
    sub-transform instead of building the body from inference_params. Those
    delegates splat optional_params into their own request body, so the internal
    knobs must be stripped before the hand-off or they leak just like the
    inference_params splat did (#30371)."""
    seeded = {param.value: "internal" for param in LiteLLMInternalParam}
    seeded.update({"temperature": 0.5})

    request_body = AmazonInvokeConfig().transform_request(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        optional_params=seeded,
        litellm_params={},
        headers={},
    )

    serialized = json.dumps(request_body)
    for param in LiteLLMInternalParam:
        assert param.value not in serialized, f"{param.value} leaked into {model} body"
    assert "temperature" in serialized
