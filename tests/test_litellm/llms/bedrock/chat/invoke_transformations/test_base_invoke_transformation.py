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
        "amazon.titan-text-express-v1",
        "ai21.j2-ultra-v1",
        "meta.llama3-8b-instruct-v1:0",
    ],
)
def test_transform_request_drops_internal_mcp_params(model):
    """skip_mcp_handler, _skip_mcp_handler and mcp_handler_context are
    LiteLLM-internal MCP control flags, not Bedrock inference parameters. The
    invoke path splats inference_params straight into the request body, so
    without filtering they reach AWS and the request is rejected with
    'extraneous key is not permitted'. Regression for
    https://github.com/BerriAI/litellm/issues/30371."""
    request_body = AmazonInvokeConfig().transform_request(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        optional_params={
            "max_tokens": 10,
            "skip_mcp_handler": True,
            "_skip_mcp_handler": True,
            "mcp_handler_context": {"server": "x"},
        },
        litellm_params={},
        headers={},
    )

    serialized = json.dumps(request_body)
    assert "skip_mcp_handler" not in serialized
    assert "mcp_handler_context" not in serialized
    assert "max_tokens" in serialized
