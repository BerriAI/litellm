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
    "model,valid_param,valid_value,valid_param_path",
    [
        ("cohere.command-text-v14", "max_tokens", 10, ("max_tokens",)),
        ("ai21.j2-ultra-v1", "maxTokens", 10, ("maxTokens",)),
        ("mistral.mistral-7b-instruct-v0:2", "max_tokens", 10, ("max_tokens",)),
        (
            "amazon.titan-text-express-v1",
            "maxTokenCount",
            10,
            ("textGenerationConfig", "maxTokenCount"),
        ),
        ("meta.llama2-13b-chat-v1", "max_gen_len", 10, ("max_gen_len",)),
    ],
)
def test_transform_request_drops_internal_params_from_typed_invoke_body(
    model, valid_param, valid_value, valid_param_path
):
    request_body = AmazonInvokeConfig().transform_request(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        optional_params={
            "skip_mcp_handler": True,
            "_skip_mcp_handler": True,
            "mcp_handler_context": {"request_id": "test"},
            "stream_chunk_size": 2048,
            "aws_region_name": "us-east-1",
            valid_param: valid_value,
        },
        litellm_params={},
        headers={},
    )

    serialized_request = json.dumps(request_body)
    for internal_param in (
        "skip_mcp_handler",
        "_skip_mcp_handler",
        "mcp_handler_context",
        "stream_chunk_size",
        "aws_region_name",
    ):
        assert internal_param not in serialized_request

    value = request_body
    for path_part in valid_param_path:
        value = value[path_part]
    assert value == valid_value
