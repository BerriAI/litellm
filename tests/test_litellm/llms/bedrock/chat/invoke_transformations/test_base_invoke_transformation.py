import json
import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)


def test_transform_request_drops_internal_params():
    """LiteLLM-internal MCP params (e.g. skip_mcp_handler) are control flags used
    inside LiteLLM and are not valid Bedrock inference parameters. Leaking them
    into the provider request body makes Bedrock reject the request. The Converse
    path already filters them via filter_internal_params; the invoke path must do
    the same."""
    request_body = AmazonInvokeConfig().transform_request(
        model="mistral.mistral-7b-instruct-v0:2",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"skip_mcp_handler": True, "max_tokens": 10},
        litellm_params={},
        headers={},
    )

    assert "skip_mcp_handler" not in json.dumps(request_body)
