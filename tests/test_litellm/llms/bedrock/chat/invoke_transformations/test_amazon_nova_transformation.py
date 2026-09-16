import json

from litellm.llms.bedrock.chat.invoke_transformations.amazon_nova_transformation import (
    AmazonInvokeNovaConfig,
)
from litellm.types.integrations.anthropic_cache_control_hook import GATEWAY_INJECTED_CACHE_METADATA_KEY

MODEL = "us.amazon.nova-pro-v1:0"
EPHEMERAL = {"type": "ephemeral"}
DEFAULT_CACHE_POINT = {"type": "default"}
TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]


def _transform_request(messages, optional_params, litellm_params=None):
    return AmazonInvokeNovaConfig().transform_request(
        model=MODEL,
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params if litellm_params is not None else {},
        headers={},
    )


def test_cache_points_are_inlined_into_the_block_they_cache(local_model_cost_map):
    """InvokeModel rejects the standalone ``{"cachePoint": ...}`` block Converse emits
    (``#/system/1: required key [text] not found``); it wants ``cachePoint`` as a key of the
    block being cached."""
    request = _transform_request(
        messages=[
            {"role": "system", "content": [{"type": "text", "text": "long system prompt", "cache_control": EPHEMERAL}]},
            {"role": "user", "content": [{"type": "text", "text": "hello", "cache_control": EPHEMERAL}]},
            {"role": "assistant", "content": "hi there", "cache_control": EPHEMERAL},
            {"role": "user", "content": "again"},
        ],
        optional_params={"max_tokens": 20},
    )
    assert request["system"] == [{"text": "long system prompt", "cachePoint": DEFAULT_CACHE_POINT}]
    assert [message["content"] for message in request["messages"]] == [
        [{"text": "hello", "cachePoint": DEFAULT_CACHE_POINT}],
        [{"text": "hi there", "cachePoint": DEFAULT_CACHE_POINT}],
        [{"text": "again"}],
    ]


def test_cache_point_with_nothing_before_it_is_dropped():
    request = AmazonInvokeNovaConfig._inline_cache_points(
        {
            "system": [{"cachePoint": DEFAULT_CACHE_POINT}],
            "messages": [{"role": "user", "content": [{"cachePoint": DEFAULT_CACHE_POINT}, {"text": "hi"}]}],
        }
    )
    assert request["system"] == []
    assert request["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]


def test_tool_config_injection_point_is_neither_placed_nor_credited(local_model_cost_map):
    """InvokeModel has no tool caching, so the point cannot land and the gateway must not be
    credited for it in spend attribution."""
    metadata = {"user_api_key": "sk-test"}
    request = _transform_request(
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"tools": TOOLS, "cache_control_injection_points": [{"location": "tool_config"}]},
        litellm_params={"metadata": metadata, "litellm_metadata": None, "model_info": {"id": "dep-bedrock"}},
    )
    assert [tool["toolSpec"]["name"] for tool in request["toolConfig"]["tools"]] == ["f"]
    assert "cachePoint" not in json.dumps(request)
    assert GATEWAY_INJECTED_CACHE_METADATA_KEY not in metadata
