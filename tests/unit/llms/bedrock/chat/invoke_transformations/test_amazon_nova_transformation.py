import json
from collections.abc import Mapping
from itertools import chain
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.llms.bedrock.chat.invoke_transformations.amazon_nova_transformation import (
    AmazonInvokeNovaConfig,
)
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.integrations.anthropic_cache_control_hook import GATEWAY_INJECTED_CACHE_METADATA_KEY

MODEL = "us.amazon.nova-pro-v1:0"
EPHEMERAL = {"type": "ephemeral"}
DEFAULT_CACHE_POINT = {"type": "default"}
TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]
TOOL_CALL = {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
PNG_DATA_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="


@pytest.fixture
def local_model_cost_map(monkeypatch):
    """Force the bundled in-repo cost map so capability and pricing assertions do not
    depend on the network-fetched ``main`` copy, which lags this branch until merge.

    ``get_model_info`` is lru_cached, so swapping ``model_cost`` is not enough on its
    own; clear on the way in and out so entries warmed against either map never leak
    across tests."""
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


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


def test_cache_point_behind_a_non_text_block_moves_back_to_the_last_text_block(local_model_cost_map):
    """InvokeModel rejects ``cachePoint`` on image, toolUse, and toolResult blocks
    (``extraneous key [cachePoint] is not permitted``), so the point a user put on an image or a
    tool result lands on the closest text block before it, and a message with no text block at
    all sends no point rather than a request AWS refuses.
    """
    request = _transform_request(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is in this picture?"},
                    {"type": "image_url", "image_url": {"url": PNG_DATA_URL}, "cache_control": EPHEMERAL},
                ],
            },
            {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny", "cache_control": EPHEMERAL},
        ],
        optional_params={"tools": TOOLS},
    )
    picture, image = request["messages"][0]["content"]
    assert picture == {"text": "what is in this picture?", "cachePoint": DEFAULT_CACHE_POINT}
    assert set(image) == {"image"}
    assert [set(block) for block in request["messages"][2]["content"]] == [{"toolResult"}]


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


@pytest.mark.parametrize("value", ["b c", "bad!"], ids=["nbsp", "outside-bedrock-charset"])
def test_caller_request_metadata_is_ignored_because_invoke_model_has_no_field_for_it(local_model_cost_map, value):
    caller_metadata = {"ticket": value}
    optional_params = AmazonInvokeNovaConfig().map_openai_params(
        non_default_params={"max_tokens": 8, "requestMetadata": caller_metadata},
        optional_params={},
        model=MODEL,
        drop_params=False,
    )

    request = _transform_request(
        messages=[{"role": "user", "content": "hi"}],
        optional_params={**optional_params, "requestMetadata": caller_metadata},
    )

    assert "requestMetadata" not in request
    assert request["inferenceConfig"] == {"maxTokens": 8}


GUARDED_CONVERSATION: Final = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "be brief"},
            {"type": "guarded_text", "text": "guarded system", "cache_control": EPHEMERAL},
        ],
    },
    {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "guarded_text", "text": "guarded user"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "hi"}, {"type": "guarded_text", "text": "guarded assistant"}]},
    {"role": "user", "content": "again"},
]


@pytest.mark.parametrize("drop_params_source", ["request", "global"])
def test_guarded_parts_go_as_text_under_drop_params_because_invoke_model_rejects_guard_content(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch, drop_params_source: str
):
    if drop_params_source == "global":
        monkeypatch.setattr(litellm, "drop_params", True)
    request: Final = _transform_request(
        messages=GUARDED_CONVERSATION,
        optional_params={"max_tokens": 20},
        litellm_params={"drop_params": True} if drop_params_source == "request" else {},
    )
    assert request["system"] == [{"text": "be brief"}, {"text": "guarded system", "cachePoint": DEFAULT_CACHE_POINT}]
    assert [message["content"] for message in request["messages"]] == [
        [{"text": "hello"}, {"text": "guarded user"}],
        [{"text": "hi"}, {"text": "guarded assistant"}],
        [{"text": "again"}],
    ]


def test_guarded_parts_reach_bedrock_without_drop_params(local_model_cost_map: None):
    request: Final = _transform_request(messages=GUARDED_CONVERSATION, optional_params={"max_tokens": 20})
    sent_blocks: Final = chain(request["system"], chain.from_iterable(message["content"] for message in request["messages"]))
    sent_guard_texts: Final = [block["guardContent"]["text"]["text"] for block in sent_blocks if "guardContent" in block]
    assert sent_guard_texts == ["guarded system", "guarded user", "guarded assistant"]


@pytest.mark.parametrize(
    ("service_tier", "sent_tier"),
    [("priority", "priority"), ("flex", "flex"), ("reserved", "reserved"), ("Standard", "default"), ("auto", None)],
)
def test_invoke_sends_the_service_tier_in_the_header_invoke_model_takes(service_tier: str, sent_tier: str | None):
    send: Final = MagicMock(return_value=httpx.Response(200))
    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(send)))

    litellm.completion(
        model=f"bedrock/invoke/{MODEL}",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        service_tier=service_tier,
        client=client,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region_name="us-east-1",
    )

    request: Final = send.call_args.args[0]
    assert request.headers.get("X-Amzn-Bedrock-Service-Tier") == sent_tier
    assert "serviceTier" not in json.loads(request.content)


@pytest.mark.parametrize(("requested_tier", "served_tier"), [("priority", "flex"), ("flex", "priority")])
def test_invoke_reports_and_bills_the_tier_bedrock_served_over_the_tier_requested(
    requested_tier: str, served_tier: str
) -> None:
    reply: Final = {
        "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1000, "outputTokens": 1000, "totalTokens": 2000},
    }

    def invoke(tier: Mapping[str, object]) -> litellm.ModelResponse:
        send: Final = MagicMock(
            return_value=httpx.Response(200, json=reply, headers={"X-Amzn-Bedrock-Service-Tier": served_tier})
        )
        return litellm.completion(
            model=f"bedrock/invoke/{MODEL}",
            messages=[{"role": "user", "content": "hi"}],
            client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(send))),
            aws_access_key_id="fake",
            aws_secret_access_key="fake",
            aws_region_name="us-east-1",
            **tier,
        )

    native_requested: Final = invoke({"serviceTier": {"type": requested_tier}})
    openai_requested: Final = invoke({"service_tier": requested_tier})
    openai_served: Final = invoke({"service_tier": served_tier})

    assert native_requested.service_tier == openai_requested.service_tier == served_tier
    assert (
        native_requested._hidden_params["response_cost"]
        == openai_requested._hidden_params["response_cost"]
        == openai_served._hidden_params["response_cost"]
    )
