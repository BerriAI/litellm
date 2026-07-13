"""
Unit tests for the Bedrock Mantle (Claude Mythos Preview) integration.

Tests cover route detection, URL construction, config dispatch for both
the /chat/completions and /messages endpoints, and project (workspace)
association via `aws_bedrock_project_id`.
"""

import json
from unittest.mock import patch

import httpx
import pytest

from litellm.llms.bedrock.common_utils import BedrockModelInfo, get_bedrock_chat_config
from litellm.llms.bedrock.chat.mantle.transformation import AmazonMantleConfig
from litellm.llms.bedrock.messages.mantle_transformation import (
    AmazonMantleMessagesConfig,
)


def _anthropic_response(url: str) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "anthropic.claude-mythos-preview",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        request=httpx.Request("POST", url),
    )


def _capture_request(url: str, headers: dict, data) -> dict:
    raw_body = data.decode("utf-8") if isinstance(data, bytes) else data or "{}"
    return {
        "path": httpx.URL(url).path,
        "headers": headers,
        "body": json.loads(raw_body),
    }


def test_get_bedrock_route_mantle():
    assert (
        BedrockModelInfo.get_bedrock_route("mantle/anthropic.claude-mythos-preview")
        == "mantle"
    )


def test_get_bedrock_route_mantle_does_not_match_other_routes():
    assert (
        BedrockModelInfo.get_bedrock_route("anthropic.claude-3-sonnet-20240229-v1:0")
        != "mantle"
    )
    assert (
        BedrockModelInfo.get_bedrock_route("converse/anthropic.claude-3-sonnet")
        != "mantle"
    )


def test_explicit_mantle_route_flag():
    assert (
        BedrockModelInfo._explicit_mantle_route(
            "mantle/anthropic.claude-mythos-preview"
        )
        is True
    )
    assert BedrockModelInfo._explicit_mantle_route("anthropic.claude-3-sonnet") is False
    assert (
        BedrockModelInfo._explicit_mantle_route("converse/anthropic.claude-3-sonnet")
        is False
    )


def test_mantle_url_construction():
    config = AmazonMantleConfig()
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={"aws_region_name": "us-east-1"},
        litellm_params={},
    )
    assert url == "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages"


def test_mantle_url_construction_different_region():
    config = AmazonMantleConfig()
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={"aws_region_name": "us-west-2"},
        litellm_params={},
    )
    assert url == "https://bedrock-mantle.us-west-2.api.aws/anthropic/v1/messages"


def test_get_bedrock_chat_config_returns_mantle_config():
    config = get_bedrock_chat_config("mantle/anthropic.claude-mythos-preview")
    assert isinstance(config, AmazonMantleConfig)


def test_get_bedrock_provider_config_for_messages_api_mantle():
    config = BedrockModelInfo.get_bedrock_provider_config_for_messages_api(
        "mantle/anthropic.claude-mythos-preview"
    )
    assert isinstance(config, AmazonMantleMessagesConfig)


def test_mantle_messages_url_construction():
    config = AmazonMantleMessagesConfig()
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={"aws_region_name": "us-east-1"},
        litellm_params={},
    )
    assert url == "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages"


_VPC_ENDPOINT = "https://vpce-0a1b2c3d.bedrock-mantle.us-gov-west-1.vpce.amazonaws.com"


def test_mantle_chat_url_honors_api_base_host():
    config = AmazonMantleConfig()
    url = config.get_complete_url(
        api_base=_VPC_ENDPOINT,
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={"aws_region_name": "us-gov-west-1"},
        litellm_params={},
    )
    assert url == f"{_VPC_ENDPOINT}/anthropic/v1/messages"


def test_mantle_chat_url_honors_api_base_full_path_without_duplication():
    config = AmazonMantleConfig()
    full = f"{_VPC_ENDPOINT}/anthropic/v1/messages"
    url = config.get_complete_url(
        api_base=full,
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={"aws_region_name": "us-gov-west-1"},
        litellm_params={},
    )
    assert url == full


def test_mantle_messages_url_honors_api_base_host():
    config = AmazonMantleMessagesConfig()
    url = config.get_complete_url(
        api_base=_VPC_ENDPOINT,
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={"aws_region_name": "us-gov-west-1"},
        litellm_params={},
    )
    assert url == f"{_VPC_ENDPOINT}/anthropic/v1/messages"
    assert "api.aws" not in url


def test_mantle_messages_url_honors_api_base_with_trailing_slash():
    config = AmazonMantleMessagesConfig()
    url = config.get_complete_url(
        api_base=f"{_VPC_ENDPOINT}/",
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={"aws_region_name": "us-gov-west-1"},
        litellm_params={},
    )
    assert url == f"{_VPC_ENDPOINT}/anthropic/v1/messages"


def test_mantle_messages_url_honors_aws_bedrock_runtime_endpoint():
    config = AmazonMantleMessagesConfig()
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={
            "aws_region_name": "us-gov-west-1",
            "aws_bedrock_runtime_endpoint": _VPC_ENDPOINT,
        },
        litellm_params={},
    )
    assert url == f"{_VPC_ENDPOINT}/anthropic/v1/messages"


def test_mantle_transform_request_strips_prefix_and_adds_model():
    config = AmazonMantleConfig()
    request = config.transform_request(
        model="mantle/anthropic.claude-mythos-preview",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"max_tokens": 100},
        litellm_params={},
        headers={},
    )
    assert request["model"] == "anthropic.claude-mythos-preview"
    assert "mantle/" not in request["model"]
    assert "stream" not in request


def test_mantle_transform_request_keeps_stream_in_body():
    config = AmazonMantleConfig()
    request = config.transform_request(
        model="mantle/anthropic.claude-mythos-preview",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"max_tokens": 100, "stream": True},
        litellm_params={},
        headers={},
    )
    assert request["stream"] is True
    assert request["model"] == "anthropic.claude-mythos-preview"


@pytest.mark.asyncio
async def test_mantle_async_transform_request_keeps_stream_in_body():
    config = AmazonMantleConfig()
    request = await config.async_transform_request(
        model="mantle/anthropic.claude-mythos-preview",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"max_tokens": 100, "stream": True},
        litellm_params={},
        headers={},
    )
    assert request["stream"] is True
    assert request["model"] == "anthropic.claude-mythos-preview"


@pytest.mark.asyncio
async def test_mantle_async_transform_request_omits_stream_when_not_streaming():
    config = AmazonMantleConfig()
    request = await config.async_transform_request(
        model="mantle/anthropic.claude-mythos-preview",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"max_tokens": 100},
        litellm_params={},
        headers={},
    )
    assert "stream" not in request


def test_mantle_messages_transform_request_keeps_stream_in_body():
    from litellm.types.router import GenericLiteLLMParams

    config = AmazonMantleMessagesConfig()
    request = config.transform_anthropic_messages_request(
        model="mantle/anthropic.claude-mythos-preview",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params={"max_tokens": 100, "stream": True},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert request["stream"] is True
    assert request["model"] == "anthropic.claude-mythos-preview"


def test_mantle_messages_transform_request_omits_stream_when_not_streaming():
    from litellm.types.router import GenericLiteLLMParams

    config = AmazonMantleMessagesConfig()
    request = config.transform_anthropic_messages_request(
        model="mantle/anthropic.claude-mythos-preview",
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params={"max_tokens": 100},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert "stream" not in request


def test_mantle_chat_streaming_uses_anthropic_sse_iterator():
    from litellm.llms.anthropic.chat.handler import ModelResponseIterator

    config = AmazonMantleConfig()
    assert config.has_custom_stream_wrapper is False
    iterator = config.get_model_response_iterator(
        streaming_response=iter([]),
        sync_stream=True,
    )
    assert isinstance(iterator, ModelResponseIterator)


def test_mantle_validate_environment_sets_workspace_header():
    config = AmazonMantleConfig()
    headers = config.validate_environment(
        headers={},
        model="mantle/anthropic.claude-mythos-preview",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={"aws_bedrock_project_id": "proj_abc123def456"},
    )
    assert headers["anthropic-workspace"] == "proj_abc123def456"


def test_mantle_validate_environment_without_project_id():
    config = AmazonMantleConfig()
    headers = config.validate_environment(
        headers={},
        model="mantle/anthropic.claude-mythos-preview",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={"aws_bedrock_project_id": None},
    )
    assert "anthropic-workspace" not in headers


def test_mantle_messages_validate_environment_sets_workspace_header():
    config = AmazonMantleMessagesConfig()
    headers, api_base = config.validate_anthropic_messages_environment(
        headers={},
        model="mantle/anthropic.claude-mythos-preview",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={"aws_bedrock_project_id": "proj_abc123def456"},
        api_base="https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages",
    )
    assert headers["anthropic-workspace"] == "proj_abc123def456"
    assert api_base == "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages"


def test_mantle_messages_validate_environment_without_project_id():
    config = AmazonMantleMessagesConfig()
    headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="mantle/anthropic.claude-mythos-preview",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
    )
    assert "anthropic-workspace" not in headers


def test_mantle_completion_sends_workspace_header_and_clean_body():
    import litellm

    requests = []

    def mock_post(self, url, data=None, headers=None, **kwargs):
        requests.append(_capture_request(url=url, headers=headers or {}, data=data))
        return _anthropic_response(url)

    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", mock_post):
        response = litellm.completion(
            model="bedrock/mantle/anthropic.claude-mythos-preview",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=10,
            aws_bedrock_project_id="proj_abc123def456",
            aws_access_key_id="fake-key",
            aws_secret_access_key="fake-secret",
            aws_region_name="us-east-1",
        )

    assert response.choices[0].message.content == "ok"
    assert len(requests) == 1
    assert requests[0]["path"] == "/anthropic/v1/messages"
    assert requests[0]["headers"]["anthropic-workspace"] == "proj_abc123def456"
    assert "aws_bedrock_project_id" not in requests[0]["body"]


@pytest.mark.asyncio
async def test_mantle_anthropic_messages_sends_workspace_header_and_clean_body():
    import litellm

    requests = []

    async def mock_post(self, url, data=None, headers=None, **kwargs):
        requests.append(_capture_request(url=url, headers=headers or {}, data=data))
        return _anthropic_response(url)

    try:
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new=mock_post,
        ):
            response = await litellm.anthropic_messages(
                model="bedrock/mantle/anthropic.claude-mythos-preview",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
                aws_bedrock_project_id="proj_abc123def456",
                aws_access_key_id="fake-key",
                aws_secret_access_key="fake-secret",
                aws_region_name="us-east-1",
            )
    finally:
        await litellm.close_litellm_async_clients()

    assert response["content"][0]["text"] == "ok"
    assert len(requests) == 1
    assert requests[0]["path"] == "/anthropic/v1/messages"
    assert requests[0]["headers"]["anthropic-workspace"] == "proj_abc123def456"
    assert "aws_bedrock_project_id" not in requests[0]["body"]


@pytest.mark.asyncio
async def test_mantle_anthropic_messages_routes_to_vpc_api_base():
    import litellm

    urls = []

    async def mock_post(self, url, data=None, headers=None, **kwargs):
        urls.append(str(url))
        return _anthropic_response(str(url))

    try:
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new=mock_post,
        ):
            await litellm.anthropic_messages(
                model="bedrock/mantle/anthropic.claude-mythos-preview",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
                api_base=_VPC_ENDPOINT,
                aws_access_key_id="fake-key",
                aws_secret_access_key="fake-secret",
                aws_region_name="us-gov-west-1",
            )
    finally:
        await litellm.close_litellm_async_clients()

    assert len(urls) == 1
    assert urls[0] == f"{_VPC_ENDPOINT}/anthropic/v1/messages"
    assert "api.aws" not in urls[0]


_ANTHROPIC_SSE_EVENTS = (
    (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_stream_test",
                "type": "message",
                "role": "assistant",
                "model": "anthropic.claude-mythos-preview",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        },
    ),
    (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "pong"},
        },
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 2},
        },
    ),
    ("message_stop", {"type": "message_stop"}),
)


def _anthropic_sse_bytes() -> bytes:
    return "".join(
        f"event: {event}\ndata: {json.dumps(payload)}\n\n"
        for event, payload in _ANTHROPIC_SSE_EVENTS
    ).encode()


def _anthropic_sse_response(url: str) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        content=_anthropic_sse_bytes(),
        headers={"content-type": "text/event-stream"},
        request=httpx.Request("POST", url),
    )


def test_mantle_completion_streaming_sends_stream_and_decodes_sse():
    import litellm

    requests = []

    def mock_post(self, url, data=None, headers=None, **kwargs):
        requests.append(_capture_request(url=url, headers=headers or {}, data=data))
        return _anthropic_sse_response(url)

    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", mock_post):
        response = litellm.completion(
            model="bedrock/mantle/anthropic.claude-mythos-preview",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=10,
            stream=True,
            aws_access_key_id="fake-key",
            aws_secret_access_key="fake-secret",
            aws_region_name="us-east-1",
        )
        chunks = list(response)

    assert len(requests) == 1
    assert requests[0]["body"]["stream"] is True
    content = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
    assert content == "pong"
    assert chunks[-1].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_mantle_acompletion_streaming_sends_stream_and_decodes_sse():
    import litellm

    requests = []

    async def mock_post(self, url, data=None, headers=None, **kwargs):
        requests.append(_capture_request(url=url, headers=headers or {}, data=data))
        return _anthropic_sse_response(url)

    try:
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new=mock_post,
        ):
            response = await litellm.acompletion(
                model="bedrock/mantle/anthropic.claude-mythos-preview",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=10,
                stream=True,
                aws_access_key_id="fake-key",
                aws_secret_access_key="fake-secret",
                aws_region_name="us-east-1",
            )
            chunks = [chunk async for chunk in response]
    finally:
        await litellm.close_litellm_async_clients()

    assert len(requests) == 1
    assert requests[0]["body"]["stream"] is True
    content = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
    assert content == "pong"
    assert chunks[-1].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_mantle_anthropic_messages_streaming_sends_stream_and_passes_through_sse():
    import litellm

    requests = []

    async def mock_post(self, url, data=None, headers=None, **kwargs):
        requests.append(_capture_request(url=url, headers=headers or {}, data=data))
        return _anthropic_sse_response(str(url))

    try:
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new=mock_post,
        ):
            response = await litellm.anthropic_messages(
                model="bedrock/mantle/anthropic.claude-mythos-preview",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=10,
                stream=True,
                aws_access_key_id="fake-key",
                aws_secret_access_key="fake-secret",
                aws_region_name="us-east-1",
            )
            raw = b"".join([chunk async for chunk in response])
    finally:
        await litellm.close_litellm_async_clients()

    assert len(requests) == 1
    assert requests[0]["body"]["stream"] is True
    text = raw.decode()
    assert "event: message_start" in text
    assert '"text": "pong"' in text
    assert "event: message_stop" in text
