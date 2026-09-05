import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import litellm
from litellm.llms.bedrock.chat import BedrockConverseLLM
from litellm.llms.bedrock.chat.converse_handler import make_sync_call
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler



def test_encode_model_id_with_inference_profile():
    """
    Test instance profile is properly encoded when used as a model
    """
    test_model = "arn:aws:bedrock:us-east-1:12345678910:application-inference-profile/ujdtmcirjhevpi"
    expected_model = "arn%3Aaws%3Abedrock%3Aus-east-1%3A12345678910%3Aapplication-inference-profile%2Fujdtmcirjhevpi"
    bedrock_converse_llm = BedrockConverseLLM()
    returned_model = bedrock_converse_llm.encode_model_id(test_model)
    assert expected_model == returned_model


def _converse_response_body() -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
    }


def _complete_with_mocked_post(model: str, **kwargs) -> str:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value=_converse_response_body())
    mock_response.text = json.dumps(_converse_response_body())
    mock_response.headers = httpx.Headers({})
    client = HTTPHandler()
    client.post = MagicMock(return_value=mock_response)

    litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        client=client,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        **kwargs,
    )
    post_call = client.post.call_args
    return post_call.args[0] if post_call.args else post_call.kwargs["url"]


@pytest.mark.parametrize(
    "model,kwargs,expected_url",
    [
        (
            "bedrock/ap-northeast-1/anthropic.claude-haiku-4-5-20251001-v1:0",
            {},
            "https://bedrock-runtime.ap-northeast-1.amazonaws.com/model/anthropic.claude-haiku-4-5-20251001-v1%3A0/converse",
        ),
        (
            "bedrock/ap-southeast-3/anthropic.claude-haiku-4-5-20251001-v1:0",
            {},
            "https://bedrock-runtime.ap-southeast-3.amazonaws.com/model/anthropic.claude-haiku-4-5-20251001-v1%3A0/converse",
        ),
        (
            "bedrock/converse/us-gov-west-1/anthropic.claude-haiku-4-5-20251001-v1:0",
            {},
            "https://bedrock-runtime.us-gov-west-1.amazonaws.com/model/anthropic.claude-haiku-4-5-20251001-v1%3A0/converse",
        ),
        (
            "bedrock/ap-northeast-1/anthropic.claude-haiku-4-5-20251001-v1:0",
            {"aws_region_name": "eu-west-1"},
            "https://bedrock-runtime.eu-west-1.amazonaws.com/model/anthropic.claude-haiku-4-5-20251001-v1%3A0/converse",
        ),
        (
            "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            {"aws_region_name": "us-east-1"},
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-haiku-4-5-20251001-v1%3A0/converse",
        ),
    ],
)
def test_region_prefixed_converse_model_calls_that_region_with_the_bare_model_id(model, kwargs, expected_url):
    assert _complete_with_mocked_post(model, **kwargs) == expected_url


def _stream_completion_with_spied_iter_bytes(model: str, **kwargs) -> MagicMock:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_bytes = MagicMock(return_value=iter([]))
    client = HTTPHandler()
    client.post = MagicMock(return_value=mock_response)

    litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        client=client,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region_name="us-east-1",
        **kwargs,
    )
    return mock_response.iter_bytes


def test_make_sync_call_does_not_rechunk_stream_by_default():
    """Re-chunking the event stream into fixed 1024-byte blocks holds small
    early events in httpx's ByteChunker until 1024 bytes accumulate, delaying
    time-to-first-chunk by the whole generation when Bedrock trickles bytes
    (e.g. buffered tool-use streams)."""
    response = MagicMock()
    response.status_code = 200
    client = MagicMock()
    client.post = MagicMock(return_value=response)

    make_sync_call(
        client=client,
        api_base="https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-sonnet-4-6/converse-stream",
        headers={},
        data="{}",
        model="anthropic.claude-sonnet-4-6",
        messages=[],
        logging_obj=MagicMock(),
    )

    response.iter_bytes.assert_called_once_with(chunk_size=None)


def test_make_sync_call_honors_explicit_stream_chunk_size():
    response = MagicMock()
    response.status_code = 200
    client = MagicMock()
    client.post = MagicMock(return_value=response)

    make_sync_call(
        client=client,
        api_base="https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-sonnet-4-6/converse-stream",
        headers={},
        data="{}",
        model="anthropic.claude-sonnet-4-6",
        messages=[],
        logging_obj=MagicMock(),
        stream_chunk_size=2048,
    )

    response.iter_bytes.assert_called_once_with(chunk_size=2048)


def test_converse_completion_forwards_bedrock_response_headers():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value=_converse_response_body())
    mock_response.text = json.dumps(_converse_response_body())
    mock_response.headers = httpx.Headers({"x-amzn-requestid": "req-123"})
    client = HTTPHandler()
    client.post = MagicMock(return_value=mock_response)

    response = litellm.completion(
        model="bedrock/converse/anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        client=client,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region_name="us-east-1",
    )

    assert response._hidden_params["additional_headers"]["llm_provider-x-amzn-requestid"] == "req-123"


def test_converse_streaming_forwards_bedrock_response_headers():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_bytes = MagicMock(return_value=iter([]))
    mock_response.headers = httpx.Headers({"x-amzn-requestid": "req-456"})
    client = HTTPHandler()
    client.post = MagicMock(return_value=mock_response)

    response = litellm.completion(
        model="bedrock/converse/anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        client=client,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region_name="us-east-1",
    )

    assert response._hidden_params["additional_headers"]["llm_provider-x-amzn-requestid"] == "req-456"


@pytest.mark.asyncio
async def test_async_converse_completion_forwards_bedrock_response_headers():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value=_converse_response_body())
    mock_response.text = json.dumps(_converse_response_body())
    mock_response.headers = httpx.Headers({"x-amzn-requestid": "req-abc"})
    client = AsyncHTTPHandler()
    client.post = AsyncMock(return_value=mock_response)

    response = await litellm.acompletion(
        model="bedrock/converse/anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        client=client,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region_name="us-east-1",
    )

    assert response._hidden_params["additional_headers"]["llm_provider-x-amzn-requestid"] == "req-abc"


@pytest.mark.asyncio
async def test_async_converse_streaming_forwards_bedrock_response_headers():
    async def _no_bytes(chunk_size=None):
        return
        yield b""

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.aiter_bytes = _no_bytes
    mock_response.headers = httpx.Headers({"x-amzn-requestid": "req-def"})
    client = AsyncHTTPHandler()
    client.post = AsyncMock(return_value=mock_response)

    response = await litellm.acompletion(
        model="bedrock/converse/anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        client=client,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region_name="us-east-1",
    )

    assert response._hidden_params["additional_headers"]["llm_provider-x-amzn-requestid"] == "req-def"


def test_completion_plumbs_stream_chunk_size_through_converse():
    iter_bytes_spy = _stream_completion_with_spied_iter_bytes(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
    )
    iter_bytes_spy.assert_called_once_with(chunk_size=None)

    iter_bytes_spy = _stream_completion_with_spied_iter_bytes(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        stream_chunk_size=2048,
    )
    iter_bytes_spy.assert_called_once_with(chunk_size=2048)
