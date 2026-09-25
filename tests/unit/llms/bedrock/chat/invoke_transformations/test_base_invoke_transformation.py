import json
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import litellm
from litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeConfig,
)
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from tests._support.stream_chunk_size import (
    LitellmParamsRecorder,
    keys_at_every_depth,
    record_litellm_params,
    recorded_stream_chunk_size,
)


@pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "amazon.titan-text-express-v1",
        "mistral.mistral-7b-instruct-v0:2",
        "anthropic.claude-sonnet-4-6",
    ],
)
def test_completion_keeps_stream_chunk_size_out_of_every_invoke_family_body(model: str) -> None:
    mock_response: Final = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_bytes = MagicMock(return_value=iter([]))
    client: Final = HTTPHandler()
    client.post = MagicMock(return_value=mock_response)

    litellm.completion(
        model=f"bedrock/invoke/{model}",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        max_tokens=10,
        client=client,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region_name="us-east-1",
        stream_chunk_size=2048,
    )

    data: Final = client.post.call_args.kwargs["data"]
    assert "stream_chunk_size" not in keys_at_every_depth(json.loads(data)), data


def test_validate_environment_maps_guardrail_config_to_invoke_headers():
    """The InvokeModel API takes the guardrail identifier/version/trace as
    X-Amzn-Bedrock-* request headers, unlike Converse which takes them in the
    body. https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html"""
    optional_params = {
        "guardrailConfig": {
            "guardrailIdentifier": "ff6ujrregl1q",
            "guardrailVersion": "DRAFT",
            "trace": "enabled",
        },
        "max_tokens": 10,
    }

    headers = AmazonInvokeConfig().validate_environment(
        headers={},
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        optional_params=optional_params,
        litellm_params={},
    )

    assert headers["X-Amzn-Bedrock-GuardrailIdentifier"] == "ff6ujrregl1q"
    assert headers["X-Amzn-Bedrock-GuardrailVersion"] == "DRAFT"
    assert headers["X-Amzn-Bedrock-Trace"] == "ENABLED"
    assert "guardrailConfig" not in optional_params


def test_validate_environment_without_guardrail_config_leaves_headers_untouched():
    headers = AmazonInvokeConfig().validate_environment(
        headers={"foo": "bar"},
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"max_tokens": 10},
        litellm_params={},
    )

    assert headers == {"foo": "bar"}


def test_validate_environment_skips_absent_guardrail_fields():
    headers = AmazonInvokeConfig().validate_environment(
        headers={},
        model="amazon.titan-text-express-v1",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"guardrailConfig": {"guardrailIdentifier": "gr-id", "guardrailVersion": "1"}},
        litellm_params={},
    )

    assert headers == {
        "X-Amzn-Bedrock-GuardrailIdentifier": "gr-id",
        "X-Amzn-Bedrock-GuardrailVersion": "1",
    }


def test_validate_environment_does_not_clobber_explicit_guardrail_headers():
    """Users worked around the missing guardrailConfig support by passing the
    AWS headers directly; an explicit header must keep winning over
    guardrailConfig regardless of casing."""
    headers = AmazonInvokeConfig().validate_environment(
        headers={"x-amzn-bedrock-guardrailidentifier": "explicit-id"},
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={
            "guardrailConfig": {"guardrailIdentifier": "config-id", "guardrailVersion": "2"},
        },
        litellm_params={},
    )

    assert headers["x-amzn-bedrock-guardrailidentifier"] == "explicit-id"
    assert "X-Amzn-Bedrock-GuardrailIdentifier" not in headers
    assert headers["X-Amzn-Bedrock-GuardrailVersion"] == "2"


@pytest.mark.parametrize(
    "bad_guardrail_config",
    [
        {"guardrailIdentifier": "gr-id", "trace": "verbose"},
        {"guardrailIdentifier": ["gr-id"]},
        "gr-id",
        {},
        {"trace": "enabled"},
    ],
)
def test_validate_environment_rejects_malformed_guardrail_config(bad_guardrail_config):
    with pytest.raises(BedrockError) as excinfo:
        AmazonInvokeConfig().validate_environment(
            headers={},
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"guardrailConfig": bad_guardrail_config},
            litellm_params={},
        )

    assert excinfo.value.status_code == 400
    assert "guardrailConfig" in str(excinfo.value)


@pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "amazon.titan-text-express-v1",
        "mistral.mistral-7b-instruct-v0:2",
        "meta.llama3-8b-instruct-v1:0",
    ],
)
def test_guardrail_config_flows_to_headers_not_request_body(model):
    """Mirrors the handler flow (validate_environment then transform_request):
    guardrailConfig must end up in the signed headers and never leak into the
    request body, where Bedrock rejects it as an extra input."""
    config = AmazonInvokeConfig()
    optional_params = {
        "guardrailConfig": {
            "guardrailIdentifier": "ff6ujrregl1q",
            "guardrailVersion": "DRAFT",
            "trace": "disabled",
        },
        "max_tokens": 10,
    }
    messages = [{"role": "user", "content": "hi"}]

    headers = config.validate_environment(
        headers={},
        model=model,
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
    )
    request_body = config.transform_request(
        model=model,
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers=headers,
    )

    assert "guardrailConfig" not in json.dumps(request_body)
    assert headers["X-Amzn-Bedrock-GuardrailIdentifier"] == "ff6ujrregl1q"
    assert headers["X-Amzn-Bedrock-GuardrailVersion"] == "DRAFT"
    assert headers["X-Amzn-Bedrock-Trace"] == "DISABLED"


def test_get_error_class_preserves_provider_headers():
    """The invoke handler path hands real provider headers to get_error_class (LIT-5428)."""
    error = AmazonInvokeConfig().get_error_class(
        error_message="Amazon Bedrock is unable to process your request.",
        status_code=500,
        headers={"x-amzn-RequestId": "req-invoke-500"},
    )

    assert isinstance(error, BedrockError)
    assert error.headers == {"x-amzn-RequestId": "req-invoke-500"}
    assert error.response.headers["x-amzn-requestid"] == "req-invoke-500"


def test_transform_response_hands_json_mode_to_nova():
    """The invoke dispatcher forwards its json_mode argument to Nova instead of dropping it."""
    from litellm.types.utils import ModelResponse

    response_json = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_nova_json",
                            "name": "json_tool_call",
                            "input": {"city": "Paris", "temperature": 21},
                        }
                    }
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 5, "outputTokens": 4, "totalTokens": 9},
    }
    raw_response = httpx.Response(200, json=response_json, request=httpx.Request("POST", "https://bedrock"))

    result = AmazonInvokeConfig().transform_response(
        model="invoke/amazon.nova-lite-v1:0",
        raw_response=raw_response,
        model_response=ModelResponse(),
        logging_obj=MagicMock(),
        request_data={},
        messages=[{"role": "user", "content": "weather"}],
        optional_params={},
        litellm_params={},
        encoding=None,
        api_key=None,
        json_mode=True,
    )

    assert result.choices[0].message.tool_calls is None
    assert json.loads(result.choices[0].message.content) == {"city": "Paris", "temperature": 21}


def _stream_invoke_completion_with_spied_client(
    monkeypatch: pytest.MonkeyPatch, **kwargs
) -> tuple[MagicMock, MagicMock, LitellmParamsRecorder]:
    recorder: Final = record_litellm_params(monkeypatch)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_bytes = MagicMock(return_value=iter([]))
    client = HTTPHandler()
    client.post = MagicMock(return_value=mock_response)

    litellm.completion(
        model="bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        client=client,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region_name="us-east-1",
        **kwargs,
    )
    return mock_response.iter_bytes, client.post, recorder


def test_completion_stream_chunk_size_reaches_iter_bytes_but_not_invoke_body(
    monkeypatch: pytest.MonkeyPatch,
):
    iter_bytes_spy, post_spy, recorder = _stream_invoke_completion_with_spied_client(monkeypatch, stream_chunk_size=64)

    iter_bytes_spy.assert_called_once_with(chunk_size=64)
    data: Final = post_spy.call_args.kwargs["data"]
    assert "stream_chunk_size" not in keys_at_every_depth(json.loads(data)), data
    assert len(recorder.seen) == 1
    assert recorded_stream_chunk_size(recorder.seen[0]) == 64


def test_completion_without_stream_chunk_size_uses_default_chunking(monkeypatch: pytest.MonkeyPatch):
    iter_bytes_spy, _, recorder = _stream_invoke_completion_with_spied_client(monkeypatch)

    iter_bytes_spy.assert_called_once_with(chunk_size=None)
    assert len(recorder.seen) == 1
    assert recorded_stream_chunk_size(recorder.seen[0]) is None


async def _astream_invoke_completion_with_spied_client(
    monkeypatch: pytest.MonkeyPatch, **kwargs
) -> tuple[MagicMock, AsyncMock, LitellmParamsRecorder]:
    async def _no_bytes():
        return
        yield b""

    mock_response = MagicMock()
    mock_response.status_code = 200
    recorder: Final = record_litellm_params(monkeypatch)
    mock_response.aiter_bytes = MagicMock(return_value=_no_bytes())
    aiter_bytes_spy = mock_response.aiter_bytes
    client = AsyncHTTPHandler()
    client.post = AsyncMock(return_value=mock_response)

    await litellm.acompletion(
        model="bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        client=client,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region_name="us-east-1",
        **kwargs,
    )
    return aiter_bytes_spy, client.post, recorder


@pytest.mark.asyncio
async def test_acompletion_stream_chunk_size_reaches_aiter_bytes_but_not_invoke_body(
    monkeypatch: pytest.MonkeyPatch,
):
    aiter_bytes_spy, post_spy, recorder = await _astream_invoke_completion_with_spied_client(
        monkeypatch, stream_chunk_size=64
    )

    aiter_bytes_spy.assert_called_once_with(chunk_size=64)
    data: Final = post_spy.call_args.kwargs["data"]
    assert "stream_chunk_size" not in keys_at_every_depth(json.loads(data)), data
    assert len(recorder.seen) == 1
    assert recorded_stream_chunk_size(recorder.seen[0]) == 64


@pytest.mark.asyncio
async def test_acompletion_without_stream_chunk_size_uses_default_chunking(monkeypatch: pytest.MonkeyPatch):
    aiter_bytes_spy, _, recorder = await _astream_invoke_completion_with_spied_client(monkeypatch)

    aiter_bytes_spy.assert_called_once_with(chunk_size=None)
    assert len(recorder.seen) == 1
    assert recorded_stream_chunk_size(recorder.seen[0]) is None


@pytest.mark.parametrize("stream_chunk_size,expected_chunk_size", [(64, 64), (None, None)])
def test_router_deployment_stream_chunk_size_reaches_iter_bytes(
    monkeypatch: pytest.MonkeyPatch, stream_chunk_size, expected_chunk_size
):
    recorder: Final = record_litellm_params(monkeypatch)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_bytes = MagicMock(return_value=iter([]))
    client = HTTPHandler()
    client.post = MagicMock(return_value=mock_response)
    deployment_params = {
        "model": "bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0",
        "aws_access_key_id": "fake",
        "aws_secret_access_key": "fake",
        "aws_region_name": "us-east-1",
    }
    router = litellm.Router(
        model_list=[
            {
                "model_name": "invoke-chunked",
                "litellm_params": deployment_params
                | ({} if stream_chunk_size is None else {"stream_chunk_size": stream_chunk_size}),
            }
        ]
    )

    router.completion(
        model="invoke-chunked",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        client=client,
    )

    mock_response.iter_bytes.assert_called_once_with(chunk_size=expected_chunk_size)
    data: Final = client.post.call_args.kwargs["data"]
    assert "stream_chunk_size" not in keys_at_every_depth(json.loads(data)), data
    assert len(recorder.seen) == 1
    assert recorded_stream_chunk_size(recorder.seen[0]) == stream_chunk_size


def test_stream_wrapper_rejects_non_int_stream_chunk_size(monkeypatch: pytest.MonkeyPatch):
    record_litellm_params(monkeypatch)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_bytes = MagicMock(return_value=iter([]))
    client = HTTPHandler()
    client.post = MagicMock(return_value=mock_response)

    with pytest.raises(litellm.BadRequestError):
        litellm.completion(
            model="bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            client=client,
            aws_access_key_id="fake",
            aws_secret_access_key="fake",
            aws_region_name="us-east-1",
            stream_chunk_size="sixty-four",
        )

    client.post.assert_not_called()
