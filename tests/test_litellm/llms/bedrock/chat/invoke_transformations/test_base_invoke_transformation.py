import json
from unittest.mock import MagicMock

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
from litellm.llms.custom_httpx.http_handler import HTTPHandler


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


def _invoke_with_mocked_post(model: str, aws_region_name: str | None) -> str:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={"outputs": [{"text": "hi", "stop_reason": "stop"}]})
    mock_response.text = json.dumps({"outputs": [{"text": "hi", "stop_reason": "stop"}]})
    mock_response.headers = httpx.Headers({})
    client = HTTPHandler()
    client.post = MagicMock(return_value=mock_response)

    litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        client=client,
        api_key="test-bearer-token",
        aws_region_name=aws_region_name,
    )
    post_call = client.post.call_args
    return post_call.args[0] if post_call.args else post_call.kwargs["url"]


@pytest.mark.parametrize(
    "model,aws_region_name,expected_url",
    [
        (
            "bedrock/us-east-1/mistral.mistral-7b-instruct-v0:2",
            None,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/mistral.mistral-7b-instruct-v0:2/invoke",
        ),
        (
            "bedrock/invoke/ap-southeast-3/mistral.mistral-7b-instruct-v0:2",
            None,
            "https://bedrock-runtime.ap-southeast-3.amazonaws.com/model/mistral.mistral-7b-instruct-v0:2/invoke",
        ),
        (
            "bedrock/invoke/us-east-1/mistral.mistral-7b-instruct-v0:2",
            "eu-west-1",
            "https://bedrock-runtime.eu-west-1.amazonaws.com/model/mistral.mistral-7b-instruct-v0:2/invoke",
        ),
        (
            "bedrock/mistral.mistral-7b-instruct-v0:2",
            None,
            "https://bedrock-runtime.us-west-2.amazonaws.com/model/mistral.mistral-7b-instruct-v0:2/invoke",
        ),
    ],
)
def test_region_prefixed_invoke_model_calls_that_region_with_the_bare_model_id(
    monkeypatch: pytest.MonkeyPatch, model: str, aws_region_name: str | None, expected_url: str
):
    monkeypatch.setenv("AWS_REGION_NAME", "us-west-2")
    monkeypatch.delenv("AWS_BEDROCK_RUNTIME_ENDPOINT", raising=False)
    assert _invoke_with_mocked_post(model, aws_region_name) == expected_url
