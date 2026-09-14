from litellm.litellm_core_utils.internal_params import LiteLLMInternalParam
import json
from unittest.mock import MagicMock

import httpx
import pytest


from litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeConfig,
)
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)
from litellm.llms.bedrock.common_utils import BedrockError


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
    seeded.update({"temperature": 0.5, "cache_control_injection_points": [{"location": "tool_config"}]})

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


def test_nova_invoke_preserves_tool_cache_points_without_leaking_internal_params():
    from copy import deepcopy

    params = {
        "tools": [{"toolSpec": {"name": "lookup", "description": "Lookup", "inputSchema": {"json": {"type": "object", "properties": {}}}}}],
        "cache_control_injection_points": [{"location": "tool_config"}],
        "skip_mcp_handler": True,
        "stream_chunk_size": 7,
    }
    original = deepcopy(params)
    body = AmazonInvokeConfig().transform_request(
        model="amazon.nova-pro-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        optional_params=params,
        litellm_params={},
        headers={},
    )
    assert body["toolConfig"]["tools"][-1] == {"cachePoint": {"type": "default"}}
    assert not any(param.value in json.dumps(body) for param in LiteLLMInternalParam)
    assert params == original
