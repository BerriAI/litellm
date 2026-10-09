"""Bedrock Runtime Chat Completions: the default for GPT 5.6 and newer, ``bedrock/chat_completions/<model>`` for the rest."""

import json
import re

import httpx
import pytest
from pydantic import BaseModel

import litellm
from litellm.llms.bedrock.chat.chat_completions.transformation import (
    AmazonBedrockRuntimeChatCompletionsConfig,
    BedrockRuntimeChatCompletionsStreamingHandler,
    ReasoningTagSplitter,
    chat_completions_reasoning_efforts_refused_for,
    split_reasoning_tag,
    with_max_completion_tokens,
)
from litellm.llms.bedrock.common_utils import (
    BEDROCK_CONVERSE_ONLY_REQUEST_KEYS,
    BedrockModelInfo,
    bedrock_request_needs_converse,
    bedrock_route_for_request,
    bedrock_runtime_chat_completions_is_default,
    get_bedrock_chat_config,
)
from litellm.llms.custom_httpx.http_handler import HTTPHandler

APPLICATION_INFERENCE_PROFILE_ARN = "arn:aws:bedrock:us-west-2:123412341234:application-inference-profile/a1b2c3"


@pytest.fixture
def local_cost_map(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


@pytest.mark.parametrize(
    "model",
    [
        "chat_completions/us.xai.grok-4.6",
        "chat_completions/global.xai.grok-4.6",
        "chat_completions/us-gov.xai.grok-4.6",
        "bedrock/chat_completions/us.xai.grok-4.6",
    ],
)
def test_chat_completions_prefix_opts_grok_into_the_native_route(local_cost_map, model):
    assert BedrockModelInfo.get_bedrock_route(model) == "chat_completions"
    assert isinstance(get_bedrock_chat_config(model), AmazonBedrockRuntimeChatCompletionsConfig)


def test_explicit_converse_prefix_still_uses_converse(local_cost_map):
    assert BedrockModelInfo.get_bedrock_route("bedrock/converse/us.xai.grok-4.6") == "converse"
    assert BedrockModelInfo.get_bedrock_route("converse/us.xai.grok-4.6") == "converse"


def test_claude_stays_on_converse(local_cost_map):
    assert BedrockModelInfo.get_bedrock_route("us.anthropic.claude-3-sonnet-20240229-v1:0") == "converse"


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/openai.gpt-oss-20b-1:0",
        "openai.gpt-oss-120b-1:0",
        "global.openai.gpt-5.5",
        "bedrock/us.openai.gpt-5.4",
        "bedrock/us-gov-west-1/openai.gpt-oss-20b-1:0",
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.openai.gpt-6-astra",
        "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/abc123xyz",
    ],
)
def test_models_without_the_prefix_stay_on_converse(local_cost_map, model):
    assert BedrockModelInfo.get_bedrock_route(model) == "converse"
    assert BedrockModelInfo.get_bedrock_route(model, {}) == "converse"
    assert isinstance(get_bedrock_chat_config(model), litellm.AmazonConverseConfig)


def test_cost_map_row_listing_chat_completions_leaves_the_default_route_alone(monkeypatch):
    entry = {
        "litellm_provider": "bedrock_converse",
        "supported_endpoints": ["/v1/chat/completions", "/v1/responses"],
        "supports_bedrock_runtime_chat_completions_tools_with_reasoning": True,
        "supports_bedrock_runtime_chat_completions_response_format": True,
    }
    monkeypatch.setattr(litellm, "model_cost", {"openai.gpt-oss-20b-1:0": entry})
    assert BedrockModelInfo.get_bedrock_route("bedrock/openai.gpt-oss-20b-1:0", {}) == "converse"
    assert (
        BedrockModelInfo.get_bedrock_route("bedrock/chat_completions/openai.gpt-oss-20b-1:0", {}) == "chat_completions"
    )


@pytest.mark.parametrize(
    "model, supported_endpoints, expected_route",
    [
        ("global.openai.gpt-5.5", ["/v1/chat/completions", "/v1/responses"], "converse"),
        ("us.openai.gpt-5.6-sol", ["/v1/chat/completions", "/v1/responses"], "chat_completions"),
        ("us.openai.gpt-5.6-sol", ["/v1/responses"], "converse"),
        ("global.openai.gpt-6-sol", ["/v1/chat/completions", "/v1/responses"], "chat_completions"),
        ("global.openai.gpt-6-sol", ["/v1/responses"], "converse"),
        ("global.openai.gpt-6-sol", [], "converse"),
        ("us.openai.gpt-6.1-sol", ["/v1/chat/completions"], "chat_completions"),
        ("global.openai.gpt-10-sol", ["/v1/chat/completions"], "chat_completions"),
        ("openai.gpt-oss-120b-1:0", ["/v1/chat/completions"], "converse"),
        ("us.xai.grok-4.6", ["/v1/chat/completions"], "chat_completions"),
        ("global.xai.grok-4.7", ["/v1/chat/completions"], "chat_completions"),
        ("global.xai.grok-4.7", ["/v1/responses"], "converse"),
        ("global.xai.grok-4.7", [], "converse"),
        ("us-gov.xai.grok-4.6", ["/v1/chat/completions"], "chat_completions"),
    ],
)
def test_default_route_needs_grok_or_gpt_56_or_newer_and_a_row_listing_chat_completions(
    monkeypatch, model, supported_endpoints, expected_route
):
    entry = {"litellm_provider": "bedrock_converse", "supported_endpoints": supported_endpoints}
    monkeypatch.setattr(litellm, "model_cost", {model: entry})
    assert bedrock_runtime_chat_completions_is_default(model) is (expected_route == "chat_completions")
    assert BedrockModelInfo.get_bedrock_route(f"bedrock/{model}", {}) == expected_route
    assert BedrockModelInfo.get_bedrock_route(f"bedrock/chat_completions/{model}", {}) == "chat_completions"
    assert BedrockModelInfo.get_bedrock_route(f"bedrock/converse/{model}", {}) == "converse"


@pytest.mark.parametrize("model", ["global.openai.gpt-5.6-sol", "openai.gpt-oss-20b-1:0", "us.xai.grok-4.6"])
def test_chat_completions_prefix_prices_like_the_bare_model(local_cost_map, model):
    prefixed = litellm.get_model_info(model=f"bedrock/chat_completions/{model}")
    bare = litellm.get_model_info(model=f"bedrock/{model}")
    assert prefixed["input_cost_per_token"] == bare["input_cost_per_token"] > 0
    assert prefixed["output_cost_per_token"] == bare["output_cost_per_token"] > 0


def test_complete_url_is_runtime_openai_chat_completions(monkeypatch):
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    monkeypatch.delenv("AWS_BEDROCK_RUNTIME_ENDPOINT", raising=False)
    cfg = AmazonBedrockRuntimeChatCompletionsConfig()
    url = cfg.get_complete_url(
        api_base=None,
        api_key=None,
        model="us.xai.grok-4.6",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions"


def test_complete_url_appends_to_openai_v1_base():
    cfg = AmazonBedrockRuntimeChatCompletionsConfig()
    url = cfg.get_complete_url(
        api_base="https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1",
        api_key=None,
        model="us.xai.grok-4.6",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions"


def test_complete_url_sends_to_the_runtime_endpoint_over_api_base_like_converse(monkeypatch):
    monkeypatch.delenv("AWS_BEDROCK_RUNTIME_ENDPOINT", raising=False)
    cfg = AmazonBedrockRuntimeChatCompletionsConfig()
    url = cfg.get_complete_url(
        api_base="https://signing-host.example.com",
        api_key=None,
        model="us.openai.gpt-5.6-sol",
        optional_params={"aws_region_name": "us-east-1", "aws_bedrock_runtime_endpoint": "https://egress.example.com/"},
        litellm_params={},
    )
    assert url == "https://egress.example.com/openai/v1/chat/completions"


def test_complete_url_sends_to_the_env_runtime_endpoint_over_api_base_like_converse(monkeypatch):
    monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", "https://env-egress.example.com")
    cfg = AmazonBedrockRuntimeChatCompletionsConfig()
    url = cfg.get_complete_url(
        api_base="https://signing-host.example.com",
        api_key=None,
        model="us.openai.gpt-5.6-sol",
        optional_params={"aws_region_name": "us-east-1"},
        litellm_params={},
    )
    assert url == "https://env-egress.example.com/openai/v1/chat/completions"


@pytest.mark.parametrize("digits", [4, 4301, 30000])
@pytest.mark.parametrize("template", ["openai.gpt-{run}", "us.openai.gpt-5.{run}", "openai.gpt-{run}.{run}-sol"])
def test_overlong_gpt_version_digits_route_to_converse_without_raising(local_cost_map, template, digits):
    model = template.format(run="9" * digits)
    assert bedrock_runtime_chat_completions_is_default(model) is False
    assert bedrock_route_for_request(model, {}, None) == "converse"


def test_project_id_is_not_sent_as_openai_project_header():
    cfg = AmazonBedrockRuntimeChatCompletionsConfig()
    headers = cfg.validate_environment(
        headers={},
        model="bedrock/chat_completions/openai.gpt-oss-20b-1:0",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={},
        litellm_params={"aws_bedrock_project_id": "proj_from_config"},
    )
    assert "OpenAI-Project" not in headers
    assert headers["Content-Type"] == "application/json"


def test_transform_request_is_openai_chat_body_not_converse():
    cfg = AmazonBedrockRuntimeChatCompletionsConfig()
    body = cfg.transform_request(
        model="bedrock/chat_completions/us.xai.grok-4.6",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={"temperature": 0.2, "aws_region_name": "us-east-1"},
        litellm_params={},
        headers={},
    )
    assert body["model"] == "us.xai.grok-4.6"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["temperature"] == 0.2
    assert "aws_region_name" not in body
    assert "inferenceConfig" not in body
    assert "messages" in body


def _chat_completion_json(content, model, tool_calls=None):
    message = {"role": "assistant", "content": content, **({"tool_calls": tool_calls} if tool_calls else {})}
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1733529600,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


CONVERSE_JSON = {
    "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
}


@pytest.fixture
def fake_aws_env(monkeypatch):
    monkeypatch.setenv("AWS_REGION_NAME", "us-west-2")
    monkeypatch.delenv("AWS_BEDROCK_RUNTIME_ENDPOINT", raising=False)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")


def _recording_client(**response_kwargs):
    requests: list[httpx.Request] = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, **response_kwargs)

    return requests, HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handle)))


@pytest.mark.parametrize(
    "model, model_path",
    [
        ("bedrock/converse/us.xai.grok-4.6", b"/model/us.xai.grok-4.6/converse"),
        ("bedrock/openai.gpt-oss-20b-1:0", b"/model/openai.gpt-oss-20b-1%3A0/converse"),
        ("bedrock/global.openai.gpt-5.5", b"/model/global.openai.gpt-5.5/converse"),
    ],
)
def test_completion_without_the_prefix_posts_converse(local_cost_map, fake_aws_env, model, model_path):
    requests, client = _recording_client(json=CONVERSE_JSON)
    response = litellm.completion(model=model, messages=[{"role": "user", "content": "hello"}], client=client)

    assert response.choices[0].message.content == "ok"
    assert [request.url.raw_path for request in requests] == [model_path]


def test_completion_posts_runtime_chat_completions(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=_chat_completion_json("ok", "us.xai.grok-4.6"))
    response = litellm.completion(
        model="bedrock/chat_completions/us.xai.grok-4.6",
        messages=[{"role": "user", "content": "hello"}],
        client=client,
    )

    assert response.choices[0].message.content == "ok"
    assert len(requests) == 1
    assert str(requests[0].url) == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions"
    body = json.loads(requests[0].content)
    assert body["model"] == "us.xai.grok-4.6"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert "inferenceConfig" not in body


def test_completion_keeps_the_aws_request_id_as_a_provider_header(local_cost_map, fake_aws_env):
    _, client = _recording_client(
        json=_chat_completion_json("ok", "us.xai.grok-4.6"), headers={"x-amzn-requestid": "req-native-1"}
    )
    response = litellm.completion(
        model="bedrock/chat_completions/us.xai.grok-4.6",
        messages=[{"role": "user", "content": "hello"}],
        client=client,
    )

    assert response._hidden_params["additional_headers"]["llm_provider-x-amzn-requestid"] == "req-native-1"


def test_region_path_sends_the_bare_model_id_to_the_path_region(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=_chat_completion_json("ok", "openai.gpt-oss-20b-1:0"))
    litellm.completion(
        model="bedrock/chat_completions/us-gov-west-1/openai.gpt-oss-20b-1:0",
        messages=[{"role": "user", "content": "hello"}],
        client=client,
    )

    assert str(requests[0].url) == "https://bedrock-runtime.us-gov-west-1.amazonaws.com/openai/v1/chat/completions"
    assert json.loads(requests[0].content)["model"] == "openai.gpt-oss-20b-1:0"
    assert "/us-gov-west-1/bedrock/aws4_request" in requests[0].headers["Authorization"]


def test_explicit_aws_region_name_wins_over_the_region_path(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=_chat_completion_json("ok", "openai.gpt-oss-20b-1:0"))
    litellm.completion(
        model="bedrock/chat_completions/us-gov-west-1/openai.gpt-oss-20b-1:0",
        messages=[{"role": "user", "content": "hello"}],
        aws_region_name="us-gov-east-1",
        client=client,
    )

    assert str(requests[0].url) == "https://bedrock-runtime.us-gov-east-1.amazonaws.com/openai/v1/chat/completions"
    assert json.loads(requests[0].content)["model"] == "openai.gpt-oss-20b-1:0"
    assert "/us-gov-east-1/bedrock/aws4_request" in requests[0].headers["Authorization"]


def test_region_path_falls_back_to_converse_in_the_path_region(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=CONVERSE_JSON)
    litellm.completion(
        model="bedrock/chat_completions/us-gov-west-1/openai.gpt-oss-20b-1:0",
        messages=[{"role": "user", "content": "hello"}],
        stop=["END"],
        client=client,
    )

    assert requests[0].url.host == "bedrock-runtime.us-gov-west-1.amazonaws.com"
    assert requests[0].url.raw_path == b"/model/openai.gpt-oss-20b-1%3A0/converse"
    assert json.loads(requests[0].content)["inferenceConfig"]["stopSequences"] == ["END"]
    assert "/us-gov-west-1/bedrock/aws4_request" in requests[0].headers["Authorization"]


OPENAI_RUNTIME_MODELS = (
    "openai.gpt-oss-20b-1:0",
    "openai.gpt-oss-120b-1:0",
    "us.openai.gpt-5.6-sol",
    "global.openai.gpt-5.6-sol",
    "us.openai.gpt-5.6-terra",
    "global.openai.gpt-5.6-terra",
    "us.openai.gpt-5.6-luna",
    "global.openai.gpt-5.6-luna",
)
GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


@pytest.mark.parametrize(
    "model",
    [
        *(f"chat_completions/{model}" for model in OPENAI_RUNTIME_MODELS),
        "bedrock/chat_completions/openai.gpt-oss-20b-1:0",
        "chat_completions/us-gov.openai.gpt-oss-20b-1:0",
        "bedrock/chat_completions/us-gov-west-1/openai.gpt-oss-20b-1:0",
        "chat_completions/us-gov-east-1/openai.gpt-oss-120b-1:0",
    ],
)
def test_openai_runtime_models_use_chat_completions_route(local_cost_map, model):
    assert BedrockModelInfo.get_bedrock_route(model) == "chat_completions"
    assert isinstance(get_bedrock_chat_config(model), AmazonBedrockRuntimeChatCompletionsConfig)


GPT_56_AND_NEWER_MODELS = (
    "global.openai.gpt-5.6-sol",
    "bedrock/us.openai.gpt-5.6-terra",
    "us.openai.gpt-5.6-luna",
    "bedrock/global.openai.gpt-6-astra",
    "us.openai.gpt-6-sol",
    "global.openai.gpt-6-luna",
    "bedrock/global.openai.gpt-6.1-sol",
    "us.openai.gpt-6.1-sol",
    "global.xai.grok-4.6",
    "bedrock/us.xai.grok-4.6",
    "global.xai.grok-4.7",
    "bedrock/us.xai.grok-4.7",
)


@pytest.mark.parametrize("model", GPT_56_AND_NEWER_MODELS)
def test_gpt_56_and_newer_default_to_chat_completions(local_cost_map, model):
    assert bedrock_runtime_chat_completions_is_default(model) is True
    assert BedrockModelInfo.get_bedrock_route(model) == "chat_completions"
    assert BedrockModelInfo.get_bedrock_route(model, {}) == "chat_completions"
    assert isinstance(get_bedrock_chat_config(model), AmazonBedrockRuntimeChatCompletionsConfig)


@pytest.mark.parametrize("model", ["us.amazon.nova-micro-v1:0", "us.anthropic.claude-haiku-4-5-20251001-v1:0"])
def test_nova_and_claude_stay_on_converse(local_cost_map, model):
    assert BedrockModelInfo.get_bedrock_route(model, {"tools": [GET_WEATHER_TOOL]}) == "converse"


@pytest.mark.parametrize(
    "model",
    [
        "chat_completions/openai.gpt-oss-20b-1:0",
        "bedrock/chat_completions/global.openai.gpt-5.6-sol",
        "bedrock/us.openai.gpt-5.6-sol",
        "global.openai.gpt-6-sol",
        "us.openai.gpt-6.1-sol",
    ],
)
def test_guardrail_config_falls_back_to_converse(local_cost_map, model):
    guardrail = {"guardrailIdentifier": "gr-1", "guardrailVersion": "1"}
    assert bedrock_request_needs_converse(model, {"guardrailConfig": guardrail}) is True
    assert BedrockModelInfo.get_bedrock_route(model, {"guardrailConfig": guardrail}) == "converse"
    assert BedrockModelInfo.get_bedrock_route(model, {"guardrailConfig": None}) == "chat_completions"


@pytest.mark.parametrize(
    "model",
    [
        "chat_completions/openai.gpt-oss-20b-1:0",
        "chat_completions/us.xai.grok-4.6",
        "bedrock/chat_completions/global.openai.gpt-5.6-sol",
    ],
)
@pytest.mark.parametrize(
    "request_params",
    [
        {"additionalModelRequestFields": {"reasoning_effort": "high"}},
        {"top_k": 40},
        {"model_id": APPLICATION_INFERENCE_PROFILE_ARN},
    ],
    ids=["additionalModelRequestFields", "top_k", "model_id"],
)
def test_converse_extension_params_fall_back_to_converse(local_cost_map, model, request_params):
    assert bedrock_request_needs_converse(model, request_params) is True
    assert BedrockModelInfo.get_bedrock_route(model, request_params) == "converse"
    assert BedrockModelInfo.get_bedrock_route(model, {key: None for key in request_params}) == "chat_completions"


@pytest.mark.parametrize(
    "model",
    ["chat_completions/openai.gpt-oss-20b-1:0", "bedrock/global.openai.gpt-5.6-sol", "bedrock/us.openai.gpt-6.1-sol"],
)
def test_stop_keeps_other_models_on_converse(local_cost_map, model):
    assert bedrock_request_needs_converse(model, {"stop": ["END"]}) is True
    assert BedrockModelInfo.get_bedrock_route(model, {"stop": ["END"]}) == "converse"


@pytest.mark.parametrize("model", ["bedrock/global.xai.grok-4.7", "bedrock/us.xai.grok-4.6"])
def test_stop_is_dropped_on_chat_completions_for_grok(local_cost_map, fake_aws_env, model):
    requests, client = _recording_client(json=_chat_completion_json("ok", model.removeprefix("bedrock/")))
    response = litellm.completion(
        model=model, messages=[{"role": "user", "content": "hello"}], stop=["</block>"], max_tokens=64, client=client
    )

    assert requests[0].url.raw_path == b"/openai/v1/chat/completions"
    body = json.loads(requests[0].content)
    assert "stop" not in body
    assert body["max_completion_tokens"] == 64
    assert response.choices[0].message.content == "ok"


def test_stop_is_dropped_on_converse_for_grok(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=CONVERSE_JSON)
    litellm.completion(
        model="bedrock/converse/global.xai.grok-4.7",
        messages=[{"role": "user", "content": "hello"}],
        stop=["</block>"],
        max_tokens=64,
        client=client,
    )

    assert requests[0].url.raw_path == b"/model/global.xai.grok-4.7/converse"
    assert json.loads(requests[0].content)["inferenceConfig"] == {"maxTokens": 64}


@pytest.mark.parametrize(
    "model", ["bedrock/us.openai.gpt-5.6-sol", "global.openai.gpt-6-sol", "bedrock/chat_completions/us.xai.grok-4.6"]
)
def test_model_id_override_is_served_by_converse_like_the_arn_model_form(local_cost_map, model):
    assert bedrock_route_for_request(model, {"model_id": APPLICATION_INFERENCE_PROFILE_ARN}, None) == "converse"
    assert bedrock_route_for_request(model, {"model_id": None}, None) == "chat_completions"


SIGV4_PARAMS = {
    "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
    "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "aws_region_name": "us-east-1",
}


@pytest.mark.parametrize("api_key", ["", None], ids=["blank", "absent"])
def test_blank_api_key_is_signed_with_sigv4_instead_of_an_empty_bearer(monkeypatch, api_key):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    cfg = AmazonBedrockRuntimeChatCompletionsConfig()
    url = "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions"
    headers = cfg.validate_environment(
        headers={},
        model="bedrock/us.openai.gpt-5.6-sol",
        messages=[{"role": "user", "content": "hello"}],
        optional_params=dict(SIGV4_PARAMS),
        litellm_params={},
        api_key=api_key,
    )
    assert "Authorization" not in headers
    signed, _ = cfg.sign_request(
        headers=headers,
        optional_params=dict(SIGV4_PARAMS),
        request_data={"model": "us.openai.gpt-5.6-sol", "messages": []},
        api_base=url,
        api_key=api_key,
    )
    assert signed["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/"), signed


def test_bearer_api_key_is_sent_as_the_authorization_header(monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    cfg = AmazonBedrockRuntimeChatCompletionsConfig()
    headers = cfg.validate_environment(
        headers={},
        model="bedrock/us.openai.gpt-5.6-sol",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={},
        litellm_params={},
        api_key="bedrock-api-key",
    )
    assert headers["Authorization"] == "Bearer bedrock-api-key"


@pytest.mark.parametrize(
    "request_params, expected_route",
    [
        ({"tools": [GET_WEATHER_TOOL]}, "converse"),
        ({"tools": [GET_WEATHER_TOOL], "reasoning_effort": "low"}, "converse"),
        ({"tools": [GET_WEATHER_TOOL], "reasoning_effort": None}, "converse"),
        ({"tools": [GET_WEATHER_TOOL], "reasoning_effort": "none"}, "chat_completions"),
        ({"reasoning_effort": "low"}, "chat_completions"),
        ({"tools": None, "reasoning_effort": "low"}, "chat_completions"),
        ({"tools": [], "reasoning_effort": "low"}, "chat_completions"),
        ({}, "chat_completions"),
    ],
)
def test_gpt56_tools_need_reasoning_none_on_chat_completions(local_cost_map, request_params, expected_route):
    assert (
        BedrockModelInfo.get_bedrock_route("chat_completions/global.openai.gpt-5.6-sol", request_params)
        == expected_route
    )
    assert (
        BedrockModelInfo.get_bedrock_route("bedrock/chat_completions/us.openai.gpt-5.6-terra", request_params)
        == expected_route
    )
    assert BedrockModelInfo.get_bedrock_route("bedrock/us.openai.gpt-5.6-sol", request_params) == expected_route
    assert BedrockModelInfo.get_bedrock_route("global.openai.gpt-6-sol", request_params) == expected_route
    assert BedrockModelInfo.get_bedrock_route("bedrock/us.openai.gpt-6.1-sol", request_params) == expected_route


@pytest.mark.parametrize("reasoning_effort", ["low", "high", None])
def test_gpt_oss_tools_with_any_reasoning_effort_stay_on_chat_completions(local_cost_map, reasoning_effort):
    params = {"tools": [GET_WEATHER_TOOL], "reasoning_effort": reasoning_effort}
    assert bedrock_request_needs_converse("openai.gpt-oss-120b-1:0", params) is False
    assert BedrockModelInfo.get_bedrock_route("chat_completions/openai.gpt-oss-120b-1:0", params) == "chat_completions"


@pytest.mark.parametrize(
    "request_params, expected_route",
    [
        ({"functions": [GET_WEATHER_TOOL["function"]]}, "converse"),
        ({"functions": [GET_WEATHER_TOOL["function"]], "reasoning_effort": "low"}, "converse"),
        ({"functions": [GET_WEATHER_TOOL["function"]], "reasoning_effort": "none"}, "chat_completions"),
        ({"functions": [], "reasoning_effort": "low"}, "chat_completions"),
    ],
)
def test_gpt56_legacy_functions_route_like_tools(local_cost_map, request_params, expected_route):
    assert (
        BedrockModelInfo.get_bedrock_route("chat_completions/global.openai.gpt-5.6-sol", request_params)
        == expected_route
    )
    assert (
        BedrockModelInfo.get_bedrock_route("chat_completions/openai.gpt-oss-120b-1:0", request_params)
        == "chat_completions"
    )


def test_thinking_block_goes_to_converse(local_cost_map):
    thinking = {"type": "enabled", "budget_tokens": 1024}
    assert BedrockModelInfo.get_bedrock_route("chat_completions/us.xai.grok-4.6", {"thinking": thinking}) == "converse"
    assert (
        BedrockModelInfo.get_bedrock_route("chat_completions/us.xai.grok-4.6", {"thinking": None}) == "chat_completions"
    )


def test_explicit_converse_prefix_wins_for_openai_models(local_cost_map):
    assert BedrockModelInfo.get_bedrock_route("bedrock/converse/openai.gpt-oss-20b-1:0") == "converse"
    assert BedrockModelInfo.get_bedrock_route("converse/global.openai.gpt-5.6-sol", {}) == "converse"
    assert BedrockModelInfo.get_bedrock_route("bedrock/converse/global.openai.gpt-6-sol", {}) == "converse"
    assert isinstance(get_bedrock_chat_config("bedrock/converse/global.openai.gpt-6-sol"), litellm.AmazonConverseConfig)


def test_map_openai_params_sends_max_tokens_as_max_completion_tokens():
    cfg = AmazonBedrockRuntimeChatCompletionsConfig()
    mapped = cfg.map_openai_params(
        non_default_params={"max_tokens": 64, "temperature": 0.1},
        optional_params={},
        model="us.xai.grok-4.6",
        drop_params=False,
    )
    assert mapped == {"max_completion_tokens": 64, "temperature": 0.1}


HTTPS_IMAGE_URL = "https://example.com/cat.png"
IMAGE_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": HTTPS_IMAGE_URL},
            {"type": "image_url", "image_url": {"url": HTTPS_IMAGE_URL, "detail": "high"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            {"type": "image_url", "image_url": {"url": "s3://bucket/key.png"}},
        ],
    }
]


def _assert_remote_images_inlined(content):
    assert content[0] == {"type": "text", "text": "what is this"}
    assert content[1]["image_url"]["url"] == f"data:image/png;base64,{HTTPS_IMAGE_URL}"
    assert content[2] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{HTTPS_IMAGE_URL}", "detail": "high"},
    }
    assert content[3]["image_url"]["url"] == "data:image/png;base64,AAA"
    assert content[4]["image_url"]["url"] == "s3://bucket/key.png"


def test_transform_request_inlines_remote_image_urls(local_cost_map, monkeypatch):
    import litellm.litellm_core_utils.prompt_templates.image_handling as image_handling

    monkeypatch.setattr(image_handling, "convert_url_to_base64", lambda url: f"data:image/png;base64,{url}")
    body = AmazonBedrockRuntimeChatCompletionsConfig().transform_request(
        model="us.xai.grok-4.6",
        messages=IMAGE_MESSAGES,
        optional_params={},
        litellm_params={},
        headers={},
    )

    _assert_remote_images_inlined(body["messages"][0]["content"])


async def test_async_transform_request_inlines_remote_image_urls(local_cost_map, monkeypatch):
    import litellm.litellm_core_utils.prompt_templates.image_handling as image_handling

    async def fake_convert(url):
        return f"data:image/png;base64,{url}"

    monkeypatch.setattr(image_handling, "async_convert_url_to_base64", fake_convert)
    cfg = AmazonBedrockRuntimeChatCompletionsConfig()
    assert cfg.uses_async_transform_request is True
    body = await cfg.async_transform_request(
        model="us.xai.grok-4.6",
        messages=IMAGE_MESSAGES,
        optional_params={},
        litellm_params={},
        headers={},
    )

    _assert_remote_images_inlined(body["messages"][0]["content"])


def test_map_openai_params_keeps_explicit_max_completion_tokens():
    cfg = AmazonBedrockRuntimeChatCompletionsConfig()
    mapped = cfg.map_openai_params(
        non_default_params={"max_tokens": 64, "max_completion_tokens": 32},
        optional_params={},
        model="openai.gpt-oss-20b-1:0",
        drop_params=False,
    )
    assert mapped == {"max_completion_tokens": 32}


def test_with_max_completion_tokens_leaves_other_params_alone():
    assert with_max_completion_tokens({"temperature": 0.5}) == {"temperature": 0.5}


@pytest.mark.parametrize(
    "model",
    ["us.xai.grok-4.6", "bedrock/us-gov-west-1/us.xai.grok-4.6"],
)
def test_map_openai_params_drops_reasoning_effort_none_for_grok(model):
    cfg = AmazonBedrockRuntimeChatCompletionsConfig()
    mapped = cfg.map_openai_params(
        non_default_params={"reasoning_effort": "none", "max_tokens": 64},
        optional_params={},
        model=model,
        drop_params=False,
    )
    assert "reasoning_effort" not in mapped


def test_map_openai_params_keeps_reasoning_effort_low_for_grok():
    cfg = AmazonBedrockRuntimeChatCompletionsConfig()
    mapped = cfg.map_openai_params(
        non_default_params={"reasoning_effort": "low", "max_tokens": 64},
        optional_params={},
        model="us.xai.grok-4.6",
        drop_params=False,
    )
    assert mapped["reasoning_effort"] == "low"


@pytest.mark.parametrize("model", ["us.xai.grok-4.6", "global.openai.gpt-5.6-sol"])
@pytest.mark.parametrize("reasoning_effort", [["low"], {"effort": "low"}, 5], ids=["list", "object", "int"])
def test_map_openai_params_refuses_a_non_string_reasoning_effort_without_drop_params(model, reasoning_effort):
    cfg = AmazonBedrockRuntimeChatCompletionsConfig()
    with pytest.raises(litellm.UnsupportedParamsError, match="drop_params") as refused:
        cfg.map_openai_params(
            non_default_params={"reasoning_effort": reasoning_effort, "max_tokens": 64},
            optional_params={},
            model=model,
            drop_params=False,
        )
    assert refused.value.status_code == 400
    assert type(reasoning_effort).__name__ in str(refused.value)


@pytest.mark.parametrize("model", ["us.xai.grok-4.6", "global.openai.gpt-5.6-sol"])
@pytest.mark.parametrize("reasoning_effort", [["low"], {"effort": "low"}, 5], ids=["list", "object", "int"])
@pytest.mark.parametrize("drop_params_via", ["request", "litellm.drop_params"])
def test_map_openai_params_drops_a_non_string_reasoning_effort_under_drop_params(
    monkeypatch, model, reasoning_effort, drop_params_via
):
    monkeypatch.setattr(litellm, "drop_params", drop_params_via == "litellm.drop_params")
    mapped = AmazonBedrockRuntimeChatCompletionsConfig().map_openai_params(
        non_default_params={"reasoning_effort": reasoning_effort, "max_tokens": 64},
        optional_params={},
        model=model,
        drop_params=drop_params_via == "request",
    )
    assert "reasoning_effort" not in mapped
    assert mapped["max_completion_tokens"] == 64


def test_map_openai_params_keeps_reasoning_effort_none_for_gpt56():
    cfg = AmazonBedrockRuntimeChatCompletionsConfig()
    mapped = cfg.map_openai_params(
        non_default_params={"reasoning_effort": "none", "max_tokens": 64},
        optional_params={},
        model="global.openai.gpt-5.6-sol",
        drop_params=False,
    )
    assert mapped["reasoning_effort"] == "none"


def test_reasoning_efforts_refused_for_is_empty_outside_xai():
    assert chat_completions_reasoning_efforts_refused_for("openai.gpt-oss-20b-1:0") == frozenset()


def test_supported_params_include_reasoning_effort_for_gpt56(local_cost_map):
    cfg = AmazonBedrockRuntimeChatCompletionsConfig()
    assert "reasoning_effort" in cfg.get_supported_openai_params("global.openai.gpt-5.6-sol")
    assert "reasoning_effort" in cfg.get_supported_openai_params("openai.gpt-oss-20b-1:0")


@pytest.mark.parametrize(
    "model, refused, kept",
    [
        (
            "bedrock/global.openai.gpt-5.6-sol",
            ("n",),
            ("temperature", "top_p", "frequency_penalty", "logprobs", "logit_bias", "reasoning_effort", "stop"),
        ),
        (
            "bedrock/us.openai.gpt-6.1-sol",
            ("n",),
            ("temperature", "top_p", "presence_penalty", "top_logprobs", "reasoning_effort", "tools", "functions"),
        ),
        (
            "us.xai.grok-4.6",
            ("frequency_penalty", "presence_penalty", "n"),
            ("stop", "logprobs", "temperature", "top_p", "logit_bias", "reasoning_effort"),
        ),
        (
            "bedrock/us-gov-west-1/openai.gpt-oss-20b-1:0",
            ("logit_bias", "n"),
            ("frequency_penalty", "presence_penalty", "stop", "logprobs", "reasoning_effort"),
        ),
    ],
)
def test_supported_params_leave_out_what_each_family_refuses(local_cost_map, model, refused, kept):
    supported = set(AmazonBedrockRuntimeChatCompletionsConfig().get_supported_openai_params(model))
    assert supported.isdisjoint(refused)
    assert set(kept) <= supported


@pytest.mark.parametrize(
    "model, param",
    [
        ("bedrock/chat_completions/us.xai.grok-4.6", {"presence_penalty": 0.5}),
        ("bedrock/chat_completions/openai.gpt-oss-20b-1:0", {"logit_bias": {"1": 1}}),
    ],
    ids=lambda value: value if isinstance(value, str) else next(iter(value)),
)
def test_refused_params_are_dropped_or_refused_before_reaching_aws(local_cost_map, fake_aws_env, model, param):
    requests, client = _recording_client(
        json=_chat_completion_json("ok", model.removeprefix("bedrock/chat_completions/"))
    )
    with pytest.raises(litellm.UnsupportedParamsError, match=next(iter(param))):
        litellm.completion(model=model, messages=[{"role": "user", "content": "hello"}], client=client, **param)
    litellm.completion(
        model=model, messages=[{"role": "user", "content": "hello"}], drop_params=True, client=client, **param
    )

    assert str(requests[0].url).endswith("/openai/v1/chat/completions")
    assert param.keys().isdisjoint(json.loads(requests[0].content))


@pytest.mark.parametrize("reasoning_effort", [3, ["high"]], ids=["int", "list"])
def test_non_string_reasoning_effort_is_refused_or_dropped_before_reaching_aws(
    local_cost_map, fake_aws_env, reasoning_effort
):
    requests, client = _recording_client(json=_chat_completion_json("ok", "global.openai.gpt-5.6-sol"))
    request = {
        "model": "bedrock/global.openai.gpt-5.6-sol",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning_effort": reasoning_effort,
        "client": client,
    }
    with pytest.raises(litellm.UnsupportedParamsError, match="reasoning_effort") as refused:
        litellm.completion(**request)
    assert refused.value.status_code == 400
    assert requests == []

    litellm.completion(**request, drop_params=True)

    assert str(requests[0].url).endswith("/openai/v1/chat/completions")
    assert "reasoning_effort" not in json.loads(requests[0].content)


GPT_PARAMS_TIED_TO_REASONING_OFF = {
    "temperature": 0.2,
    "top_p": 0.9,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.5,
    "logprobs": True,
    "top_logprobs": 2,
}


@pytest.mark.parametrize("model", ["bedrock/global.openai.gpt-5.6-sol", "bedrock/us.openai.gpt-6-sol"])
@pytest.mark.parametrize("reasoning", [{}, {"reasoning_effort": "low"}], ids=["effort_unset", "effort_low"])
@pytest.mark.parametrize("param", list(GPT_PARAMS_TIED_TO_REASONING_OFF))
def test_gpt_sampling_params_are_refused_or_dropped_while_reasoning(
    local_cost_map, fake_aws_env, model, reasoning, param
):
    requests, client = _recording_client(json=_chat_completion_json("ok", model.removeprefix("bedrock/")))
    request = {"model": model, "messages": [{"role": "user", "content": "hello"}], "client": client, **reasoning}
    with pytest.raises(litellm.UnsupportedParamsError, match=param):
        litellm.completion(**request, **{param: GPT_PARAMS_TIED_TO_REASONING_OFF[param]})
    litellm.completion(**request, drop_params=True, **{param: GPT_PARAMS_TIED_TO_REASONING_OFF[param]})

    body = json.loads(requests[0].content)
    assert str(requests[0].url).endswith("/openai/v1/chat/completions")
    assert param not in body
    assert body.get("reasoning_effort") == reasoning.get("reasoning_effort")


@pytest.mark.parametrize("model", ["bedrock/global.openai.gpt-5.6-sol", "bedrock/us.openai.gpt-6-sol"])
def test_gpt_sampling_params_reach_aws_with_reasoning_effort_none(local_cost_map, fake_aws_env, model):
    requests, client = _recording_client(json=_chat_completion_json("ok", model.removeprefix("bedrock/")))
    litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        reasoning_effort="none",
        client=client,
        **GPT_PARAMS_TIED_TO_REASONING_OFF,
    )

    body = json.loads(requests[0].content)
    assert str(requests[0].url).endswith("/openai/v1/chat/completions")
    assert body["reasoning_effort"] == "none"
    assert {key: body[key] for key in GPT_PARAMS_TIED_TO_REASONING_OFF} == GPT_PARAMS_TIED_TO_REASONING_OFF


def test_split_reasoning_tag_splits_leading_tag():
    assert split_reasoning_tag("<reasoning>plan it\n</reasoning>\n\nHello") == ("plan it\n", "Hello")


def test_split_reasoning_tag_drops_an_empty_tag():
    assert split_reasoning_tag("<reasoning></reasoning>Hello") == (None, "Hello")


@pytest.mark.parametrize(
    "content",
    [
        "<reasoning>plan it\n</reasoning>\n\nHello",
        "<reasoning>never closed",
        "<reas",
        "Hello <reasoning>later</reasoning>",
        "<reasoning></reasoning>",
    ],
)
@pytest.mark.parametrize("chunk_size", [1, 3, 7])
def test_split_reasoning_tag_matches_the_streamed_split(content, chunk_size):
    chunks = [content[start : start + chunk_size] for start in range(0, len(content), chunk_size)]
    streamed_reasoning, streamed_content = _run_splitter(chunks)

    assert split_reasoning_tag(content) == (streamed_reasoning or None, streamed_content)


def test_split_reasoning_tag_passes_plain_content_through():
    assert split_reasoning_tag("Hello") == (None, "Hello")


def test_split_reasoning_tag_ignores_tag_after_content_starts():
    content = "Hello <reasoning>not mine</reasoning>"
    assert split_reasoning_tag(content) == (None, content)


def _run_splitter(chunks):
    state = ReasoningTagSplitter()
    reasoning = ""
    content = ""
    for chunk in chunks:
        state, fed_reasoning, fed_content = state.feed(chunk)
        reasoning += fed_reasoning
        content += fed_content
    state, flushed_reasoning, flushed_content = state.flush()
    return reasoning + flushed_reasoning, content + flushed_content


def test_reasoning_tag_splitter_handles_tags_split_across_chunks():
    assert _run_splitter(["<reas", "oning>I think", " so</reas", "oning>\n\nHel", "lo"]) == ("I think so", "Hello")


def test_reasoning_tag_splitter_passes_plain_content_through():
    assert _run_splitter(["Hel", "lo <reasoning>later</reasoning>"]) == ("", "Hello <reasoning>later</reasoning>")


def test_reasoning_tag_splitter_flushes_unclosed_reasoning():
    assert _run_splitter(["<reasoning>never clo", "sed"]) == ("never closed", "")


def test_reasoning_tag_splitter_releases_a_false_tag_prefix():
    assert _run_splitter(["<", "b>x"]) == ("", "<b>x")


def _stream_chunk(delta, finish_reason=None, index=0, model="openai.gpt-oss-20b-1:0"):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1733529600,
        "model": model,
        "choices": [{"index": index, "delta": delta, "finish_reason": finish_reason}],
    }


def test_streaming_handler_splits_reasoning_deltas_per_choice(local_cost_map):
    handler = BedrockRuntimeChatCompletionsStreamingHandler(streaming_response=iter(()), sync_stream=True)

    first = handler.chunk_parser(_stream_chunk({"role": "assistant", "content": "<reasoning>I think"}))
    assert first.choices[0].delta.reasoning_content == "I think"
    assert not first.choices[0].delta.content

    second = handler.chunk_parser(_stream_chunk({"content": " so</reasoning>\n\nHello"}))
    assert second.choices[0].delta.reasoning_content == " so"
    assert second.choices[0].delta.content == "Hello"

    tool_call = {"index": 0, "id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
    third = handler.chunk_parser(_stream_chunk({"content": None, "tool_calls": [tool_call]}))
    assert third.choices[0].delta.tool_calls[0].function.name == "get_weather"

    last = handler.chunk_parser(_stream_chunk({}, finish_reason="stop"))
    assert last.choices[0].finish_reason == "stop"


def _reasoning_of(parsed):
    return getattr(parsed.choices[0].delta, "reasoning_content", None)


def test_streaming_handler_keeps_split_state_per_choice_index(local_cost_map):
    handler = BedrockRuntimeChatCompletionsStreamingHandler(streaming_response=iter(()), sync_stream=True)

    opened = handler.chunk_parser(_stream_chunk({"content": "<reasoning>first"}, index=0))
    assert _reasoning_of(opened) == "first"

    plain = handler.chunk_parser(_stream_chunk({"content": "plain answer"}, index=1))
    assert _reasoning_of(plain) is None
    assert plain.choices[0].delta.content == "plain answer"

    still_reasoning = handler.chunk_parser(_stream_chunk({"content": " more"}, index=0))
    assert _reasoning_of(still_reasoning) == " more"
    assert not still_reasoning.choices[0].delta.content


def test_streaming_handler_flushes_held_text_on_an_empty_final_delta(local_cost_map):
    handler = BedrockRuntimeChatCompletionsStreamingHandler(streaming_response=iter(()), sync_stream=True)

    held = handler.chunk_parser(_stream_chunk({"content": "<reas"}))
    assert not held.choices[0].delta.content

    final = handler.chunk_parser(_stream_chunk({}, finish_reason="stop"))
    assert final.choices[0].delta.content == "<reas"
    assert _reasoning_of(final) is None

    unclosed = BedrockRuntimeChatCompletionsStreamingHandler(streaming_response=iter(()), sync_stream=True)
    unclosed.chunk_parser(_stream_chunk({"content": "<reasoning>almost done</reas"}))
    drained = unclosed.chunk_parser(_stream_chunk({}, finish_reason="length"))
    assert _reasoning_of(drained) == "</reas"


def test_gpt_oss_completion_hits_chat_completions_and_splits_reasoning(local_cost_map, fake_aws_env):
    requests, client = _recording_client(
        json=_chat_completion_json("<reasoning>plan</reasoning>\n\nHi", "openai.gpt-oss-20b-1:0")
    )
    response = litellm.completion(
        model="bedrock/chat_completions/openai.gpt-oss-20b-1:0",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=64,
        reasoning_effort="low",
        tools=[GET_WEATHER_TOOL],
        client=client,
    )

    assert str(requests[0].url) == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions"
    body = json.loads(requests[0].content)
    assert body["model"] == "openai.gpt-oss-20b-1:0"
    assert body["max_completion_tokens"] == 64
    assert "max_tokens" not in body
    assert body["reasoning_effort"] == "low"
    assert body["tools"] == [GET_WEATHER_TOOL]
    assert response.choices[0].message.reasoning_content == "plan"
    assert response.choices[0].message.content == "Hi"


def test_gpt56_tools_with_reasoning_effort_go_to_converse(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=CONVERSE_JSON)
    response = litellm.completion(
        model="bedrock/chat_completions/global.openai.gpt-5.6-sol",
        messages=[{"role": "user", "content": "hello"}],
        tools=[GET_WEATHER_TOOL],
        reasoning_effort="low",
        client=client,
    )

    assert requests[0].url.raw_path.endswith(b"/model/global.openai.gpt-5.6-sol/converse")
    assert json.loads(requests[0].content)["toolConfig"]["tools"][0]["toolSpec"]["name"] == "get_weather"
    assert response.choices[0].message.content == "ok"


def test_gpt56_tools_with_reasoning_none_stay_on_chat_completions(local_cost_map, fake_aws_env):
    tool_calls = [
        {"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
    ]
    requests, client = _recording_client(json=_chat_completion_json(None, "global.openai.gpt-5.6-sol", tool_calls))
    response = litellm.completion(
        model="bedrock/chat_completions/global.openai.gpt-5.6-sol",
        messages=[{"role": "user", "content": "weather in Paris"}],
        tools=[GET_WEATHER_TOOL],
        reasoning_effort="none",
        max_tokens=64,
        client=client,
    )

    assert str(requests[0].url) == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions"
    body = json.loads(requests[0].content)
    assert body["tools"] == [GET_WEATHER_TOOL]
    assert body["reasoning_effort"] == "none"
    assert body["max_completion_tokens"] == 64
    assert response.choices[0].message.tool_calls[0].function.name == "get_weather"


@pytest.mark.parametrize("model", ["global.openai.gpt-6-sol", "us.openai.gpt-5.6-sol", "us.openai.gpt-6.1-sol"])
def test_gpt_56_and_newer_completion_without_the_prefix_posts_runtime_chat_completions(
    local_cost_map, fake_aws_env, model
):
    requests, client = _recording_client(json=_chat_completion_json("ok", model))
    response = litellm.completion(
        model=f"bedrock/{model}",
        messages=[{"role": "user", "content": "hello"}],
        reasoning_effort="low",
        client=client,
    )

    assert str(requests[0].url) == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions"
    body = json.loads(requests[0].content)
    assert body["model"] == model
    assert body["reasoning_effort"] == "low"
    assert "inferenceConfig" not in body
    assert response.choices[0].message.content == "ok"
    assert response._hidden_params["response_cost"] > 0


def test_gpt6_without_the_prefix_tools_with_reasoning_effort_go_to_converse(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=CONVERSE_JSON)
    response = litellm.completion(
        model="bedrock/global.openai.gpt-6-sol",
        messages=[{"role": "user", "content": "hello"}],
        tools=[GET_WEATHER_TOOL],
        reasoning_effort="low",
        client=client,
    )

    assert requests[0].url.raw_path.endswith(b"/model/global.openai.gpt-6-sol/converse")
    body = json.loads(requests[0].content)
    assert body["toolConfig"]["tools"][0]["toolSpec"]["name"] == "get_weather"
    assert body["additionalModelRequestFields"]["reasoning"] == {"effort": "low"}
    assert response.choices[0].message.content == "ok"


def test_gpt6_without_the_prefix_guardrail_config_goes_to_converse(local_cost_map, fake_aws_env):
    guardrail = {"guardrailIdentifier": "gr-1", "guardrailVersion": "1"}
    requests, client = _recording_client(json=CONVERSE_JSON)
    litellm.completion(
        model="bedrock/global.openai.gpt-6-sol",
        messages=[{"role": "user", "content": "hello"}],
        guardrailConfig=guardrail,
        client=client,
    )

    assert requests[0].url.raw_path.endswith(b"/model/global.openai.gpt-6-sol/converse")
    assert json.loads(requests[0].content)["guardrailConfig"] == guardrail


@pytest.mark.parametrize(
    "converse_only_param",
    [
        {"guardrailConfig": {"guardrailIdentifier": "gr-1", "guardrailVersion": "1"}},
        {"performanceConfig": {"latency": "optimized"}},
        {"requestMetadata": {"team": "search"}},
        {"serviceTier": {"type": "priority"}},
    ],
    ids=lambda param: next(iter(param)),
)
def test_converse_only_request_keys_go_to_converse(local_cost_map, fake_aws_env, converse_only_param):
    requests, client = _recording_client(json=CONVERSE_JSON)
    litellm.completion(
        model="bedrock/chat_completions/openai.gpt-oss-20b-1:0",
        messages=[{"role": "user", "content": "hello"}],
        client=client,
        **converse_only_param,
    )

    assert requests[0].url.raw_path.endswith(b"/model/openai.gpt-oss-20b-1%3A0/converse")
    ((key, value),) = converse_only_param.items()
    assert json.loads(requests[0].content)[key] == value


def test_converse_only_keys_cover_every_converse_config_block():
    assert set(litellm.AmazonConverseConfig.get_config_blocks()) <= BEDROCK_CONVERSE_ONLY_REQUEST_KEYS


def test_operator_owned_request_metadata_goes_to_converse(local_cost_map, fake_aws_env, monkeypatch):
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", ["user_api_key_team_alias"])
    requests, client = _recording_client(json=CONVERSE_JSON)
    litellm.completion(
        model="bedrock/chat_completions/openai.gpt-oss-20b-1:0",
        messages=[{"role": "user", "content": "hello"}],
        metadata={"user_api_key_team_alias": "search"},
        client=client,
    )

    assert requests[0].url.raw_path.endswith(b"/model/openai.gpt-oss-20b-1%3A0/converse")
    assert json.loads(requests[0].content)["requestMetadata"] == {"user_api_key_team_alias": "search"}


def test_dropped_converse_only_key_keeps_the_request_on_chat_completions(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=_chat_completion_json("ok", "openai.gpt-oss-20b-1:0"))
    litellm.completion(
        model="bedrock/chat_completions/openai.gpt-oss-20b-1:0",
        messages=[{"role": "user", "content": "hello"}],
        guardrailConfig={"guardrailIdentifier": "gr-1", "guardrailVersion": "1"},
        additional_drop_params=["guardrailConfig"],
        max_tokens=8,
        client=client,
    )

    assert str(requests[0].url) == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions"
    body = json.loads(requests[0].content)
    assert "guardrailConfig" not in body
    assert body["max_completion_tokens"] == 8
    assert "inferenceConfig" not in body


def test_dropped_tools_keep_gpt56_reasoning_request_on_chat_completions(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=_chat_completion_json("ok", "global.openai.gpt-5.6-sol"))
    litellm.completion(
        model="bedrock/chat_completions/global.openai.gpt-5.6-sol",
        messages=[{"role": "user", "content": "hello"}],
        tools=[GET_WEATHER_TOOL],
        reasoning_effort="low",
        additional_drop_params=["tools"],
        client=client,
    )

    assert str(requests[0].url) == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions"
    body = json.loads(requests[0].content)
    assert "tools" not in body
    assert body["reasoning_effort"] == "low"


def test_legacy_functions_stay_on_chat_completions(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=_chat_completion_json("ok", "openai.gpt-oss-20b-1:0"))
    litellm.completion(
        model="bedrock/chat_completions/openai.gpt-oss-20b-1:0",
        messages=[{"role": "user", "content": "hello"}],
        functions=[GET_WEATHER_TOOL["function"]],
        client=client,
    )

    assert str(requests[0].url) == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions"
    assert json.loads(requests[0].content)["functions"] == [GET_WEATHER_TOOL["function"]]


def test_gpt56_legacy_functions_with_reasoning_fall_back_to_converse(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=CONVERSE_JSON)
    with pytest.raises(litellm.UnsupportedParamsError, match="functions"):
        litellm.completion(
            model="bedrock/chat_completions/global.openai.gpt-5.6-sol",
            messages=[{"role": "user", "content": "hello"}],
            functions=[GET_WEATHER_TOOL["function"]],
            reasoning_effort="low",
            client=client,
        )
    litellm.completion(
        model="bedrock/chat_completions/global.openai.gpt-5.6-sol",
        messages=[{"role": "user", "content": "hello"}],
        functions=[GET_WEATHER_TOOL["function"]],
        reasoning_effort="low",
        drop_params=True,
        client=client,
    )

    assert requests[0].url.raw_path.endswith(b"/model/global.openai.gpt-5.6-sol/converse")
    body = json.loads(requests[0].content)
    assert "functions" not in body
    assert "toolConfig" not in body


def test_grok_thinking_block_is_served_by_converse(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=CONVERSE_JSON)
    thinking = {"type": "enabled", "budget_tokens": 1024}
    litellm.completion(
        model="bedrock/chat_completions/us.xai.grok-4.6",
        messages=[{"role": "user", "content": "hello"}],
        thinking=thinking,
        client=client,
    )

    assert requests[0].url.raw_path.endswith(b"/model/us.xai.grok-4.6/converse")
    assert json.loads(requests[0].content)["additionalModelRequestFields"]["thinking"] == thinking


def test_converse_fallback_validates_against_converse_params(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=CONVERSE_JSON)
    guardrail = {"guardrailIdentifier": "gr-1", "guardrailVersion": "1"}
    with pytest.raises(litellm.UnsupportedParamsError, match="seed"):
        litellm.completion(
            model="bedrock/chat_completions/openai.gpt-oss-20b-1:0",
            messages=[{"role": "user", "content": "hello"}],
            guardrailConfig=guardrail,
            seed=7,
            client=client,
        )
    litellm.completion(
        model="bedrock/chat_completions/openai.gpt-oss-20b-1:0",
        messages=[{"role": "user", "content": "hello"}],
        guardrailConfig=guardrail,
        seed=7,
        drop_params=True,
        client=client,
    )

    assert requests[0].url.raw_path.endswith(b"/model/openai.gpt-oss-20b-1%3A0/converse")
    assert "seed" not in json.loads(requests[0].content)


def test_n_is_rejected_before_reaching_chat_completions(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=_chat_completion_json("ok", "openai.gpt-oss-20b-1:0"))
    with pytest.raises(litellm.UnsupportedParamsError, match="'n'"):
        litellm.completion(
            model="bedrock/chat_completions/openai.gpt-oss-20b-1:0",
            messages=[{"role": "user", "content": "hello"}],
            n=2,
            client=client,
        )
    litellm.completion(
        model="bedrock/chat_completions/openai.gpt-oss-20b-1:0",
        messages=[{"role": "user", "content": "hello"}],
        n=2,
        drop_params=True,
        client=client,
    )

    assert "n" not in json.loads(requests[0].content)


def _sse(chunks):
    return ("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()


def test_gpt_oss_streaming_completion_splits_reasoning(local_cost_map, fake_aws_env):
    chunks = (
        _stream_chunk({"role": "assistant", "content": "<reasoning>plan"}),
        _stream_chunk({"content": "</reasoning>\n\nHi"}),
        _stream_chunk({}, finish_reason="stop"),
    )
    requests, client = _recording_client(content=_sse(chunks), headers={"content-type": "text/event-stream"})
    stream = litellm.completion(
        model="bedrock/chat_completions/openai.gpt-oss-20b-1:0",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        client=client,
    )
    deltas = [chunk.choices[0].delta for chunk in stream]

    assert [str(request.url) for request in requests] == [
        "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions"
    ]
    assert json.loads(requests[0].content)["stream"] is True
    assert "".join(getattr(delta, "reasoning_content", None) or "" for delta in deltas) == "plan"
    assert "".join(delta.content or "" for delta in deltas) == "Hi"


def test_streaming_handler_keeps_native_reasoning_next_to_the_tagged_split(local_cost_map):
    handler = BedrockRuntimeChatCompletionsStreamingHandler(streaming_response=iter(()), sync_stream=True)
    parsed = handler.chunk_parser(
        _stream_chunk({"reasoning": "native ", "content": "<reasoning>tagged</reasoning>Hi"}, finish_reason="stop")
    )

    assert parsed.choices[0].delta.reasoning_content == "native tagged"
    assert parsed.choices[0].delta.content == "Hi"


RESPONSE_FORMAT_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "schema": {"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"]},
        "strict": True,
    },
}


class Answer(BaseModel):
    word: str


@pytest.mark.parametrize(
    "model", ["chat_completions/openai.gpt-oss-20b-1:0", "bedrock/chat_completions/openai.gpt-oss-120b-1:0"]
)
@pytest.mark.parametrize(
    "response_format, expected_route",
    [
        (RESPONSE_FORMAT_JSON_SCHEMA, "converse"),
        ({"type": "json_object"}, "converse"),
        (Answer, "converse"),
        ({"type": "text"}, "chat_completions"),
        (None, "chat_completions"),
    ],
    ids=["json_schema", "json_object", "pydantic", "text", "none"],
)
def test_gpt_oss_response_format_falls_back_to_converse(local_cost_map, model, response_format, expected_route):
    params = {"response_format": response_format}
    assert bedrock_request_needs_converse(model, params) is (expected_route == "converse")
    assert BedrockModelInfo.get_bedrock_route(model, params) == expected_route


RESPONSE_FORMAT_ENFORCING_MODELS = [
    "chat_completions/global.openai.gpt-5.6-sol",
    "chat_completions/us.xai.grok-4.6",
    "bedrock/chat_completions/us-gov.xai.grok-4.6",
    "global.openai.gpt-6-sol",
    "bedrock/us.openai.gpt-6.1-sol",
]


JSON_OBJECT_WITH_RESPONSE_SCHEMA = {
    "type": "json_object",
    "response_schema": RESPONSE_FORMAT_JSON_SCHEMA["json_schema"]["schema"],
}


@pytest.mark.parametrize("model", RESPONSE_FORMAT_ENFORCING_MODELS)
@pytest.mark.parametrize("response_format", [RESPONSE_FORMAT_JSON_SCHEMA, Answer], ids=["json_schema", "pydantic"])
def test_json_schema_response_format_stays_on_chat_completions_where_aws_enforces_it(
    local_cost_map, model, response_format
):
    params = {"response_format": response_format}
    assert bedrock_request_needs_converse(model, params) is False
    assert BedrockModelInfo.get_bedrock_route(model, params) == "chat_completions"


@pytest.mark.parametrize("model", RESPONSE_FORMAT_ENFORCING_MODELS)
@pytest.mark.parametrize(
    "response_format",
    [{"type": "json_object"}, JSON_OBJECT_WITH_RESPONSE_SCHEMA],
    ids=["json_object", "json_object_with_response_schema"],
)
def test_json_object_keeps_converse_where_aws_would_demand_the_word_json(local_cost_map, model, response_format):
    params = {"response_format": response_format}
    assert bedrock_request_needs_converse(model, params) is True
    assert BedrockModelInfo.get_bedrock_route(model, params) == "converse"


SYNTHETIC_NATIVE_MODEL = "chat_completions/vendor.native-model-v1:0"


@pytest.mark.parametrize(
    "capability_flags, request_params, needs_converse",
    [
        ({}, {"tools": [GET_WEATHER_TOOL], "reasoning_effort": "low"}, True),
        ({}, {"tools": [GET_WEATHER_TOOL]}, True),
        ({}, {"tools": [GET_WEATHER_TOOL], "reasoning_effort": "none"}, False),
        (
            {"supports_bedrock_runtime_chat_completions_tools_with_reasoning": True},
            {"tools": [GET_WEATHER_TOOL], "reasoning_effort": "low"},
            False,
        ),
        ({}, {"response_format": RESPONSE_FORMAT_JSON_SCHEMA}, True),
        (
            {"supports_bedrock_runtime_chat_completions_response_format": True},
            {"response_format": RESPONSE_FORMAT_JSON_SCHEMA},
            False,
        ),
        (
            {"supports_bedrock_runtime_chat_completions_response_format": True},
            {"response_format": RESPONSE_FORMAT_JSON_SCHEMA, "tools": [GET_WEATHER_TOOL], "reasoning_effort": "low"},
            True,
        ),
    ],
)
def test_capability_flags_are_read_from_the_cost_map(monkeypatch, capability_flags, request_params, needs_converse):
    entry = {"litellm_provider": "bedrock_converse", **capability_flags}
    monkeypatch.setattr(litellm, "model_cost", {"vendor.native-model-v1:0": entry})
    assert bedrock_request_needs_converse(SYNTHETIC_NATIVE_MODEL, request_params) is needs_converse
    route = bedrock_route_for_request(SYNTHETIC_NATIVE_MODEL, request_params, None)
    assert (route == "chat_completions") is (not needs_converse)


def test_route_for_request_ignores_dropped_params(local_cost_map):
    params = {"response_format": RESPONSE_FORMAT_JSON_SCHEMA, "guardrailConfig": {"guardrailIdentifier": "gr-1"}}
    model = "chat_completions/openai.gpt-oss-20b-1:0"
    assert bedrock_route_for_request(model, params, None) == "converse"
    assert bedrock_route_for_request(model, params, ["guardrailConfig"]) == "converse"
    assert bedrock_route_for_request(model, params, ["guardrailConfig", "response_format"]) == "chat_completions"


def test_gpt_oss_response_format_goes_to_converse_with_json_tool_call(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=CONVERSE_JSON)
    litellm.completion(
        model="bedrock/chat_completions/openai.gpt-oss-20b-1:0",
        messages=[{"role": "user", "content": "Reply with the single word pong."}],
        response_format=RESPONSE_FORMAT_JSON_SCHEMA,
        max_tokens=64,
        client=client,
    )

    assert requests[0].url.raw_path.endswith(b"/model/openai.gpt-oss-20b-1%3A0/converse")
    body = json.loads(requests[0].content)
    assert body["toolConfig"]["tools"][0]["toolSpec"]["name"] == "json_tool_call"
    assert body["toolConfig"]["toolChoice"] == {"tool": {"name": "json_tool_call"}}
    assert body["inferenceConfig"]["maxTokens"] == 64
    assert "response_format" not in body
    assert "max_completion_tokens" not in body


def test_gpt56_response_format_is_sent_as_is_on_chat_completions(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=_chat_completion_json('{"word": "pong"}', "global.openai.gpt-5.6-sol"))
    response = litellm.completion(
        model="bedrock/chat_completions/global.openai.gpt-5.6-sol",
        messages=[{"role": "user", "content": "Reply with the single word pong."}],
        response_format=RESPONSE_FORMAT_JSON_SCHEMA,
        client=client,
    )

    assert str(requests[0].url) == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions"
    assert json.loads(requests[0].content)["response_format"] == RESPONSE_FORMAT_JSON_SCHEMA
    assert response.choices[0].message.content == '{"word": "pong"}'


def test_gpt56_schema_less_json_object_goes_to_converse_without_a_schema_tool(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=CONVERSE_JSON)
    litellm.completion(
        model="bedrock/chat_completions/global.openai.gpt-5.6-sol",
        messages=[{"role": "user", "content": "Reply with the single word pong."}],
        response_format={"type": "json_object"},
        max_tokens=64,
        client=client,
    )

    assert requests[0].url.raw_path.endswith(b"/model/global.openai.gpt-5.6-sol/converse")
    body = json.loads(requests[0].content)
    assert "toolConfig" not in body
    assert "response_format" not in body
    assert body["inferenceConfig"]["maxTokens"] == 64


def test_gpt56_json_object_with_response_schema_goes_to_converse_as_a_json_tool(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=CONVERSE_JSON)
    litellm.completion(
        model="bedrock/chat_completions/global.openai.gpt-5.6-sol",
        messages=[{"role": "user", "content": "Reply with the single word pong."}],
        response_format=JSON_OBJECT_WITH_RESPONSE_SCHEMA,
        max_tokens=64,
        client=client,
    )

    assert requests[0].url.raw_path.endswith(b"/model/global.openai.gpt-5.6-sol/converse")
    body = json.loads(requests[0].content)
    assert body["toolConfig"]["tools"][0]["toolSpec"]["name"] == "json_tool_call"
    assert body["toolConfig"]["toolChoice"] == {"tool": {"name": "json_tool_call"}}
    assert "response_format" not in body


LITERAL_TAGGED_ANSWER = "<reasoning>not thinking</reasoning> Hello"


@pytest.mark.parametrize("model", ["openai.gpt-5.6-sol", "us.xai.grok-4.6"])
def test_streaming_handler_keeps_a_literal_reasoning_tag_outside_gpt_oss(local_cost_map, model):
    handler = BedrockRuntimeChatCompletionsStreamingHandler(streaming_response=iter(()), sync_stream=True)

    opened = handler.chunk_parser({**_stream_chunk({"content": "<reasoning>not thinking"}), "model": model})
    assert opened.choices[0].delta.content == "<reasoning>not thinking"
    assert _reasoning_of(opened) is None

    closed = handler.chunk_parser(
        {**_stream_chunk({"content": "</reasoning> Hello"}, finish_reason="stop"), "model": model}
    )
    assert closed.choices[0].delta.content == "</reasoning> Hello"
    assert _reasoning_of(closed) is None


@pytest.mark.parametrize(
    "model, expected_content, expected_reasoning",
    [
        ("bedrock/global.openai.gpt-5.6-sol", LITERAL_TAGGED_ANSWER, None),
        ("bedrock/chat_completions/us.xai.grok-4.6", LITERAL_TAGGED_ANSWER, None),
        ("bedrock/chat_completions/openai.gpt-oss-20b-1:0", "Hello", "not thinking"),
        ("bedrock/chat_completions/openai.gpt-oss-safeguard-20b", "Hello", "not thinking"),
    ],
)
def test_reasoning_tag_split_applies_to_gpt_oss_answers_only(
    local_cost_map, fake_aws_env, model, expected_content, expected_reasoning
):
    model_id = model.removeprefix("bedrock/").removeprefix("chat_completions/")
    requests, client = _recording_client(json=_chat_completion_json(LITERAL_TAGGED_ANSWER, model_id))

    response = litellm.completion(
        model=model, messages=[{"role": "user", "content": "hello"}], client=client, max_tokens=64
    )

    assert str(requests[0].url).endswith("/openai/v1/chat/completions")
    assert response.choices[0].message.content == expected_content
    assert getattr(response.choices[0].message, "reasoning_content", None) == expected_reasoning


@pytest.mark.parametrize(
    "capability_flags, expected_content, expected_reasoning",
    [
        ({"supports_bedrock_runtime_chat_completions_inline_reasoning": True}, "Hello", "not thinking"),
        ({}, LITERAL_TAGGED_ANSWER, None),
    ],
)
def test_reasoning_tag_split_is_read_from_the_cost_map(
    monkeypatch, fake_aws_env, capability_flags, expected_content, expected_reasoning
):
    model_id = SYNTHETIC_NATIVE_MODEL.removeprefix("chat_completions/")
    monkeypatch.setattr(litellm, "model_cost", {model_id: {"litellm_provider": "bedrock_converse", **capability_flags}})
    requests, client = _recording_client(json=_chat_completion_json(LITERAL_TAGGED_ANSWER, model_id))

    response = litellm.completion(
        model=f"bedrock/{SYNTHETIC_NATIVE_MODEL}",
        messages=[{"role": "user", "content": "hello"}],
        client=client,
        max_tokens=64,
    )

    assert str(requests[0].url).endswith("/openai/v1/chat/completions")
    assert response.choices[0].message.content == expected_content
    assert getattr(response.choices[0].message, "reasoning_content", None) == expected_reasoning

    handler = BedrockRuntimeChatCompletionsStreamingHandler(streaming_response=iter(()), sync_stream=True)
    chunk = handler.chunk_parser(
        {**_stream_chunk({"content": LITERAL_TAGGED_ANSWER}, finish_reason="stop"), "model": model_id}
    )
    assert chunk.choices[0].delta.content == expected_content
    assert _reasoning_of(chunk) == expected_reasoning


def _tool_call(tool_call_id, name):
    return {"id": tool_call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


def test_positional_tool_call_ids_are_minted_unique_per_response(local_cost_map, fake_aws_env):
    reply = _chat_completion_json(
        None,
        "global.xai.grok-4.7",
        tool_calls=[_tool_call("call_0", "read_a"), _tool_call("call_1", "read_b")],
    )
    _, client = _recording_client(json=reply)
    responses = [
        litellm.completion(
            model="bedrock/chat_completions/global.xai.grok-4.7",
            messages=[{"role": "user", "content": "hello"}],
            client=client,
        )
        for _ in range(2)
    ]

    ids = [tool_call.id for response in responses for tool_call in response.choices[0].message.tool_calls]
    assert len(set(ids)) == 4
    assert all(re.fullmatch(r"call_[0-9a-f]{32}", tool_call_id) for tool_call_id in ids)
    assert [tool_call.function.name for tool_call in responses[0].choices[0].message.tool_calls] == ["read_a", "read_b"]


def test_provider_unique_tool_call_ids_pass_through(local_cost_map, fake_aws_env):
    reply = _chat_completion_json(
        None, "openai.gpt-oss-20b-1:0", tool_calls=[_tool_call("chatcmpl-tool-90090f0c1c521528", "read_a")]
    )
    _, client = _recording_client(json=reply)
    response = litellm.completion(
        model="bedrock/chat_completions/openai.gpt-oss-20b-1:0",
        messages=[{"role": "user", "content": "hello"}],
        client=client,
    )

    assert response.choices[0].message.tool_calls[0].id == "chatcmpl-tool-90090f0c1c521528"


def _streamed_tool_call_ids(tool_call_deltas, model="global.xai.grok-4.7"):
    handler = BedrockRuntimeChatCompletionsStreamingHandler(streaming_response=iter(()), sync_stream=True)
    return [
        tool_call.id
        for delta in tool_call_deltas
        for tool_call in handler.chunk_parser(_stream_chunk({"tool_calls": [delta]}, model=model))
        .choices[0]
        .delta.tool_calls
    ]


def test_streamed_positional_tool_call_ids_are_minted_once_per_tool_call(local_cost_map):
    deltas = [
        {"index": 0, **_tool_call("call_0", "read_a")},
        {"index": 0, "function": {"arguments": '{"x":1}'}},
        {"index": 0, "id": "call_0", "function": {"arguments": "}"}},
        {"index": 1, **_tool_call("call_1", "read_b")},
    ]
    first_stream = _streamed_tool_call_ids(deltas)
    second_stream = _streamed_tool_call_ids(deltas)

    assert first_stream[0] == first_stream[2]
    assert first_stream[1] is None
    assert len({first_stream[0], first_stream[3], second_stream[0], second_stream[3]}) == 4
    assert all(re.fullmatch(r"call_[0-9a-f]{32}", first_stream[i]) for i in (0, 3))


def test_streamed_provider_unique_tool_call_ids_pass_through(local_cost_map):
    assert _streamed_tool_call_ids(
        [{"index": 0, **_tool_call("chatcmpl-tool-8c9232df5019ff4f", "read_a")}], model="openai.gpt-oss-20b-1:0"
    ) == ["chatcmpl-tool-8c9232df5019ff4f"]


def test_streamed_positional_tool_call_ids_are_minted_for_gpt_too(local_cost_map):
    deltas = [{"index": 0, **_tool_call("call_0", "read_a")}, {"index": 1, **_tool_call("call_1", "read_b")}]

    ids = _streamed_tool_call_ids(deltas, model="global.openai.gpt-5.6-sol")

    assert len(set(ids)) == 2
    assert all(re.fullmatch(r"call_[0-9a-f]{32}", tool_call_id) for tool_call_id in ids)


def test_positional_tool_call_ids_are_minted_for_gpt_too(local_cost_map, fake_aws_env):
    reply = _chat_completion_json(
        None, "global.openai.gpt-5.6-sol", tool_calls=[_tool_call("call_0", "read_a"), _tool_call("call_1", "read_b")]
    )
    _, client = _recording_client(json=reply)
    response = litellm.completion(
        model="bedrock/global.openai.gpt-5.6-sol", messages=[{"role": "user", "content": "hello"}], client=client
    )

    ids = [tool_call.id for tool_call in response.choices[0].message.tool_calls]
    assert len(set(ids)) == 2
    assert all(re.fullmatch(r"call_[0-9a-f]{32}", tool_call_id) for tool_call_id in ids)
