"""Native Bedrock Runtime Chat Completions: Grok, gpt-oss and GPT-5.6 stay on /openai/v1/chat/completions."""

import json

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
    get_bedrock_chat_config,
    uses_bedrock_runtime_chat_completions,
)
from litellm.llms.custom_httpx.http_handler import HTTPHandler


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
        "us.xai.grok-4.6",
        "global.xai.grok-4.6",
        "us-gov.xai.grok-4.6",
        "bedrock/us.xai.grok-4.6",
    ],
)
def test_grok_runtime_models_use_chat_completions_route(local_cost_map, model):
    assert uses_bedrock_runtime_chat_completions(model) is True
    assert BedrockModelInfo.get_bedrock_route(model) == "chat_completions"
    assert isinstance(get_bedrock_chat_config(model), AmazonBedrockRuntimeChatCompletionsConfig)


def test_explicit_converse_prefix_still_uses_converse(local_cost_map):
    assert BedrockModelInfo.get_bedrock_route("bedrock/converse/us.xai.grok-4.6") == "converse"
    assert BedrockModelInfo.get_bedrock_route("converse/us.xai.grok-4.6") == "converse"


def test_claude_stays_on_converse(local_cost_map):
    assert uses_bedrock_runtime_chat_completions("us.anthropic.claude-3-sonnet-20240229-v1:0") is False
    assert BedrockModelInfo.get_bedrock_route("us.anthropic.claude-3-sonnet-20240229-v1:0") == "converse"


@pytest.mark.parametrize(
    "entry",
    [
        {"litellm_provider": "bedrock_converse"},
        {"litellm_provider": "bedrock_converse", "supported_endpoints": ["/v1/responses"]},
        {"litellm_provider": "bedrock_converse", "supports_bedrock_runtime_chat_completions": True},
        {"litellm_provider": "bedrock_mantle", "supported_endpoints": ["/v1/chat/completions", "/v1/responses"]},
        {"litellm_provider": "openai", "supported_endpoints": ["/v1/chat/completions"]},
    ],
)
def test_chat_completions_missing_from_supported_endpoints_means_no_chat_completions_route(monkeypatch, entry):
    monkeypatch.setattr(litellm, "model_cost", {"us.xai.grok-4.6": entry})
    assert uses_bedrock_runtime_chat_completions("us.xai.grok-4.6") is False
    assert BedrockModelInfo.get_bedrock_route("us.xai.grok-4.6") == "converse"


def test_chat_completions_in_supported_endpoints_opts_into_the_native_route(monkeypatch):
    entry = {"litellm_provider": "bedrock_converse", "supported_endpoints": ["/v1/chat/completions", "/v1/responses"]}
    monkeypatch.setattr(litellm, "model_cost", {"us.xai.grok-4.6": entry})
    assert uses_bedrock_runtime_chat_completions("us.xai.grok-4.6") is True
    assert BedrockModelInfo.get_bedrock_route("bedrock/us.xai.grok-4.6") == "chat_completions"


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


def test_transform_request_is_openai_chat_body_not_converse():
    cfg = AmazonBedrockRuntimeChatCompletionsConfig()
    body = cfg.transform_request(
        model="bedrock/us.xai.grok-4.6",
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


def test_completion_posts_runtime_chat_completions(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=_chat_completion_json("ok", "us.xai.grok-4.6"))
    response = litellm.completion(
        model="us.xai.grok-4.6",
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


def test_region_path_sends_the_bare_model_id_to_the_path_region(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=_chat_completion_json("ok", "openai.gpt-oss-20b-1:0"))
    litellm.completion(
        model="bedrock/us-gov-west-1/openai.gpt-oss-20b-1:0",
        messages=[{"role": "user", "content": "hello"}],
        client=client,
    )

    assert str(requests[0].url) == "https://bedrock-runtime.us-gov-west-1.amazonaws.com/openai/v1/chat/completions"
    assert json.loads(requests[0].content)["model"] == "openai.gpt-oss-20b-1:0"
    assert "/us-gov-west-1/bedrock/aws4_request" in requests[0].headers["Authorization"]


def test_explicit_aws_region_name_wins_over_the_region_path(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=_chat_completion_json("ok", "openai.gpt-oss-20b-1:0"))
    litellm.completion(
        model="bedrock/us-gov-west-1/openai.gpt-oss-20b-1:0",
        messages=[{"role": "user", "content": "hello"}],
        aws_region_name="us-gov-east-1",
        client=client,
    )

    assert str(requests[0].url) == "https://bedrock-runtime.us-gov-east-1.amazonaws.com/openai/v1/chat/completions"
    assert json.loads(requests[0].content)["model"] == "openai.gpt-oss-20b-1:0"
    assert "/us-gov-east-1/bedrock/aws4_request" in requests[0].headers["Authorization"]


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
        *OPENAI_RUNTIME_MODELS,
        "bedrock/openai.gpt-oss-20b-1:0",
        "us-gov.openai.gpt-oss-20b-1:0",
        "bedrock/us-gov-west-1/openai.gpt-oss-20b-1:0",
        "us-gov-east-1/openai.gpt-oss-120b-1:0",
    ],
)
def test_openai_runtime_models_use_chat_completions_route(local_cost_map, model):
    assert uses_bedrock_runtime_chat_completions(model) is True
    assert BedrockModelInfo.get_bedrock_route(model) == "chat_completions"
    assert isinstance(get_bedrock_chat_config(model), AmazonBedrockRuntimeChatCompletionsConfig)


@pytest.mark.parametrize("model", ["us.amazon.nova-micro-v1:0", "us.anthropic.claude-haiku-4-5-20251001-v1:0"])
def test_nova_and_claude_stay_on_converse(local_cost_map, model):
    assert uses_bedrock_runtime_chat_completions(model) is False
    assert BedrockModelInfo.get_bedrock_route(model, {"tools": [GET_WEATHER_TOOL]}) == "converse"


@pytest.mark.parametrize("model", ["openai.gpt-oss-20b-1:0", "global.openai.gpt-5.6-sol"])
def test_guardrail_config_falls_back_to_converse(local_cost_map, model):
    guardrail = {"guardrailIdentifier": "gr-1", "guardrailVersion": "1"}
    assert bedrock_request_needs_converse(model, {"guardrailConfig": guardrail}) is True
    assert BedrockModelInfo.get_bedrock_route(model, {"guardrailConfig": guardrail}) == "converse"
    assert BedrockModelInfo.get_bedrock_route(model, {"guardrailConfig": None}) == "chat_completions"


@pytest.mark.parametrize("model", ["openai.gpt-oss-20b-1:0", "us.xai.grok-4.6"])
def test_additional_model_request_fields_fall_back_to_converse(local_cost_map, model):
    request_params = {"additionalModelRequestFields": {"reasoning_effort": "high"}}
    assert bedrock_request_needs_converse(model, request_params) is True
    assert BedrockModelInfo.get_bedrock_route(model, request_params) == "converse"
    assert BedrockModelInfo.get_bedrock_route(model, {key: None for key in request_params}) == "chat_completions"


@pytest.mark.parametrize("model", ["openai.gpt-oss-20b-1:0", "us.xai.grok-4.6", "global.openai.gpt-5.6-sol"])
def test_top_k_stays_on_chat_completions(local_cost_map, fake_aws_env, model):
    request_params = {"top_k": 40}
    assert bedrock_request_needs_converse(model, request_params) is False
    assert BedrockModelInfo.get_bedrock_route(model, request_params) == "chat_completions"

    requests, client = _recording_client(json=_chat_completion_json("ok", model))
    litellm.completion(
        model=f"bedrock/{model}",
        messages=[{"role": "user", "content": "hello"}],
        top_k=40,
        client=client,
    )

    assert str(requests[0].url) == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions"
    body = json.loads(requests[0].content)
    assert body["top_k"] == 40
    assert "inferenceConfig" not in body


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
    assert BedrockModelInfo.get_bedrock_route("global.openai.gpt-5.6-sol", request_params) == expected_route
    assert BedrockModelInfo.get_bedrock_route("bedrock/us.openai.gpt-5.6-terra", request_params) == expected_route


@pytest.mark.parametrize("reasoning_effort", ["low", "high", None])
def test_gpt_oss_tools_with_any_reasoning_effort_stay_on_chat_completions(local_cost_map, reasoning_effort):
    params = {"tools": [GET_WEATHER_TOOL], "reasoning_effort": reasoning_effort}
    assert bedrock_request_needs_converse("openai.gpt-oss-120b-1:0", params) is False
    assert BedrockModelInfo.get_bedrock_route("openai.gpt-oss-120b-1:0", params) == "chat_completions"


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
    assert BedrockModelInfo.get_bedrock_route("global.openai.gpt-5.6-sol", request_params) == expected_route
    assert BedrockModelInfo.get_bedrock_route("openai.gpt-oss-120b-1:0", request_params) == "chat_completions"


def test_thinking_block_goes_to_converse(local_cost_map):
    thinking = {"type": "enabled", "budget_tokens": 1024}
    assert BedrockModelInfo.get_bedrock_route("us.xai.grok-4.6", {"thinking": thinking}) == "converse"
    assert BedrockModelInfo.get_bedrock_route("us.xai.grok-4.6", {"thinking": None}) == "chat_completions"


def test_explicit_converse_prefix_wins_for_openai_models(local_cost_map):
    assert BedrockModelInfo.get_bedrock_route("bedrock/converse/openai.gpt-oss-20b-1:0") == "converse"
    assert BedrockModelInfo.get_bedrock_route("converse/global.openai.gpt-5.6-sol", {}) == "converse"


def test_map_openai_params_sends_max_tokens_as_max_completion_tokens():
    cfg = AmazonBedrockRuntimeChatCompletionsConfig()
    mapped = cfg.map_openai_params(
        non_default_params={"max_tokens": 64, "temperature": 0.1},
        optional_params={},
        model="global.openai.gpt-5.6-sol",
        drop_params=False,
    )
    assert mapped == {"max_completion_tokens": 64, "temperature": 0.1}


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
            ("frequency_penalty", "presence_penalty", "stop", "logprobs", "top_logprobs", "n"),
            ("temperature", "top_p", "logit_bias", "reasoning_effort", "tools", "functions"),
        ),
        (
            "us.xai.grok-4.6",
            ("frequency_penalty", "presence_penalty", "n"),
            ("stop", "logprobs", "top_p", "logit_bias", "reasoning_effort"),
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
        ("bedrock/global.openai.gpt-5.6-sol", {"frequency_penalty": 0.5}),
        ("bedrock/global.openai.gpt-5.6-sol", {"logprobs": True, "top_logprobs": 2}),
        ("bedrock/us.xai.grok-4.6", {"presence_penalty": 0.5}),
        ("bedrock/openai.gpt-oss-20b-1:0", {"logit_bias": {"1": 1}}),
    ],
    ids=lambda value: value if isinstance(value, str) else next(iter(value)),
)
def test_refused_params_are_dropped_or_refused_before_reaching_aws(local_cost_map, fake_aws_env, model, param):
    requests, client = _recording_client(json=_chat_completion_json("ok", model.removeprefix("bedrock/")))
    with pytest.raises(litellm.UnsupportedParamsError, match=next(iter(param))):
        litellm.completion(model=model, messages=[{"role": "user", "content": "hello"}], client=client, **param)
    litellm.completion(
        model=model, messages=[{"role": "user", "content": "hello"}], drop_params=True, client=client, **param
    )

    assert str(requests[0].url).endswith("/openai/v1/chat/completions")
    assert param.keys().isdisjoint(json.loads(requests[0].content))


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


def _stream_chunk(delta, finish_reason=None, index=0):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1733529600,
        "model": "openai.gpt-oss-20b-1:0",
        "choices": [{"index": index, "delta": delta, "finish_reason": finish_reason}],
    }


def test_streaming_handler_splits_reasoning_deltas_per_choice():
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


def test_streaming_handler_keeps_split_state_per_choice_index():
    handler = BedrockRuntimeChatCompletionsStreamingHandler(streaming_response=iter(()), sync_stream=True)

    opened = handler.chunk_parser(_stream_chunk({"content": "<reasoning>first"}, index=0))
    assert _reasoning_of(opened) == "first"

    plain = handler.chunk_parser(_stream_chunk({"content": "plain answer"}, index=1))
    assert _reasoning_of(plain) is None
    assert plain.choices[0].delta.content == "plain answer"

    still_reasoning = handler.chunk_parser(_stream_chunk({"content": " more"}, index=0))
    assert _reasoning_of(still_reasoning) == " more"
    assert not still_reasoning.choices[0].delta.content


def test_streaming_handler_flushes_held_text_on_an_empty_final_delta():
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
        model="bedrock/openai.gpt-oss-20b-1:0",
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
        model="bedrock/global.openai.gpt-5.6-sol",
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
        model="bedrock/global.openai.gpt-5.6-sol",
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
        model="bedrock/openai.gpt-oss-20b-1:0",
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
        model="bedrock/openai.gpt-oss-20b-1:0",
        messages=[{"role": "user", "content": "hello"}],
        metadata={"user_api_key_team_alias": "search"},
        client=client,
    )

    assert requests[0].url.raw_path.endswith(b"/model/openai.gpt-oss-20b-1%3A0/converse")
    assert json.loads(requests[0].content)["requestMetadata"] == {"user_api_key_team_alias": "search"}


def test_dropped_converse_only_key_keeps_the_request_on_chat_completions(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=_chat_completion_json("ok", "openai.gpt-oss-20b-1:0"))
    litellm.completion(
        model="bedrock/openai.gpt-oss-20b-1:0",
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
        model="bedrock/global.openai.gpt-5.6-sol",
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
        model="bedrock/openai.gpt-oss-20b-1:0",
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
            model="bedrock/global.openai.gpt-5.6-sol",
            messages=[{"role": "user", "content": "hello"}],
            functions=[GET_WEATHER_TOOL["function"]],
            reasoning_effort="low",
            client=client,
        )
    litellm.completion(
        model="bedrock/global.openai.gpt-5.6-sol",
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
        model="bedrock/us.xai.grok-4.6",
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
            model="bedrock/openai.gpt-oss-20b-1:0",
            messages=[{"role": "user", "content": "hello"}],
            guardrailConfig=guardrail,
            seed=7,
            client=client,
        )
    litellm.completion(
        model="bedrock/openai.gpt-oss-20b-1:0",
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
            model="bedrock/openai.gpt-oss-20b-1:0",
            messages=[{"role": "user", "content": "hello"}],
            n=2,
            client=client,
        )
    litellm.completion(
        model="bedrock/openai.gpt-oss-20b-1:0",
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
        model="bedrock/openai.gpt-oss-20b-1:0",
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


def test_streaming_handler_keeps_native_reasoning_next_to_the_tagged_split():
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


@pytest.mark.parametrize("model", ["openai.gpt-oss-20b-1:0", "bedrock/openai.gpt-oss-120b-1:0"])
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


RESPONSE_FORMAT_ENFORCING_MODELS = ["global.openai.gpt-5.6-sol", "us.xai.grok-4.6", "bedrock/us-gov.xai.grok-4.6"]


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


SYNTHETIC_NATIVE_MODEL = "vendor.native-model-v1:0"


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
    entry = {
        "litellm_provider": "bedrock_converse",
        "supported_endpoints": ["/v1/chat/completions"],
        **capability_flags,
    }
    monkeypatch.setattr(litellm, "model_cost", {SYNTHETIC_NATIVE_MODEL: entry})
    assert bedrock_request_needs_converse(SYNTHETIC_NATIVE_MODEL, request_params) is needs_converse
    route = bedrock_route_for_request(SYNTHETIC_NATIVE_MODEL, request_params, None)
    assert (route == "chat_completions") is (not needs_converse)


def test_route_for_request_ignores_dropped_params(local_cost_map):
    params = {"response_format": RESPONSE_FORMAT_JSON_SCHEMA, "guardrailConfig": {"guardrailIdentifier": "gr-1"}}
    assert bedrock_route_for_request("openai.gpt-oss-20b-1:0", params, None) == "converse"
    assert bedrock_route_for_request("openai.gpt-oss-20b-1:0", params, ["guardrailConfig"]) == "converse"
    assert (
        bedrock_route_for_request("openai.gpt-oss-20b-1:0", params, ["guardrailConfig", "response_format"])
        == "chat_completions"
    )


def test_gpt_oss_response_format_goes_to_converse_with_json_tool_call(local_cost_map, fake_aws_env):
    requests, client = _recording_client(json=CONVERSE_JSON)
    litellm.completion(
        model="bedrock/openai.gpt-oss-20b-1:0",
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
        model="bedrock/global.openai.gpt-5.6-sol",
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
        model="bedrock/global.openai.gpt-5.6-sol",
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
        model="bedrock/global.openai.gpt-5.6-sol",
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
