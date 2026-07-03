import httpx
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.mistral.conversations.transformation import (
    MistralConversationsConfig,
)
from litellm.types.utils import ModelResponse


@pytest.fixture(autouse=True)
def add_mistral_api_key_to_env(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-mistral-api-key-12345")


def _logging_obj():
    return Logging(
        model="mistral-medium-latest",
        messages=[{"role": "user", "content": "x"}],
        stream=False,
        call_type="completion",
        start_time=None,
        litellm_call_id="123",
        function_id="abc",
    )


def _raw(json_body: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json=json_body,
        request=httpx.Request("POST", "https://api.mistral.ai/v1/conversations"),
    )


def _transform(json_body: dict) -> ModelResponse:
    return MistralConversationsConfig().transform_response(
        model="mistral-medium-latest",
        raw_response=_raw(json_body),
        model_response=ModelResponse(),
        logging_obj=_logging_obj(),
        request_data={},
        messages=[],
        optional_params={},
        litellm_params={},
        encoding=None,
    )


CONVERSATIONS_RESPONSE = {
    "conversation_id": "conv_abc",
    "outputs": [
        {"type": "tool.execution", "name": "web_search"},
        {
            "type": "message.output",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Spain won "},
                {
                    "type": "tool_reference",
                    "tool": "web_search",
                    "title": "UEFA",
                    "url": "https://uefa.com/euro2024",
                },
                {"type": "text", "text": "Euro 2024."},
            ],
        },
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20, "connectors": {"web_search": 1}},
}


def test_transform_request_maps_to_conversations_shape():
    cfg = MistralConversationsConfig()
    body = cfg.transform_request(
        model="mistral-medium-latest",
        messages=[
            {"role": "system", "content": "Be terse"},
            {"role": "user", "content": "Who won the last Euro?"},
        ],
        optional_params={"web_search_options": {}, "temperature": 0.3, "max_tokens": 100},
        litellm_params={},
        headers={},
    )
    assert body["model"] == "mistral-medium-latest"
    assert body["inputs"] == [{"role": "user", "content": "Who won the last Euro?"}]
    assert body["instructions"] == "Be terse"
    assert body["tools"] == [{"type": "web_search"}]
    assert body["completion_args"] == {"temperature": 0.3, "max_tokens": 100}
    assert body["store"] is False
    assert "web_search_options" not in body
    assert "messages" not in body


def test_transform_request_premium_and_function_passthrough():
    cfg = MistralConversationsConfig()
    body = cfg.transform_request(
        model="mistral-medium-latest",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={
            "tools": [
                {"type": "web_search_premium"},
                {"type": "function", "function": {"name": "f"}},
            ]
        },
        litellm_params={},
        headers={},
    )
    assert body["tools"] == [
        {"type": "web_search_premium"},
        {"type": "function", "function": {"name": "f"}},
    ]


def test_transform_request_drops_other_builtin_connectors():
    cfg = MistralConversationsConfig()
    body = cfg.transform_request(
        model="mistral-medium-latest",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={
            "web_search_options": {},
            "tools": [
                {"type": "code_interpreter"},
                {"type": "function", "function": {"name": "f"}},
            ],
        },
        litellm_params={},
        headers={},
    )
    assert body["tools"] == [
        {"type": "web_search"},
        {"type": "function", "function": {"name": "f"}},
    ]
    assert {"type": "code_interpreter"} not in body["tools"]


def test_transform_request_maps_random_seed_from_extra_body():
    cfg = MistralConversationsConfig()
    body = cfg.transform_request(
        model="mistral-medium-latest",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"web_search_options": {}, "extra_body": {"random_seed": 7}},
        litellm_params={},
        headers={},
    )
    assert body["completion_args"] == {"random_seed": 7}


def test_get_complete_url():
    cfg = MistralConversationsConfig()
    assert (
        cfg.get_complete_url("https://api.mistral.ai/v1", None, "m", {}, {})
        == "https://api.mistral.ai/v1/conversations"
    )


def test_transform_response_chunked_content_and_citations():
    cfg = MistralConversationsConfig()
    mr = cfg.transform_response(
        model="mistral-medium-latest",
        raw_response=_raw(CONVERSATIONS_RESPONSE),
        model_response=ModelResponse(),
        logging_obj=_logging_obj(),
        request_data={},
        messages=[],
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    message = mr.choices[0].message
    assert message.content == "Spain won Euro 2024."
    assert mr.choices[0].finish_reason == "stop"

    annotations = message.annotations
    assert annotations[0]["type"] == "url_citation"
    assert annotations[0]["url_citation"]["url"] == "https://uefa.com/euro2024"
    assert annotations[0]["url_citation"]["title"] == "UEFA"

    assert mr.usage.prompt_tokens == 12
    assert mr.usage.total_tokens == 20
    assert mr.usage.prompt_tokens_details.web_search_requests == 1
    assert mr.id == "conv_abc"


def test_transform_response_counts_web_searches_from_connectors():
    mr = _transform(
        {
            "conversation_id": "c",
            "outputs": [{"type": "message.output", "role": "assistant", "content": "ok"}],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "connectors": {"web_search": 1, "web_search_premium": 2},
            },
        }
    )
    assert mr.usage.prompt_tokens_details.web_search_requests == 3
    assert mr.usage.web_search_premium_requests == 2


def test_transform_response_counts_fall_back_to_tool_execution_by_name():
    mr = _transform(
        {
            "conversation_id": "c",
            "outputs": [
                {"type": "tool.execution", "name": "web_search"},
                {"type": "tool.execution", "name": "web_search_premium"},
                {"type": "tool.execution", "name": "code_interpreter"},
                {"type": "message.output", "role": "assistant", "content": "ok"},
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    assert mr.usage.prompt_tokens_details.web_search_requests == 2
    assert mr.usage.web_search_premium_requests == 1


def test_transform_response_string_content_no_citations():
    cfg = MistralConversationsConfig()
    mr = cfg.transform_response(
        model="m",
        raw_response=_raw(
            {
                "conversation_id": "c2",
                "outputs": [{"type": "message.output", "role": "assistant", "content": "Plain answer."}],
            }
        ),
        model_response=ModelResponse(),
        logging_obj=_logging_obj(),
        request_data={},
        messages=[],
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert mr.choices[0].message.content == "Plain answer."
    assert getattr(mr.choices[0].message, "annotations", None) in (None, [])
    assert mr.usage.total_tokens == 0


def test_transform_response_without_message_output_is_graceful():
    cfg = MistralConversationsConfig()
    mr = cfg.transform_response(
        model="m",
        raw_response=_raw({"outputs": [{"type": "tool.execution", "name": "web_search"}]}),
        model_response=ModelResponse(),
        logging_obj=_logging_obj(),
        request_data={},
        messages=[],
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert mr.choices[0].message.content is None


@pytest.mark.parametrize(
    "web_search_kwargs",
    [
        {"tools": [{"type": "web_search"}]},
        {"web_search_options": {}},
    ],
)
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_completion_routes_web_search_to_conversations(sync_mode, web_search_kwargs, respx_mock):
    litellm.disable_aiohttp_transport = True

    conversations_route = respx_mock.post("https://api.mistral.ai/v1/conversations").respond(
        json=CONVERSATIONS_RESPONSE
    )
    chat_route = respx_mock.post("https://api.mistral.ai/v1/chat/completions").respond(json={"choices": []})

    kwargs = dict(
        model="mistral/mistral-medium-latest",
        messages=[{"role": "user", "content": "Who won the last Euro?"}],
        **web_search_kwargs,
    )
    if sync_mode:
        response = litellm.completion(**kwargs)
    else:
        response = await litellm.acompletion(**kwargs)

    assert conversations_route.called
    assert not chat_route.called

    sent = conversations_route.calls[0].request
    assert sent.url.path == "/v1/conversations"
    import json

    body = json.loads(sent.content)
    assert body["tools"] == [{"type": "web_search"}]
    assert body["inputs"] == [{"role": "user", "content": "Who won the last Euro?"}]
    assert "messages" not in body

    assert response.choices[0].message.content == "Spain won Euro 2024."
    assert response.choices[0].message.annotations[0]["url_citation"]["url"] == ("https://uefa.com/euro2024")


@pytest.mark.asyncio
async def test_completion_without_web_search_uses_chat_completions(respx_mock):
    litellm.disable_aiohttp_transport = True

    chat_route = respx_mock.post("https://api.mistral.ai/v1/chat/completions").respond(
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 1,
            "model": "mistral-medium-latest",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    conversations_route = respx_mock.post("https://api.mistral.ai/v1/conversations").respond(
        json=CONVERSATIONS_RESPONSE
    )

    response = litellm.completion(
        model="mistral/mistral-medium-latest",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert chat_route.called
    assert not conversations_route.called
    assert response.choices[0].message.content == "hi"


@pytest.mark.asyncio
async def test_completion_web_search_streaming_fakes_stream(respx_mock):
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://api.mistral.ai/v1/conversations").respond(json=CONVERSATIONS_RESPONSE)

    stream = litellm.completion(
        model="mistral/mistral-medium-latest",
        messages=[{"role": "user", "content": "Who won the last Euro?"}],
        tools=[{"type": "web_search"}],
        stream=True,
    )
    content = "".join(chunk.choices[0].delta.content or "" for chunk in stream)
    assert content == "Spain won Euro 2024."
