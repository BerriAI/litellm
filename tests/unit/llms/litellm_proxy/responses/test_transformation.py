import json
from collections.abc import Callable, Iterable
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.llms.litellm_proxy.responses.transformation import LiteLLMProxyResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager
from tests.unit.proxy.conftest import httpx_transport

_PROXY_RESPONSES_URL: Final = "https://my-proxy.example.com/responses"


@pytest.mark.usefixtures(httpx_transport.__name__)
def test_litellm_proxy_responses_request_uses_the_openai_wire_format() -> None:
    with respx.mock(assert_all_called=True) as mock:
        route: Final = mock.post(_PROXY_RESPONSES_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "resp_proxy_1",
                    "object": "response",
                    "created_at": 1750000000,
                    "status": "completed",
                    "model": "gpt-5.5",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_proxy_1",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "pong", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
                },
            )
        )
        response: Final = litellm.responses(
            model="litellm_proxy/gpt-5.5",
            input="ping",
            max_output_tokens=16,
            api_base="https://my-proxy.example.com/",
            api_key="sk-proxy-key",
        )
        request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer sk-proxy-key"
    assert json.loads(request.content) == {"model": "gpt-5.5", "input": "ping", "max_output_tokens": 16}
    assert (response.usage.input_tokens, response.usage.output_tokens) == (3, 1)
    assert response.output[0].content[0].text == "pong"


def test_provider_config_manager_returns_litellm_proxy_responses_config() -> None:
    config: Final = ProviderConfigManager.get_provider_responses_api_config(
        model="litellm_proxy/gpt-5.5", provider=LlmProviders.LITELLM_PROXY
    )
    assert isinstance(config, LiteLLMProxyResponsesAPIConfig)
    assert config.custom_llm_provider == LlmProviders.LITELLM_PROXY


@pytest.mark.parametrize("api_base", ["https://my-proxy.example.com", "https://my-proxy.example.com/"])
def test_get_complete_url_appends_responses_path(api_base: str) -> None:
    assert (
        LiteLLMProxyResponsesAPIConfig().get_complete_url(api_base=api_base, litellm_params={})
        == "https://my-proxy.example.com/responses"
    )


def test_get_complete_url_requires_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_PROXY_API_BASE", raising=False)
    with pytest.raises(ValueError, match="api_base not set"):
        LiteLLMProxyResponsesAPIConfig().get_complete_url(api_base=None, litellm_params={})


@pytest.mark.parametrize("drop_params", [False, True])
def test_compact_request_forwards_metadata_and_user_to_the_downstream_proxy(drop_params: bool) -> None:
    url, data = LiteLLMProxyResponsesAPIConfig().transform_compact_response_api_request(
        model="gpt-5.5",
        input="hi",
        response_api_optional_request_params={"metadata": {"ticket": "qa-1"}, "user": "end-user-1", "temperature": 0.2},
        api_base="https://my-proxy.example.com/responses",
        litellm_params=GenericLiteLLMParams(drop_params=drop_params),
        headers={},
    )

    assert url == "https://my-proxy.example.com/responses/compact"
    assert {key: data[key] for key in data if key != "input"} == {
        "model": "gpt-5.5",
        "metadata": {"ticket": "qa-1"},
        "user": "end-user-1",
        "temperature": 0.2,
    }


_GUARDRAIL_BLOCK: Final = {
    "error": {
        "message": "Content blocked: secret_project_codename pattern detected",
        "type": "invalid_request_error",
        "param": None,
        "code": "400",
        "provider_specific_fields": {"guardrail_name": "block-secret-project", "guardrail_mode": "pre_call"},
    }
}


def _proxy_rejection(route: str, response: httpx.Response, call: Callable[[], object]) -> litellm.BadRequestError:
    with respx.mock(assert_all_called=True) as mock:
        _ = mock.post(f"https://my-proxy.example.com/{route}").mock(return_value=response)
        with pytest.raises(litellm.BadRequestError) as caught:
            _ = call()
    return caught.value


def _consume(stream: Iterable[object]) -> None:
    for _ in stream:
        pass


_PROXY_PARAMS: Final = {"api_base": "https://my-proxy.example.com/", "api_key": "sk-proxy-key"}


@pytest.mark.usefixtures(httpx_transport.__name__)
@pytest.mark.parametrize("stream", [False, True])
def test_proxy_400_reaches_responses_caller_with_the_same_body_as_chat_completions(stream: bool) -> None:
    chat_error: Final = _proxy_rejection(
        "chat/completions",
        httpx.Response(400, json=_GUARDRAIL_BLOCK),
        lambda: litellm.completion(
            model="litellm_proxy/gpt-5.5", messages=[{"role": "user", "content": "project bluebird"}], **_PROXY_PARAMS
        ),
    )
    responses_error: Final = _proxy_rejection(
        "responses",
        httpx.Response(400, json=_GUARDRAIL_BLOCK),
        lambda: (
            _consume(litellm.responses(model="litellm_proxy/gpt-5.5", input="project bluebird", stream=True, **_PROXY_PARAMS))
            if stream
            else litellm.responses(model="litellm_proxy/gpt-5.5", input="project bluebird", **_PROXY_PARAMS)
        ),
    )
    assert responses_error.body == chat_error.body == _GUARDRAIL_BLOCK["error"]
    assert (responses_error.code, responses_error.type) == (chat_error.code, chat_error.type)
    assert str(responses_error) == str(chat_error)


@pytest.mark.usefixtures(httpx_transport.__name__)
def test_proxy_error_without_a_json_object_body_still_raises_with_its_text() -> None:
    error: Final = _proxy_rejection(
        "responses",
        httpx.Response(400, text="upstream proxy rejected the request"),
        lambda: litellm.responses(model="litellm_proxy/gpt-5.5", input="hi", **_PROXY_PARAMS),
    )
    assert error.body is None
    assert "upstream proxy rejected the request" in str(error)


@pytest.mark.usefixtures(httpx_transport.__name__)
def test_proxy_error_object_without_an_error_member_is_the_body_as_sent() -> None:
    sent: Final = {"message": "upstream proxy rejected the request", "type": "invalid_request_error"}
    error: Final = _proxy_rejection(
        "responses",
        httpx.Response(400, json=sent),
        lambda: litellm.responses(model="litellm_proxy/gpt-5.5", input="hi", **_PROXY_PARAMS),
    )
    assert error.body == sent


@pytest.mark.usefixtures(httpx_transport.__name__)
def test_proxy_error_whose_error_member_is_text_still_raises_with_that_text() -> None:
    error: Final = _proxy_rejection(
        "responses",
        httpx.Response(400, json={"error": "upstream proxy rejected the request"}),
        lambda: litellm.responses(model="litellm_proxy/gpt-5.5", input="hi", **_PROXY_PARAMS),
    )
    assert "upstream proxy rejected the request" in str(error)
