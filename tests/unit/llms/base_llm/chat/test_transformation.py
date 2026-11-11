import json

import httpx
import pytest
import respx

import litellm


def test_base_http_handler_sends_a_caller_extra_body_over_the_request_unchanged(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", "True")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route = respx_mock.post(url__regex=r"https://api\.deepseek\.com/.*chat/completions").mock(
        return_value=httpx.Response(
            200, json={"id": "c", "object": "chat.completion", "created": 0, "model": "m", "choices": []}
        )
    )

    litellm.completion(
        model="deepseek/deepseek-chat",
        messages=[{"role": "user", "content": "hi"}],
        api_key="sk-test",
        temperature=0.5,
        extra_body={"foo": 1, "temperature": 0.9, "metadata": {"b": "2"}},
    )

    body = json.loads(route.calls.last.request.content)
    assert body["foo"] == 1
    assert body["temperature"] == 0.9
    assert body["metadata"] == {"b": "2"}
