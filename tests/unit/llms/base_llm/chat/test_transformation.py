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


def test_get_model_response_iterator_for_request_forwards_json_mode_for_async_stream() -> None:
    from litellm.llms.base_llm.chat.transformation import BaseConfig

    class DummyChatTransformation(BaseConfig):
        def get_supported_openai_params(self, *args, **kwargs):
            return []

        def map_openai_params(self, *args, **kwargs):
            return {}

        def transform_request(self, *args, **kwargs):
            return {}

        def validate_environment(self, *args, **kwargs):
            return {}

        def transform_response(self, *args, **kwargs):
            pass

        def get_error_class(self, *args, **kwargs):
            pass

        def get_model_response_iterator(self, streaming_response, sync_stream, json_mode=False):
            return {"sync_stream": sync_stream, "json_mode": json_mode}

    transformation = DummyChatTransformation()

    # Verify sync_stream=True preserves json_mode
    sync_result = transformation.get_model_response_iterator_for_request(
        streaming_response=None,
        sync_stream=True,
        json_mode=True,
        optional_params={},
    )
    assert sync_result == {"sync_stream": True, "json_mode": True}

    # Verify sync_stream=False (async stream) now forwards json_mode instead of dropping it
    async_result = transformation.get_model_response_iterator_for_request(
        streaming_response=None,
        sync_stream=False,
        json_mode=True,
        optional_params={},
    )
    assert async_result == {"sync_stream": False, "json_mode": True}
