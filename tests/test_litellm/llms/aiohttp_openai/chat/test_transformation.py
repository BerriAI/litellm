import asyncio
from typing import Any, Dict

from litellm.llms.aiohttp_openai.chat.transformation import AiohttpOpenAIChatConfig
from litellm.types.utils import ModelResponse


class _FakeClientResponse:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    async def json(self) -> Dict[str, Any]:
        return self._payload


class _NoopLogging:
    def post_call(self, *args: Any, **kwargs: Any) -> None:
        return None


def _run_transform(payload: Dict[str, Any]) -> ModelResponse:
    config = AiohttpOpenAIChatConfig()
    return asyncio.run(
        config.transform_response(
            model="glm-5.2",
            raw_response=_FakeClientResponse(payload),
            model_response=ModelResponse(),
            logging_obj=_NoopLogging(),
            request_data={},
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )
    )


def test_transform_response_forwards_prompt_tokens_details():
    """
    Regression test for https://github.com/BerriAI/litellm/issues/33967

    The aiohttp openai-compatible handler must forward prompt_tokens_details
    (e.g. cached_tokens on a cache hit) returned by the upstream provider.
    """
    payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "glm-5.2",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": 2000,
            "completion_tokens": 10,
            "total_tokens": 2010,
            "prompt_tokens_details": {"cached_tokens": 1088},
        },
    }

    response = _run_transform(payload)

    usage = response.usage
    assert usage.prompt_tokens == 2000
    assert usage.completion_tokens == 10
    assert usage.total_tokens == 2010
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.cached_tokens == 1088


def test_transform_response_without_usage():
    """A provider that omits usage entirely should not raise."""
    payload = {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "created": 1,
        "model": "glm-5.2",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
        ],
    }

    response = _run_transform(payload)

    assert response.id == "chatcmpl-2"
    assert response.choices[0].message.content == "hi"
