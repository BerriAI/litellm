import json
from typing import Final, cast

import httpx
import pytest
import respx

import litellm
from litellm.types.utils import ModelResponse

msg1 = [{"role": "user", "content": "hi 1"}]
msg2 = [{"role": "user", "content": "hi 2"}]


def test_batch_completion_return_exceptions_true(respx_mock: respx.MockRouter):
    """Test batch_completion's return_exceptions.

    With an invalid API key, we expect an error to be returned rather than raised.
    The error type may be AuthenticationError (from API) or InternalServerError
    (from connection issues), depending on network conditions.
    """
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            401,
            json={
                "error": {
                    "message": "Incorrect API key provided: sk_xxx.",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )
    )
    res = litellm.batch_completion(
        model="gpt-3.5-turbo",
        messages=[msg1, msg2],
        api_key="sk_xxx",
    )

    assert isinstance(
        res[0],
        (
            litellm.exceptions.AuthenticationError,
            litellm.exceptions.InternalServerError,
        ),
    ), f"Expected AuthenticationError or InternalServerError, got {type(res[0])}"


def test_batch_completion_returns_one_response_per_message(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-batch",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "batch answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )
    responses: Final[list[ModelResponse]] = cast(
        list[ModelResponse],
        litellm.batch_completion(
            model="openai/gpt-4o-mini",
            messages=[
                [{"role": "user", "content": "first prompt"}],
                [{"role": "user", "content": "second prompt"}],
            ],
            api_key="test-key",
            max_workers=1,
        ),
    )

    request_bodies: Final = tuple(json.loads(call.request.content) for call in route.calls)
    assert route.call_count == 2
    assert tuple(body["messages"][0]["content"] for body in request_bodies) == (
        "first prompt",
        "second prompt",
    )
    assert tuple(response.choices[0].message.content for response in responses) == (
        "batch answer",
        "batch answer",
    )
