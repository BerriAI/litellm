import asyncio
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


def _batch_migration_router() -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {
                    "model": f"openai/{name}",
                    "api_key": "test-key",
                    "api_base": "https://batch-migration.local/v1",
                },
            }
            for name in ("first", "second")
        ],
        num_retries=0,
    )


def _batch_migration_response(request: httpx.Request) -> httpx.Response:
    request_body: Final = json.loads(request.content)
    model: Final = request_body["model"]
    prompt: Final = request_body["messages"][0]["content"]
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-batch-migration",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"{model}:{prompt}"},
                    "finish_reason": "stop",
                }
            ],
        },
    )


@pytest.mark.asyncio
async def test_batch_completion_multiple_models_returns_each_model_response(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://batch-migration.local/v1/chat/completions").mock(
        side_effect=_batch_migration_response
    )
    router: Final = _batch_migration_router()

    responses: Final = cast(
        list[ModelResponse],
        await router.abatch_completion(
            models=["first", "second"],
            messages=[{"role": "user", "content": "same prompt"}],
        ),
    )

    assert route.call_count == 2
    assert frozenset(response.choices[0].message.content for response in responses) == {
        "first:same prompt",
        "second:same prompt",
    }


@pytest.mark.asyncio
async def test_batch_completion_multiple_models_and_messages_preserves_each_pair(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://batch-migration.local/v1/chat/completions").mock(
        side_effect=_batch_migration_response
    )
    router: Final = _batch_migration_router()

    responses: Final = cast(
        list[list[ModelResponse]],
        await router.abatch_completion(
            models=["first", "second"],
            messages=[
                [{"role": "user", "content": "first prompt"}],
                [{"role": "user", "content": "second prompt"}],
            ],
        ),
    )

    assert route.call_count == 4
    assert frozenset(
        response.choices[0].message.content
        for model_responses in responses
        for response in model_responses
    ) == {
        "first:first prompt",
        "first:second prompt",
        "second:first prompt",
        "second:second prompt",
    }


@pytest.mark.asyncio
async def test_batch_completion_fastest_response_prefers_the_mocked_deployment(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    release_provider: Final = asyncio.Event()

    async def response(request: httpx.Request) -> httpx.Response:
        await release_provider.wait()
        return _batch_migration_response(request)

    respx_mock.post("https://batch-migration.local/v1/chat/completions").mock(side_effect=response)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "first",
                "litellm_params": {
                    "model": "openai/first",
                    "api_key": "test-key",
                    "api_base": "https://batch-migration.local/v1",
                },
                "model_info": {"id": "first-deployment"},
            },
            {
                "model_name": "second",
                "litellm_params": {
                    "model": "openai/second",
                    "api_key": "test-key",
                    "api_base": "https://batch-migration.local/v1",
                    "mock_response": "mocked deployment response",
                },
                "model_info": {"id": "second-deployment"},
            },
        ],
        num_retries=0,
    )
    try:
        result: Final = await router.abatch_completion_fastest_response(
            model="first,second",
            messages=[{"role": "user", "content": "fastest prompt"}],
        )
    finally:
        release_provider.set()

    assert result._hidden_params["model_id"] == "second-deployment"
    assert result.choices[0].message.content == "mocked deployment response"


@pytest.mark.asyncio
async def test_batch_completion_fastest_response_streams_mocked_chunks() -> None:
    router: Final = _batch_migration_router()

    response: Final = await router.abatch_completion_fastest_response(
        model="first,second",
        messages=[{"role": "user", "content": "stream prompt"}],
        stream=True,
        mock_response="stream answer",
    )
    chunks: Final = tuple([chunk async for chunk in response])
    text: Final = "".join(
        chunk.choices[0].delta.content or ""
        for chunk in chunks
        if chunk.choices and chunk.choices[0].delta is not None
    )

    assert text == "stream answer"
    assert chunks[-1].choices[0].finish_reason == "stop"
