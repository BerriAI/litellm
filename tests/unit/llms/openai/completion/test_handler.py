import asyncio
import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.types.utils import TextCompletionResponse
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


def test_convert_dict_to_text_completion_response():
    input_dict = {
        "id": "cmpl-ALVLPJgRkqpTomotoOMi3j0cAaL4L",
        "choices": [
            {
                "finish_reason": "length",
                "index": 0,
                "logprobs": {
                    "text_offset": [0, 5],
                    "token_logprobs": [None, -12.203847],
                    "tokens": ["hello", " crisp"],
                    "top_logprobs": [None, {",": -2.1568563}],
                },
                "text": "hello crisp",
            }
        ],
        "created": 1729688739,
        "model": "davinci-002",
        "object": "text_completion",
        "system_fingerprint": None,
        "usage": {
            "completion_tokens": 1,
            "prompt_tokens": 1,
            "total_tokens": 2,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        },
    }

    response = TextCompletionResponse(**input_dict)

    assert response.id == "cmpl-ALVLPJgRkqpTomotoOMi3j0cAaL4L"
    assert len(response.choices) == 1
    assert response.choices[0].finish_reason == "length"
    assert response.choices[0].index == 0
    assert response.choices[0].text == "hello crisp"
    assert response.created == 1729688739
    assert response.model == "davinci-002"
    assert response.object == "text_completion"
    assert response.system_fingerprint is None
    assert response.usage.completion_tokens == 1
    assert response.usage.prompt_tokens == 1
    assert response.usage.total_tokens == 2
    assert response.usage.completion_tokens_details is None
    assert response.usage.prompt_tokens_details is None

    # Test logprobs
    assert response.choices[0].logprobs.text_offset == [0, 5]
    assert response.choices[0].logprobs.token_logprobs == [None, -12.203847]
    assert response.choices[0].logprobs.tokens == ["hello", " crisp"]
    assert response.choices[0].logprobs.top_logprobs == [None, {",": -2.1568563}]


@pytest.mark.asyncio
async def test_acompletion_uses_optimized_http_client():
    """
    Test that OpenAITextCompletion.acompletion uses BaseOpenAILLM.get_async_http_client()
    instead of litellm.aclient_session directly.

    Related issue: https://github.com/BerriAI/litellm/issues/17676
    """
    from litellm.llms.openai.common_utils import BaseOpenAILLM
    from litellm.llms.openai.completion.handler import OpenAITextCompletion

    mock_http_client = MagicMock()
    mock_async_openai = AsyncMock()
    mock_async_openai.completions.with_raw_response.create = AsyncMock(
        return_value=MagicMock(
            parse=MagicMock(
                return_value=MagicMock(
                    model_dump=MagicMock(
                        return_value={
                            "id": "test-id",
                            "object": "text_completion",
                            "created": 1234567890,
                            "model": "gpt-3.5-turbo-instruct",
                            "choices": [
                                {
                                    "text": "test response",
                                    "index": 0,
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 5,
                                "completion_tokens": 10,
                                "total_tokens": 15,
                            },
                        }
                    )
                )
            )
        )
    )

    with patch.object(BaseOpenAILLM, "_get_async_http_client", return_value=mock_http_client) as mock_get_client:
        with patch(
            "litellm.llms.openai.workload_identity.AsyncOpenAI",
            return_value=mock_async_openai,
        ) as mock_openai_class:
            handler = OpenAITextCompletion()
            logging_obj = MagicMock()
            logging_obj.post_call = MagicMock()

            await handler.acompletion(
                logging_obj=logging_obj,
                api_base="https://api.openai.com/v1",
                data={"prompt": "test", "model": "gpt-3.5-turbo-instruct"},
                headers={},
                model_response=MagicMock(),
                api_key="test-key",
                model="gpt-3.5-turbo-instruct",
                timeout=30.0,
                max_retries=2,
            )

            # Verify _get_async_http_client was called
            mock_get_client.assert_called_once()

            # Verify AsyncOpenAI was initialized with the http_client from _get_async_http_client
            mock_openai_class.assert_called_once()
            call_kwargs = mock_openai_class.call_args.kwargs
            assert call_kwargs["http_client"] == mock_http_client


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown(event_loop):
    import litellm

    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    importlib.reload(litellm)
    asyncio.set_event_loop(event_loop)
    yield
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    pending = asyncio.all_tasks(event_loop)
    for task in pending:
        task.cancel()
    if pending:
        event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


_SCALAR_DEFAULTS = {
    "num_retries": getattr(litellm, "num_retries", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
    "cohere_key": getattr(litellm, "cohere_key", None),
}
