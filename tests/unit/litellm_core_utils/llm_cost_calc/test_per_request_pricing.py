import asyncio

import pytest

import litellm
from litellm import ModelResponse, Router
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import Usage

REQUEST_PRICE = 0.05
INPUT_COST_PER_TOKEN = 1e-6
OUTPUT_COST_PER_TOKEN = 2e-6


def _mock_response(tokens: int) -> ModelResponse:
    return ModelResponse(
        model="flat-priced-model",
        choices=[{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop", "index": 0}],
        usage=Usage(prompt_tokens=tokens, completion_tokens=tokens, total_tokens=2 * tokens),
    )


def _router(**pricing: float) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "flat-model",
                "litellm_params": {"model": "openai/flat-priced-model", "api_key": "fake-key", **pricing},
            }
        ]
    )


@pytest.mark.parametrize("tokens", [5, 50_000])
def test_router_completion_charges_flat_price_per_request_regardless_of_tokens(tokens: int) -> None:
    response = _router(input_cost_per_query=REQUEST_PRICE).completion(
        model="flat-model", messages=[{"role": "user", "content": "hi"}], mock_response=_mock_response(tokens)
    )

    assert response.usage.prompt_tokens == tokens
    assert response._hidden_params["response_cost"] == pytest.approx(REQUEST_PRICE)


def test_request_price_replaces_token_prices_set_on_the_same_deployment() -> None:
    router = _router(
        input_cost_per_query=REQUEST_PRICE,
        input_cost_per_token=INPUT_COST_PER_TOKEN,
        output_cost_per_token=OUTPUT_COST_PER_TOKEN,
    )

    response = router.completion(
        model="flat-model", messages=[{"role": "user", "content": "hi"}], mock_response=_mock_response(10_000)
    )

    assert response._hidden_params["response_cost"] == pytest.approx(REQUEST_PRICE)


def test_sdk_completion_accepts_per_call_request_price() -> None:
    response = litellm.completion(
        model="openai/flat-priced-model",
        api_key="fake-key",
        messages=[{"role": "user", "content": "hi"}],
        mock_response=_mock_response(10_000),
        input_cost_per_query=REQUEST_PRICE,
    )

    assert response._hidden_params["response_cost"] == pytest.approx(REQUEST_PRICE)


def test_embedding_charges_flat_price_per_request() -> None:
    response = litellm.embedding(
        model="openai/flat-priced-embedding",
        api_key="fake-key",
        input=["first input", "second input"],
        mock_response=[0.1, 0.2],
        input_cost_per_query=REQUEST_PRICE,
    )

    assert response._hidden_params["response_cost"] == pytest.approx(REQUEST_PRICE)


class _CostRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.costs: list[float] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self.costs.append(kwargs["standard_logging_object"]["response_cost"])


@pytest.mark.asyncio
async def test_streamed_completion_logs_flat_price_once() -> None:
    recorder = _CostRecorder()
    litellm.callbacks = [recorder]
    try:
        stream = await _router(input_cost_per_query=REQUEST_PRICE).acompletion(
            model="flat-model",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="a streamed answer with several chunks",
            stream=True,
        )
        _ = [chunk async for chunk in stream]
        for _attempt in range(50):
            if recorder.costs:
                break
            await asyncio.sleep(0.05)
    finally:
        litellm.callbacks = []

    assert recorder.costs == [pytest.approx(REQUEST_PRICE)]


def test_token_priced_deployment_still_charges_by_tokens() -> None:
    router = _router(input_cost_per_token=INPUT_COST_PER_TOKEN, output_cost_per_token=OUTPUT_COST_PER_TOKEN)

    response = router.completion(
        model="flat-model", messages=[{"role": "user", "content": "hi"}], mock_response=_mock_response(1_000)
    )

    assert response._hidden_params["response_cost"] == pytest.approx(
        1_000 * INPUT_COST_PER_TOKEN + 1_000 * OUTPUT_COST_PER_TOKEN
    )
