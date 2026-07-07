import httpx
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.mistral.conversations.transformation import (
    MistralConversationsConfig,
)
from litellm.llms.mistral.cost_calculator import (
    MISTRAL_WEB_SEARCH_COST_PER_CALL,
    cost_per_web_search_request,
)
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage


@pytest.fixture(autouse=True)
def add_mistral_api_key_to_env(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-mistral-api-key-12345")


def test_per_call_rate_matches_published_price():
    assert MISTRAL_WEB_SEARCH_COST_PER_CALL == pytest.approx(0.03)


@pytest.mark.parametrize("num_requests, expected", [(0, 0.0), (1, 0.03), (3, 0.09)])
def test_cost_per_web_search_request_scales_with_count(num_requests, expected):
    details = PromptTokensDetailsWrapper(web_search_requests=num_requests) if num_requests else None
    usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, prompt_tokens_details=details)
    model_info = litellm.get_model_info("mistral/mistral-medium-latest")
    assert cost_per_web_search_request(usage=usage, model_info=model_info) == pytest.approx(expected)


def test_cost_per_web_search_request_no_details_is_zero():
    usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    model_info = litellm.get_model_info("mistral/mistral-medium-latest")
    assert cost_per_web_search_request(usage=usage, model_info=model_info) == 0.0


def test_get_cost_for_web_search_request_dispatches_for_mistral():
    """The shared dispatcher must route provider 'mistral' to this cost calculator;
    without the mistral branch the surcharge silently becomes None."""
    from litellm.llms import get_cost_for_web_search_request

    usage = Usage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        prompt_tokens_details=PromptTokensDetailsWrapper(web_search_requests=2),
    )
    model_info = litellm.get_model_info("mistral/mistral-medium-latest")
    assert get_cost_for_web_search_request("mistral", usage=usage, model_info=model_info) == pytest.approx(0.06)


def _conversation_response(connectors: dict = None, execution_names: list = None) -> ModelResponse:
    """Build a transformed ModelResponse from a Conversations payload.

    ``connectors`` populates the authoritative ``usage.connectors`` billed counts;
    ``execution_names`` adds tool.execution entries to exercise the fallback path.
    """
    outputs = [{"type": "tool.execution", "name": name} for name in (execution_names or [])]
    outputs.append(
        {
            "type": "message.output",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Spain won."},
                {"type": "tool_reference", "tool": "web_search", "title": "UEFA", "url": "https://uefa.com"},
            ],
        }
    )
    usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    if connectors is not None:
        usage["connectors"] = connectors
    raw = httpx.Response(
        200,
        json={"conversation_id": "c1", "outputs": outputs, "usage": usage},
        request=httpx.Request("POST", "https://api.mistral.ai/v1/conversations"),
    )
    logging_obj = Logging(
        model="mistral-medium-latest",
        messages=[{"role": "user", "content": "x"}],
        stream=False,
        call_type="completion",
        start_time=None,
        litellm_call_id="1",
        function_id="f",
    )
    return MistralConversationsConfig().transform_response(
        model="mistral-medium-latest",
        raw_response=raw,
        model_response=ModelResponse(),
        logging_obj=logging_obj,
        request_data={},
        messages=[],
        optional_params={},
        litellm_params={},
        encoding=None,
    )


def _web_search_surcharge(response: ModelResponse) -> float:
    prompt_cost, completion_cost = litellm.cost_per_token(
        model="mistral-medium-latest", prompt_tokens=100, completion_tokens=20, custom_llm_provider="mistral"
    )
    full = litellm.completion_cost(
        completion_response=response, model="mistral/mistral-medium-latest", custom_llm_provider="mistral"
    )
    return full - (prompt_cost + completion_cost)


def test_completion_cost_adds_web_search_on_top_of_tokens():
    """The web search charge (count x $0.03) is added to the token cost end to end."""
    response = _conversation_response(connectors={"web_search": 2})
    assert response.usage.prompt_tokens_details.web_search_requests == 2
    assert _web_search_surcharge(response) == pytest.approx(0.06)


def test_completion_cost_premium_rate():
    """web_search_premium calls are billed at $0.05 each."""
    response = _conversation_response(connectors={"web_search_premium": 2})
    assert response.usage.web_search_premium_requests == 2
    assert _web_search_surcharge(response) == pytest.approx(0.10)


def test_completion_cost_mixed_standard_and_premium():
    """A turn mixing tiers is billed 1 x $0.03 + 1 x $0.05."""
    response = _conversation_response(connectors={"web_search": 1, "web_search_premium": 1})
    assert response.usage.prompt_tokens_details.web_search_requests == 2
    assert _web_search_surcharge(response) == pytest.approx(0.08)


def test_other_connectors_are_not_billed_as_web_search():
    """A non-web-search connector in usage.connectors adds no web search charge."""
    response = _conversation_response(connectors={"code_interpreter": 3})
    assert response.usage.prompt_tokens_details is None
    assert _web_search_surcharge(response) == 0.0


def test_falls_back_to_tool_execution_count_without_connectors():
    """Without usage.connectors, the count comes from tool.execution entries by name."""
    response = _conversation_response(execution_names=["web_search", "web_search_premium", "code_interpreter"])
    assert response.usage.prompt_tokens_details.web_search_requests == 2
    assert _web_search_surcharge(response) == pytest.approx(0.08)


def test_completion_cost_no_web_search_has_no_surcharge():
    """A Mistral response without web search is billed on tokens only."""
    raw = httpx.Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 1,
            "model": "mistral-medium-latest",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        },
        request=httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions"),
    )
    logging_obj = Logging(
        model="mistral-medium-latest",
        messages=[{"role": "user", "content": "x"}],
        stream=False,
        call_type="completion",
        start_time=None,
        litellm_call_id="1",
        function_id="f",
    )
    response = litellm.MistralConfig().transform_response(
        model="mistral-medium-latest",
        raw_response=raw,
        model_response=ModelResponse(),
        logging_obj=logging_obj,
        request_data={},
        messages=[],
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    prompt_cost, completion_cost = litellm.cost_per_token(
        model="mistral-medium-latest", prompt_tokens=100, completion_tokens=20, custom_llm_provider="mistral"
    )
    full = litellm.completion_cost(
        completion_response=response, model="mistral/mistral-medium-latest", custom_llm_provider="mistral"
    )
    assert full == pytest.approx(prompt_cost + completion_cost)
