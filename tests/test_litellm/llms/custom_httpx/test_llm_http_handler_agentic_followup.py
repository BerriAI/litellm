from typing import Final
from unittest.mock import Mock

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.chat_completion_agentic_loop import (
    _execute_chat_completion_agentic_plan,
)
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.integrations.custom_logger import AgenticLoopPlan, AgenticLoopRequestPatch


@pytest.fixture
def followup_calls(monkeypatch) -> list[dict[str, object]]:
    calls: Final[list[dict[str, object]]] = []

    async def fake_followup(**kwargs: object) -> str:
        calls.append(kwargs)
        return "followup-response"

    monkeypatch.setattr(litellm, "aresponses", fake_followup)
    monkeypatch.setattr(litellm, "acompletion", fake_followup)
    return calls


def _plan_copying_request_kwargs(
    request_kwargs: dict[str, object], optional_params: dict[str, object]
) -> AgenticLoopPlan:
    """Shaped like the Headroom and Compresr guardrails, which copy the whole request kwargs into the plan"""
    return AgenticLoopPlan(
        run_agentic_loop=True,
        request_patch=AgenticLoopRequestPatch(
            model="gpt-5",
            messages=[{"role": "user", "content": "x"}],
            optional_params=dict(optional_params),
            kwargs=dict(request_kwargs),
        ),
    )


@pytest.mark.asyncio
async def test_responses_followup_survives_a_plan_that_repeats_a_request_param(followup_calls):
    request_kwargs: Final = {"prompt_cache_key": "thread-1", "metadata": {"user": "u1"}}
    optional_params: Final = {"prompt_cache_key": "thread-1"}

    response: Final = await BaseLLMHTTPHandler()._execute_responses_agentic_plan(
        plan=_plan_copying_request_kwargs(request_kwargs, optional_params),
        model="gpt-5",
        response_api_optional_request_params=dict(optional_params),
        logging_obj=Mock(litellm_call_id="call-1"),
        kwargs=dict(request_kwargs),
        depth=0,
        max_loops=3,
        fingerprints=[],
        fingerprint="fp",
        callback=CustomLogger(),
    )

    assert response == "followup-response"
    assert len(followup_calls) == 1
    assert followup_calls[0]["prompt_cache_key"] == "thread-1"
    assert followup_calls[0]["metadata"] == {"user": "u1"}
    assert followup_calls[0]["_agentic_loop_depth"] == 1


@pytest.mark.asyncio
async def test_responses_followup_sends_the_plans_request_param_over_a_stale_kwargs_copy(followup_calls):
    await BaseLLMHTTPHandler()._execute_responses_agentic_plan(
        plan=AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=AgenticLoopRequestPatch(
                model="gpt-5",
                messages=[{"role": "user", "content": "x"}],
                optional_params={"prompt_cache_key": "from-plan-params"},
                kwargs={"prompt_cache_key": "stale-copy"},
            ),
        ),
        model="gpt-5",
        response_api_optional_request_params={"prompt_cache_key": "from-request"},
        logging_obj=Mock(litellm_call_id="call-1"),
        kwargs={},
        depth=0,
        max_loops=3,
        fingerprints=[],
        fingerprint="fp",
        callback=CustomLogger(),
    )

    assert followup_calls[0]["prompt_cache_key"] == "from-plan-params"


@pytest.mark.asyncio
async def test_handler_chat_followup_survives_a_plan_that_repeats_a_request_param(followup_calls):
    request_kwargs: Final = {"temperature": 0.2, "api_base": "https://a", "model": "gpt-5"}
    optional_params: Final = {"temperature": 0.2}

    response: Final = await BaseLLMHTTPHandler()._execute_chat_completion_agentic_plan(
        plan=_plan_copying_request_kwargs(request_kwargs, optional_params),
        model="gpt-5",
        messages=[{"role": "user", "content": "x"}],
        optional_params=dict(optional_params),
        kwargs=dict(request_kwargs),
        custom_llm_provider="openai",
        depth=0,
        max_loops=3,
        fingerprints=[],
        fingerprint="fp",
    )

    assert response == "followup-response"
    assert followup_calls[0]["temperature"] == 0.2
    assert followup_calls[0]["api_base"] == "https://a"
    assert followup_calls[0]["model"] == "openai/gpt-5"


@pytest.mark.asyncio
async def test_core_chat_followup_survives_request_kwargs_that_repeat_a_request_param(followup_calls):
    request_kwargs: Final = {"temperature": 0.2, "api_base": "https://a"}
    optional_params: Final = {"temperature": 0.2}

    response: Final = await _execute_chat_completion_agentic_plan(
        plan=_plan_copying_request_kwargs(request_kwargs, optional_params),
        callback=CustomLogger(),
        model="gpt-5",
        optional_params=dict(optional_params),
        kwargs=dict(request_kwargs),
        logging_obj=Mock(litellm_call_id="call-1"),
        custom_llm_provider="openai",
        depth=0,
        max_loops=3,
        fingerprints=[],
        fingerprint="fp",
    )

    assert response == "followup-response"
    assert followup_calls[0]["temperature"] == 0.2
    assert followup_calls[0]["api_base"] == "https://a"
