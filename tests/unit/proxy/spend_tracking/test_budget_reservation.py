from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

import litellm
from litellm.caching import DualCache
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.proxy import proxy_server
from litellm.proxy._types import (
    LiteLLM_OrganizationTable,
    LiteLLM_ProjectTableCachedObj,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    project_cache_key,
    team_membership_reservation_cache_key,
)
from litellm.proxy.spend_tracking.budget_reservation import (
    DEFAULT_MAX_OUTPUT_TOKENS_FALLBACK,
    _get_team_member_budget_counter,
    estimate_request_input_cost,
    estimate_request_max_cost,
    get_budget_window_start,
    release_unbound_budget_reservation,
    count_request_input_tokens,
    reserve_budget_for_request,
)
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

TOKEN_COUNTING_ROUTES: Final = (
    "/responses/input_tokens",
    "/v1/responses/input_tokens",
    "/openai/v1/responses/input_tokens",
    "/utils/token_counter",
    "/v1/messages/count_tokens",
    "/v1beta/models/gemini-3.8-flash:countTokens",
    "/models/gemini-3.8-flash:countTokens",
    "/bedrock/v1/messages/count-tokens",
    "/bedrock/model/us.anthropic.claude-sonnet-4-6/count-tokens",
    "/vertex_ai/v1/projects/p/locations/us-east5/publishers/anthropic/models/count-tokens:rawPredict",
    "/vertex-ai/v1/projects/p/locations/us-east5/publishers/anthropic/models/count-tokens:rawPredict",
)


def _budgeted_token() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", token="hashed-token", max_budget=100.0, spend=0.0)


async def _reserve(route: str) -> dict | None:
    return await reserve_budget_for_request(
        request_body={"model": "gpt-4o", "input": "hello"},
        route=route,
        llm_router=None,
        valid_token=_budgeted_token(),
        team_object=None,
        user_object=None,
        prisma_client=None,
        user_api_key_cache=UserApiKeyCache(),
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", TOKEN_COUNTING_ROUTES)
async def test_token_counting_routes_are_exempt_from_budget_reservation(route):
    assert await _reserve(route) is None


@pytest.mark.asyncio
async def test_non_exempt_llm_route_still_reserves_budget():
    reservation: Final = await _reserve("/v1/responses")

    assert reservation is not None
    assert reservation["reserved_cost"] > 0


@pytest.mark.asyncio
async def test_reservation_carries_the_admission_input_token_count():
    reservation: Final = await _reserve("/v1/responses")
    expected: Final = litellm.token_counter(model="gpt-4o", text="hello")

    assert reservation is not None
    assert expected > 0
    assert reservation["input_tokens"] == expected


ANTHROPIC_MESSAGES: Final = [{"role": "user", "content": "hello!!!"}]
COUNT_TOKENS_REQUESTS: Final[tuple[tuple[str, dict[str, object]], ...]] = (
    ("/v1/messages/count_tokens", {"model": "claude-sonnet-5", "messages": ANTHROPIC_MESSAGES}),
    ("/v1beta/models/gemini-3.8-flash:countTokens", {"contents": [{"role": "user", "parts": [{"text": "hello!!!"}]}]}),
    (
        "/vertex_ai/v1/projects/p/locations/us-east5/publishers/anthropic/models/count-tokens:rawPredict",
        {"model": "claude-sonnet-5", "messages": ANTHROPIC_MESSAGES},
    ),
    ("/bedrock/v1/messages/count-tokens", {"model": "claude-sonnet-5", "messages": ANTHROPIC_MESSAGES}),
)
TINY_BUDGET_KEY_TOKEN: Final = "hashed-count-tokens-key"


@pytest.fixture
def spend_counter_cache(monkeypatch: pytest.MonkeyPatch) -> DualCache:
    cache: Final = DualCache()
    monkeypatch.setattr(proxy_server, "spend_counter_cache", cache)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    return cache


async def _reserve_for_tiny_budget_key(route: str, request_body: dict[str, object]) -> dict[str, object] | None:
    return await reserve_budget_for_request(
        request_body=request_body,
        route=route,
        llm_router=None,
        valid_token=UserAPIKeyAuth(token=TINY_BUDGET_KEY_TOKEN, max_budget=0.01, spend=0.0),
        team_object=None,
        user_object=None,
        prisma_client=None,
        user_api_key_cache=UserApiKeyCache(),
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("route", "request_body"), COUNT_TOKENS_REQUESTS)
async def test_repeated_token_counting_never_touches_a_tiny_budget(
    spend_counter_cache: DualCache, route: str, request_body: dict[str, object]
):
    counter_key: Final = f"spend:key:{TINY_BUDGET_KEY_TOKEN}"

    assert await _reserve_for_tiny_budget_key(route, request_body) is None
    assert await _reserve_for_tiny_budget_key(route, request_body) is None
    assert spend_counter_cache.in_memory_cache.get_cache(key=counter_key) is None

    completion: Final = await _reserve_for_tiny_budget_key(
        "/v1/messages", {"model": "claude-sonnet-5", "max_tokens": 16, "messages": ANTHROPIC_MESSAGES}
    )
    assert completion is not None
    reserved_cost: Final = completion["reserved_cost"]
    assert isinstance(reserved_cost, float)
    assert reserved_cost > 0
    assert spend_counter_cache.in_memory_cache.get_cache(key=counter_key) == pytest.approx(reserved_cost)


BEDROCK_SONNET: Final = "us.anthropic.claude-sonnet-4-6"
CONVERSE_BODY: Final = {
    "messages": [{"role": "user", "content": [{"text": "Reply with one word: pong"}]}],
    "inferenceConfig": {"maxTokens": 5},
}
INVOKE_BODY: Final = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 5,
    "messages": [{"role": "user", "content": "Reply with one word: pong"}],
}


def test_bedrock_converse_body_reserves_the_prompt_not_the_context_window():
    converse_cost: Final = estimate_request_max_cost(
        request_body=CONVERSE_BODY,
        route=f"/bedrock/model/{BEDROCK_SONNET}/converse",
        llm_router=None,
        input_token_counts={},
    )
    invoke_cost: Final = estimate_request_max_cost(
        request_body=INVOKE_BODY,
        route=f"/bedrock/model/{BEDROCK_SONNET}/invoke",
        llm_router=None,
        input_token_counts={},
    )
    assert converse_cost is not None and invoke_cost is not None
    assert invoke_cost < converse_cost < 2 * invoke_cost


GEMINI_PROMPT: Final = "Say hello in five words"
GEMINI_SYSTEM_PARTS: Final = ("Answer tersely", "Use British spelling")


@pytest.mark.parametrize(
    "route",
    (
        "/vertex_ai/v1/projects/p/locations/global/publishers/google/models/gemini-3.8-flash:generateContent",
        "/gemini/v1beta/models/gemini-3.8-flash:generateContent",
        "/v1beta/models/gemini-3.8-flash:generateContent",
    ),
)
@pytest.mark.parametrize(
    ("native_output_config", "chat_output_limit"),
    (
        ({"generationConfig": {"maxOutputTokens": 5}}, {"max_tokens": 5}),
        ({"generationConfig": {"maxOutputTokens": 5, "candidateCount": 3}}, {"max_tokens": 5, "n": 3}),
        ({"generation_config": {"max_output_tokens": 5, "candidate_count": 3}}, {"n": 3}),
        ({"generationConfig": {"candidate_count": 3}}, {"n": 3}),
        ({}, {}),
    ),
)
@pytest.mark.parametrize("system_texts", (GEMINI_SYSTEM_PARTS[:1], GEMINI_SYSTEM_PARTS))
def test_gemini_generate_content_body_reserves_the_prompt_not_the_context_window(
    route: str,
    native_output_config: dict[str, object],
    chat_output_limit: dict[str, object],
    system_texts: tuple[str, ...],
) -> None:
    native_cost: Final = estimate_request_max_cost(
        request_body={
            "systemInstruction": {"parts": [{"text": text} for text in system_texts]},
            "contents": [{"role": "user", "parts": [{"text": GEMINI_PROMPT}]}],
            **native_output_config,
        },
        route=route,
        llm_router=None,
        input_token_counts={},
    )
    chat_cost: Final = estimate_request_max_cost(
        request_body={
            "model": "gemini-3.8-flash",
            "messages": [
                {"role": "system", "content": "\n".join(system_texts)},
                {"role": "user", "content": GEMINI_PROMPT},
            ],
            **chat_output_limit,
        },
        route="/chat/completions",
        llm_router=None,
        input_token_counts={},
    )
    assert native_cost is not None
    assert native_cost == chat_cost


GEMINI_ROUTE: Final = "/v1beta/models/gemini-3.8-flash:generateContent"
GEMINI_TEXT_CONTENT: Final = {"role": "user", "parts": [{"text": GEMINI_PROMPT}]}
GEMINI_AUDIO_PART: Final = {"inlineData": {"mimeType": "audio/wav", "data": "UklGRg" * 40_000}}


def _gemini_hold(body: dict[str, object], input_tokens: int | None = None) -> float | None:
    return estimate_request_max_cost(
        request_body=body,
        route=GEMINI_ROUTE,
        llm_router=None,
        input_token_counts={} if input_tokens is None else {"gemini-3.8-flash": input_tokens},
    )


@pytest.mark.parametrize(
    "body",
    (
        {"contents": [{"role": "user", "parts": [{"text": GEMINI_PROMPT}, GEMINI_AUDIO_PART]}]},
        {"contents": [{"role": "user", "parts": [{"fileData": {"mimeType": "video/mp4", "fileUri": "gs://b/v.mp4"}}]}]},
        {"contents": [{"role": "user", "parts": {"fileData": {"mimeType": "video/mp4", "fileUri": "gs://b/v.mp4"}}}]},
        {"contents": [{"role": "user", "parts": [{"inline_data": {"mime_type": "image/png", "data": "iVBORw0K"}}]}]},
        {"contents": [{"role": "model", "parts": [{"executableCode": {"language": "PYTHON", "code": "print(1)"}}]}]},
        {"contents": [GEMINI_TEXT_CONTENT, {"role": "function", "parts": [{"text": GEMINI_PROMPT * 500}]}]},
        {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"functionResponse": {"name": "listen", "response": {}, "parts": [GEMINI_AUDIO_PART]}}],
                }
            ]
        },
        {"systemInstruction": {"parts": [GEMINI_AUDIO_PART]}, "contents": [GEMINI_TEXT_CONTENT]},
        {"systemInstruction": {"parts": GEMINI_AUDIO_PART}, "contents": [GEMINI_TEXT_CONTENT]},
        {"tools": [{"function_declarations": [{"name": "lookup"}]}], "contents": [GEMINI_TEXT_CONTENT]},
        {"tools": [{"googleSearch": {}}], "contents": [GEMINI_TEXT_CONTENT]},
        {"cachedContent": "cachedContents/abc", "contents": [GEMINI_TEXT_CONTENT]},
    ),
)
def test_gemini_body_the_prompt_count_cannot_see_whole_keeps_the_context_window_hold(body: dict[str, object]) -> None:
    context_window: Final = litellm.get_model_info("gemini-3.8-flash")["max_input_tokens"]
    assert context_window is not None
    assert _gemini_hold(body) == _gemini_hold({"contents": [GEMINI_TEXT_CONTENT]}, input_tokens=context_window)


def test_gemini_tool_turns_and_thought_signatures_count_as_prompt() -> None:
    context_window: Final = litellm.get_model_info("gemini-3.8-flash")["max_input_tokens"]
    assert context_window is not None
    tool_turns: Final = {
        "tools": [{"functionDeclarations": [{"name": "lookup", "parameters": {"type": "object"}}]}],
        "contents": [
            GEMINI_TEXT_CONTENT,
            {"role": "model", "parts": [{"functionCall": {"name": "lookup", "args": {}}, "thoughtSignature": "c2ln"}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "lookup", "response": {"answer": "hi"}}}]},
        ],
    }
    text_hold: Final = _gemini_hold({"contents": [GEMINI_TEXT_CONTENT]})
    tool_turns_hold: Final = _gemini_hold(tool_turns)
    context_window_hold: Final = _gemini_hold({"contents": [GEMINI_TEXT_CONTENT]}, input_tokens=context_window)
    assert text_hold is not None and tool_turns_hold is not None and context_window_hold is not None
    assert text_hold < tool_turns_hold < context_window_hold


GEMINI_CHAT_BODY: Final = {
    "model": "gemini-3.8-flash",
    "messages": [{"role": "user", "content": GEMINI_PROMPT}],
}


def _chat_hold(body: dict[str, object]) -> float | None:
    return estimate_request_max_cost(
        request_body=body,
        route="/chat/completions",
        llm_router=None,
        input_token_counts={},
    )


@pytest.mark.parametrize(
    "gemini_output_fields",
    (
        {"generationConfig": {"maxOutputTokens": 5}},
        {"generationConfig": {"candidateCount": 3}},
        {"generation_config": {"candidate_count": 3}},
        {"contents": [], "generationConfig": {"maxOutputTokens": 5}},
    ),
)
def test_gemini_output_fields_on_a_chat_body_leave_its_hold_alone(gemini_output_fields: dict[str, object]) -> None:
    plain_hold: Final = _chat_hold(GEMINI_CHAT_BODY)
    assert plain_hold is not None
    assert _chat_hold({**GEMINI_CHAT_BODY, **gemini_output_fields}) == plain_hold


@pytest.mark.parametrize("cap", (0, -1000))
def test_a_gemini_output_cap_of_zero_or_below_holds_the_default_output(cap: int) -> None:
    uncapped_hold: Final = _gemini_hold({"contents": [GEMINI_TEXT_CONTENT]})
    assert uncapped_hold is not None
    assert (
        _gemini_hold({"contents": [GEMINI_TEXT_CONTENT], "generationConfig": {"maxOutputTokens": cap}})
        == uncapped_hold
    )


@pytest.mark.parametrize(
    ("output_fields", "same_hold_fields"),
    (
        ({"generationConfig": {"max_output_tokens": 30000}}, {"generationConfig": {"maxOutputTokens": 30000}}),
        ({"generation_config": {"maxOutputTokens": 30000}}, {"generationConfig": {"maxOutputTokens": 30000}}),
        ({"generation_config": {"max_output_tokens": 30000}}, {"generationConfig": {"maxOutputTokens": 30000}}),
        (
            {"generationConfig": {"maxOutputTokens": 8000}, "generation_config": {"max_output_tokens": 30000}},
            {"generationConfig": {"maxOutputTokens": 30000}},
        ),
        ({"generation_config": {"max_output_tokens": 5}}, {}),
        (
            {"generationConfig": {"maxOutputTokens": 8000, "max_output_tokens": 5}},
            {"generationConfig": {"maxOutputTokens": 8000}},
        ),
    ),
)
def test_an_output_cap_spelling_v1beta_drops_can_only_raise_the_hold(
    output_fields: dict[str, object], same_hold_fields: dict[str, object]
) -> None:
    expected_hold: Final = _gemini_hold({"contents": [GEMINI_TEXT_CONTENT], **same_hold_fields})
    assert expected_hold is not None
    assert _gemini_hold({"contents": [GEMINI_TEXT_CONTENT], **output_fields}) == expected_hold


UNPRICEABLE_COUNTS: Final = (10**400, "9" * 400, 1e308, math.inf)


@pytest.mark.parametrize("count", UNPRICEABLE_COUNTS)
def test_a_candidate_count_too_large_to_price_holds_one_candidate(count: object) -> None:
    one_candidate_hold: Final = _gemini_hold({"contents": [GEMINI_TEXT_CONTENT]})
    assert one_candidate_hold is not None
    assert (
        _gemini_hold({"contents": [GEMINI_TEXT_CONTENT], "generationConfig": {"candidateCount": count}})
        == one_candidate_hold
    )


WIDE_WINDOW: Final = {"max_input_tokens": 1_000_000, "max_output_tokens": 100_000}
LOW_OUTPUT_CEILING: Final = {**WIDE_WINDOW, "max_output_tokens": 1000}


def _priced_gemini_hold(
    model_info: dict[str, object], generation_config: dict[str, object], input_tokens: int = 10
) -> float | None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "assistant",
                "litellm_params": {
                    "model": "vertex_ai/gemini-3.8-flash",
                    "input_cost_per_token": 1e-6,
                    "output_cost_per_token": 4e-6,
                },
                "model_info": model_info,
            }
        ]
    )
    return estimate_request_max_cost(
        request_body={"contents": [GEMINI_TEXT_CONTENT], "generationConfig": generation_config},
        route="/v1beta/models/assistant:generateContent",
        llm_router=router,
        input_token_counts={"assistant": input_tokens},
    )


@pytest.mark.parametrize(
    ("model_info", "generation_config", "same_hold_model_info", "same_hold_config"),
    (
        (WIDE_WINDOW, {"maxOutputTokens": 8000, "candidateCount": 4}, WIDE_WINDOW, {"maxOutputTokens": 32000}),
        (WIDE_WINDOW, {"maxOutputTokens": 20000, "candidate_count": 2}, WIDE_WINDOW, {"maxOutputTokens": 40000}),
        (WIDE_WINDOW, {"candidateCount": 2}, WIDE_WINDOW, {"maxOutputTokens": 2 * DEFAULT_MAX_OUTPUT_TOKENS_FALLBACK}),
        (LOW_OUTPUT_CEILING, {"candidateCount": 4}, {**WIDE_WINDOW, "max_output_tokens": 4000}, {}),
    ),
)
def test_candidates_hold_what_one_candidate_writing_all_their_output_holds(
    model_info: dict[str, object],
    generation_config: dict[str, object],
    same_hold_model_info: dict[str, object],
    same_hold_config: dict[str, object],
) -> None:
    expected_hold: Final = _priced_gemini_hold(same_hold_model_info, same_hold_config)
    assert expected_hold is not None
    assert _priced_gemini_hold(model_info, generation_config) == expected_hold


@pytest.mark.parametrize(
    ("model_info", "generation_config"),
    (
        (WIDE_WINDOW, {"candidateCount": 1e300}),
        (WIDE_WINDOW, {"maxOutputTokens": 100_000, "candidateCount": 8}),
        (LOW_OUTPUT_CEILING, {"candidateCount": 1e300}),
    ),
)
def test_candidates_never_hold_more_than_the_whole_context_window_hold(
    model_info: dict[str, object], generation_config: dict[str, object]
) -> None:
    context_window_hold: Final = _priced_gemini_hold(model_info, {}, input_tokens=1_000_000)
    assert context_window_hold is not None
    assert _priced_gemini_hold(model_info, generation_config) == context_window_hold


def test_candidates_never_hold_less_than_one_candidate() -> None:
    narrow_window: Final = {"max_input_tokens": 10, "max_output_tokens": 100_000}
    one_candidate_hold: Final = _priced_gemini_hold(narrow_window, {"maxOutputTokens": 50_000})
    assert one_candidate_hold is not None
    assert _priced_gemini_hold(narrow_window, {"maxOutputTokens": 50_000, "candidateCount": 2}) == one_candidate_hold


@pytest.mark.parametrize("count", UNPRICEABLE_COUNTS)
def test_an_n_too_large_to_price_holds_one_choice(count: object) -> None:
    one_choice_hold: Final = _chat_hold(GEMINI_CHAT_BODY)
    assert one_choice_hold is not None
    assert _chat_hold({**GEMINI_CHAT_BODY, "n": count}) == one_choice_hold


def _group_generate_content_hold(router: Router, generation_config: dict[str, object]) -> float | None:
    return estimate_request_max_cost(
        request_body={"contents": [GEMINI_TEXT_CONTENT], "generationConfig": generation_config},
        route="/v1beta/models/assistant:generateContent",
        llm_router=router,
        input_token_counts={"assistant": 10},
    )


@pytest.mark.parametrize(
    ("backend_models", "candidates_are_generated"),
    (
        (("vertex_ai/claude-haiku-5-5",), False),
        (("anthropic/claude-haiku-5-5",), False),
        (("gemini/gemini-3.8-flash",), True),
        (("vertex_ai/gemini-3.8-flash",), True),
        (("vertex_ai/claude-haiku-5-5", "gemini/gemini-3.8-flash"), True),
    ),
)
def test_a_candidate_count_is_held_only_when_a_deployment_generates_the_candidates(
    backend_models: tuple[str, ...], candidates_are_generated: bool
) -> None:
    router: Final = Router(
        model_list=[
            {"model_name": "assistant", "litellm_params": {"model": model, "api_key": "sk-test"}}
            for model in backend_models
        ]
    )
    one_candidate_hold: Final = _group_generate_content_hold(router, {"maxOutputTokens": 100})
    three_candidate_hold: Final = _group_generate_content_hold(router, {"maxOutputTokens": 100, "candidateCount": 3})

    assert one_candidate_hold is not None
    assert three_candidate_hold is not None
    assert (three_candidate_hold > one_candidate_hold) is candidates_are_generated
    assert three_candidate_hold >= one_candidate_hold


def test_a_candidate_count_for_a_gemini_name_no_deployment_serves_is_held() -> None:
    router: Final = Router(
        model_list=[
            {"model_name": "assistant", "litellm_params": {"model": "anthropic/claude-haiku-5-5", "api_key": "sk-test"}}
        ]
    )
    one_candidate_hold: Final = estimate_request_max_cost(
        request_body={"contents": [GEMINI_TEXT_CONTENT], "generationConfig": {"maxOutputTokens": 100}},
        route=GEMINI_ROUTE,
        llm_router=router,
        input_token_counts={"gemini-3.8-flash": 10},
    )
    three_candidate_hold: Final = estimate_request_max_cost(
        request_body={
            "contents": [GEMINI_TEXT_CONTENT],
            "generationConfig": {"maxOutputTokens": 100, "candidateCount": 3},
        },
        route=GEMINI_ROUTE,
        llm_router=router,
        input_token_counts={"gemini-3.8-flash": 10},
    )

    assert one_candidate_hold is not None
    assert three_candidate_hold is not None
    assert three_candidate_hold > one_candidate_hold


N_SENDING_GROUPS: Final = (
    Router(
        model_list=[
            {"model_name": "assistant", "litellm_params": {"model": "anthropic/claude-haiku-5-5", "api_key": "sk"}},
            {"model_name": "backup", "litellm_params": {"model": "openai/gpt-5.4-mini", "api_key": "sk"}},
        ],
        fallbacks=[{"assistant": ["backup"]}],
    ),
    Router(
        model_list=[
            {
                "model_name": "assistant",
                "litellm_params": {
                    "model": "mistral/mistral-small-latest",
                    "api_key": "sk",
                    "allowed_openai_params": ["n"],
                },
            }
        ]
    ),
)


@pytest.mark.parametrize("router", N_SENDING_GROUPS)
def test_an_n_is_held_when_a_fallback_or_an_allowlist_can_still_send_it(router: Router) -> None:
    holds: Final = tuple(
        estimate_request_max_cost(
            request_body={"model": "assistant", "messages": GEMINI_CHAT_BODY["messages"], **output_fields},
            route="/v1/chat/completions",
            llm_router=router,
            input_token_counts={"assistant": 10},
        )
        for output_fields in ({"max_tokens": 100}, {"max_tokens": 100, "n": 3})
    )

    one_choice_hold, three_choice_hold = holds
    assert one_choice_hold is not None
    assert three_choice_hold is not None
    assert three_choice_hold > one_choice_hold


HOLDS_FAR_OVER_A_TINY_BUDGET: Final = (
    ("/v1/chat/completions", {**GEMINI_CHAT_BODY, "n": 1e300}),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("route", "request_body"), HOLDS_FAR_OVER_A_TINY_BUDGET)
async def test_a_hold_far_over_the_budget_keeps_the_spend_already_counted(
    spend_counter_cache: DualCache, route: str, request_body: dict[str, object]
) -> None:
    counter_key: Final = f"spend:key:{TINY_BUDGET_KEY_TOKEN}"
    counted_spend: Final = 0.004
    hold_estimate: Final = estimate_request_max_cost(request_body=request_body, route=route, llm_router=None)
    assert hold_estimate is not None and 1e20 < hold_estimate < math.inf
    await spend_counter_cache.async_set_cache(key=counter_key, value=counted_spend)

    reservation: Final = await _reserve_for_tiny_budget_key(route, request_body)

    assert reservation is not None
    assert spend_counter_cache.in_memory_cache.get_cache(key=counter_key) == pytest.approx(0.01)
    await release_unbound_budget_reservation(reservation)
    assert spend_counter_cache.in_memory_cache.get_cache(key=counter_key) == pytest.approx(counted_spend)


def test_a_negative_max_tokens_holds_the_default_output() -> None:
    uncapped_hold: Final = _chat_hold(GEMINI_CHAT_BODY)
    assert uncapped_hold is not None
    assert _chat_hold({**GEMINI_CHAT_BODY, "max_tokens": -1000}) == uncapped_hold


def test_a_text_completion_with_max_tokens_zero_holds_only_its_prompt() -> None:
    body: Final = {"model": "gemini-3.8-flash", "prompt": GEMINI_PROMPT * 50, "max_tokens": 0}
    prompt_cost: Final = estimate_request_input_cost(request_body=body, route="/v1/completions", llm_router=None)
    assert prompt_cost is not None and prompt_cost > 0
    assert estimate_request_max_cost(request_body=body, route="/v1/completions", llm_router=None) == prompt_cost


GEMINI_LONG_CONTENTS: Final = {"contents": [{"role": "user", "parts": [{"text": GEMINI_PROMPT * 500}]}]}


@pytest.mark.parametrize(
    "stray_fields",
    (
        {"prompt": ""},
        {"messages": [], "n": 5},
        {"input": "", "max_tokens": 1},
        {"query": "", "documents": [], "max_completion_tokens": 1},
        {"inferenceConfig": {"maxTokens": 1}},
    ),
)
def test_fields_of_other_apis_on_a_gemini_route_leave_its_hold_alone(stray_fields: dict[str, object]) -> None:
    body: Final = {**GEMINI_LONG_CONTENTS, "generationConfig": {"maxOutputTokens": 4000, "candidateCount": 2}}
    gemini_hold: Final = _gemini_hold(body)
    assert gemini_hold is not None
    assert _gemini_hold({**body, **stray_fields}) == gemini_hold


@pytest.mark.parametrize("stray_input", ({"prompt": ""}, {"messages": []}, {"input": ""}))
@pytest.mark.parametrize("model", ("gpt-4", "claude-sonnet-5-5"))
@pytest.mark.asyncio
async def test_a_stray_input_field_on_a_gemini_route_never_hides_the_counted_contents(
    model: str, stray_input: dict[str, object]
) -> None:
    async def counted(body: dict[str, object]) -> dict[str, int]:
        return dict(
            await count_request_input_tokens(
                request_body=body,
                route=f"/v1beta/models/{model}:generateContent",
                llm_router=None,
                team_id=None,
                raw_body=json.dumps(body).encode(),
            )
        )

    contents_counts: Final = await counted(GEMINI_LONG_CONTENTS)
    assert contents_counts[model] > 1000
    assert await counted({**GEMINI_LONG_CONTENTS, **stray_input}) == contents_counts


def _tiered_deployment(input_cost_per_token: float) -> Deployment:
    return Deployment(
        model_name="tiered-group",
        litellm_params=LiteLLM_Params(model="dashscope/qwen3-max", api_key="sk-fake"),
        model_info=ModelInfo(
            id="tiered-deployment",
            max_output_tokens=1000,
            tiered_pricing=[
                {
                    "input_cost_per_token": input_cost_per_token,
                    "output_cost_per_token": input_cost_per_token,
                    "range": [0, 128000],
                }
            ],
        ),
    )


TIERED_BODY: Final = {"model": "tiered-group", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 10}


def test_repeated_estimates_reuse_cached_model_cost_info() -> None:
    router: Final = Router(model_list=[_tiered_deployment(1e-06).model_dump()])
    first: Final = estimate_request_max_cost(request_body=TIERED_BODY, route="/chat/completions", llm_router=router)
    hits_before: Final = router.cached_deployment_model_info.cache_info().hits

    second: Final = estimate_request_max_cost(request_body=TIERED_BODY, route="/chat/completions", llm_router=router)

    assert second == first
    assert router.cached_deployment_model_info.cache_info().hits == hits_before + 1


def test_deployment_pricing_update_invalidates_cached_estimate() -> None:
    router: Final = Router(model_list=[_tiered_deployment(1e-06).model_dump()])
    before: Final = estimate_request_max_cost(request_body=TIERED_BODY, route="/chat/completions", llm_router=router)
    assert before is not None

    router.upsert_deployment(_tiered_deployment(1e-03))

    after: Final = estimate_request_max_cost(request_body=TIERED_BODY, route="/chat/completions", llm_router=router)
    assert after is not None
    assert math.isclose(after, before * 1000)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expiry_offset, expected_max_budget",
    [
        (timedelta(days=1), 3.0),
        (timedelta(days=-1), 2.0),
    ],
)
async def test_team_member_reservation_counter_honours_temp_budget_increase(
    expiry_offset: timedelta, expected_max_budget: float
) -> None:
    user_id: Final = "member-temp"
    team_id: Final = "team-temp"
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(
        key=team_membership_reservation_cache_key(user_id=user_id, team_id=team_id),
        value=LiteLLM_TeamMembership(
            user_id=user_id,
            team_id=team_id,
            spend=0.5,
            budget_id="budget-temp",
            litellm_budget_table=LiteLLM_BudgetTable(
                max_budget=2.0,
                temp_budget_increase=1.0,
                temp_budget_expiry=datetime.now(timezone.utc) + expiry_offset,
            ),
        ),
    )

    counter: Final = await _get_team_member_budget_counter(
        valid_token=UserAPIKeyAuth(token="hashed", user_id=user_id, team_id=team_id),
        team_object=LiteLLM_TeamTable(team_id=team_id),
        user_object=LiteLLM_UserTable(user_id=user_id),
        user_api_key_cache=cache,
    )

    assert counter is not None
    assert counter.max_budget == expected_max_budget
    assert counter.fallback_spend == 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "default_cap, expiry_offset, expected_max_budget",
    [
        (2.0, timedelta(days=1), 3.0),
        (2.0, timedelta(days=-1), 2.0),
        (0.0, timedelta(days=1), None),
    ],
)
async def test_team_member_reservation_counter_adds_temp_increase_to_live_team_default(
    default_cap: float, expiry_offset: timedelta, expected_max_budget: float | None
) -> None:
    user_id: Final = "member-bare"
    team_id: Final = "team-bare"
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(
        key="team_member_default_budget:default-bare",
        value=LiteLLM_BudgetTable(budget_id="default-bare", max_budget=default_cap),
    )
    await cache.async_set_cache(
        key=team_membership_reservation_cache_key(user_id=user_id, team_id=team_id),
        value=LiteLLM_TeamMembership(
            user_id=user_id,
            team_id=team_id,
            spend=0.5,
            budget_id="budget-bare",
            litellm_budget_table=LiteLLM_BudgetTable(
                max_budget=None,
                temp_budget_increase=1.0,
                temp_budget_expiry=datetime.now(timezone.utc) + expiry_offset,
            ),
        ),
    )

    counter: Final = await _get_team_member_budget_counter(
        valid_token=UserAPIKeyAuth(token="hashed", user_id=user_id, team_id=team_id),
        team_object=LiteLLM_TeamTable(team_id=team_id, metadata={"team_member_budget_id": "default-bare"}),
        user_object=LiteLLM_UserTable(user_id=user_id),
        user_api_key_cache=cache,
    )

    if expected_max_budget is None:
        assert counter is None
        return
    assert counter is not None
    assert counter.max_budget == expected_max_budget
    assert counter.fallback_spend == 0.5


@pytest.mark.asyncio
async def test_reservation_starts_unbound_to_any_callback():
    reservation: Final = await _reserve("/v1/responses")

    assert reservation is not None
    assert reservation["callback_bound"] is False


@pytest.mark.asyncio
async def test_release_unbound_budget_reservation_frees_the_counter(spend_counter_cache: DualCache):
    counter_key: Final = f"spend:key:{TINY_BUDGET_KEY_TOKEN}"
    reservation: Final = await _reserve_for_tiny_budget_key(
        "/v1/chat/completions", {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
    )
    assert reservation is not None
    assert spend_counter_cache.in_memory_cache.get_cache(key=counter_key) == pytest.approx(reservation["reserved_cost"])

    await release_unbound_budget_reservation(reservation)

    assert spend_counter_cache.in_memory_cache.get_cache(key=counter_key) == pytest.approx(0.0)
    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_release_unbound_budget_reservation_leaves_a_bound_one_to_its_callback(spend_counter_cache: DualCache):
    counter_key: Final = f"spend:key:{TINY_BUDGET_KEY_TOKEN}"
    reservation: Final = await _reserve_for_tiny_budget_key(
        "/v1/chat/completions", {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
    )
    assert reservation is not None
    reservation["callback_bound"] = True

    await release_unbound_budget_reservation(reservation)

    assert spend_counter_cache.in_memory_cache.get_cache(key=counter_key) == pytest.approx(reservation["reserved_cost"])
    assert reservation["finalized"] is False


@pytest.mark.asyncio
async def test_reservation_holds_cost_against_cached_org_and_project_budgets(spend_counter_cache: DualCache):
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(
        key="org_id:org-budgeted:with_budget",
        value=LiteLLM_OrganizationTable(
            organization_id="org-budgeted",
            budget_id="org-budget",
            spend=1.5,
            models=[],
            created_by="admin",
            updated_by="admin",
            litellm_budget_table=LiteLLM_BudgetTable(max_budget=10.0),
        ),
        model_type=LiteLLM_OrganizationTable,
    )
    await cache.async_set_cache(
        key=project_cache_key("project-budgeted"),
        value=LiteLLM_ProjectTableCachedObj(
            project_id="project-budgeted", spend=2.5, litellm_budget_table=LiteLLM_BudgetTable(max_budget=20.0)
        ),
        model_type=LiteLLM_ProjectTableCachedObj,
    )

    reservation: Final = await reserve_budget_for_request(
        request_body={"model": "gpt-4o", "input": "hello"},
        route="/v1/responses",
        llm_router=None,
        valid_token=UserAPIKeyAuth(token="hashed-org-project", org_id="org-budgeted", project_id="project-budgeted"),
        team_object=None,
        user_object=None,
        prisma_client=None,
        user_api_key_cache=cache,
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
    )

    assert reservation is not None
    reserved_cost: Final = reservation["reserved_cost"]
    assert reserved_cost > 0
    assert reservation["entries"] == [
        {
            "counter_key": "spend:org:org-budgeted",
            "entity_type": "Organization",
            "entity_id": "org-budgeted",
            "reserved_cost": reserved_cost,
            "applied_adjustment": 0.0,
        },
        {
            "counter_key": "spend:project:project-budgeted",
            "entity_type": "Project",
            "entity_id": "project-budgeted",
            "reserved_cost": reserved_cost,
            "applied_adjustment": 0.0,
        },
    ]
    assert spend_counter_cache.in_memory_cache.get_cache(key="spend:org:org-budgeted") == pytest.approx(reserved_cost)
    assert spend_counter_cache.in_memory_cache.get_cache(key="spend:project:project-budgeted") == pytest.approx(
        reserved_cost
    )


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        (
            '{"budget_duration": "1h", "reset_at": "2030-01-01T01:00:00Z"}',
            datetime(2030, 1, 1, 0, 0, tzinfo=timezone.utc),
        ),
        ('{"reset_at": "2030-01-01T01:00:00Z"}', None),
        ("{}", None),
        ('["budget_duration", "1h"]', None),
        ('"1h"', None),
        ("null", None),
        ("not json", None),
    ],
)
def test_budget_window_start_reads_json_encoded_windows(window: str, expected: datetime | None) -> None:
    assert get_budget_window_start(window) == expected


@pytest.mark.asyncio
async def test_reservation_counts_a_pdf_with_the_deployment_behind_a_gpt_named_group() -> None:
    from litellm.proxy.spend_tracking.input_tokens import count_input_tokens_for_model
    from tests.unit.proxy.spend_tracking.test_input_tokens import _one_page_pdf_message

    request_body: Final = {"model": "gpt-4o", "messages": _one_page_pdf_message()}
    router: Final = Router(
        model_list=[{"model_name": "gpt-4o", "litellm_params": {"model": "anthropic/claude-haiku-5-5", "api_key": "x"}}]
    )

    counts: Final = await count_request_input_tokens(
        request_body=request_body, route="/chat/completions", llm_router=router, team_id=None
    )

    assert counts == {"gpt-4o": count_input_tokens_for_model(request_body=request_body, model="claude-haiku-5-5")}
    assert counts["gpt-4o"] > count_input_tokens_for_model(request_body=request_body, model="gpt-4o")


_MULTILINGUAL_MESSAGES: Final = [{"role": "user", "content": "日本語のテキストとहिन्दी पाठ और русский текст. " * 20}]


def _router_with_groups(*groups: tuple[str, str]) -> Router:
    return Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": model, "api_key": "sk-test"}} for name, model in groups
        ]
    )


async def _input_tokens(router: Router, model: str) -> dict[str, int]:
    return dict(
        await count_request_input_tokens(
            request_body={"model": model, "messages": _MULTILINGUAL_MESSAGES},
            route="/v1/chat/completions",
            llm_router=router,
            team_id=None,
        )
    )


async def _reserved_input_tokens(router: Router, model: str, route: str, messages: list[dict[str, object]]) -> int:
    counts: Final = await count_request_input_tokens(
        request_body={"model": model, "messages": messages}, route=route, llm_router=router, team_id=None
    )
    return counts[model]


@pytest.mark.asyncio
async def test_an_openai_chat_request_reserves_what_openai_chat_bills_on_top_of_each_pdf_files_pages() -> None:
    from litellm.constants import OPENAI_CHAT_PDF_FILE_TOKENS
    from tests.unit.proxy.spend_tracking.test_input_tokens import _one_page_pdf_message

    router: Final = _router_with_groups(("on-openai", "openai/gpt-5.6-sol"), ("on-azure", "azure/gpt-5.6-sol"))
    two_pdfs: Final = [*_one_page_pdf_message(), *_one_page_pdf_message()]

    assert await _reserved_input_tokens(router, "on-openai", "/v1/chat/completions", two_pdfs) == (
        await _reserved_input_tokens(router, "on-azure", "/v1/chat/completions", two_pdfs)
        + 2 * OPENAI_CHAT_PDF_FILE_TOKENS
    )
    assert await _reserved_input_tokens(router, "on-openai", "/v1/messages", two_pdfs) == (
        await _reserved_input_tokens(router, "on-azure", "/v1/messages", two_pdfs)
    )
    assert await _reserved_input_tokens(router, "on-openai", "/v1/chat/completions", _MULTILINGUAL_MESSAGES) == (
        await _reserved_input_tokens(router, "on-azure", "/v1/chat/completions", _MULTILINGUAL_MESSAGES)
    )


@pytest.mark.asyncio
async def test_alias_group_of_an_openai_model_reserves_on_that_models_token_count():
    router: Final = _router_with_groups(("gpt-6.1-sol", "openai/gpt-6.1-sol"), ("prod-gpt-6", "openai/gpt-6.1-sol"))

    by_name: Final = await _input_tokens(router, "gpt-6.1-sol")

    assert await _input_tokens(router, "prod-gpt-6") == {"prod-gpt-6": by_name["gpt-6.1-sol"]}


@pytest.mark.asyncio
async def test_model_group_alias_of_an_openai_group_reserves_on_that_groups_token_count():
    router: Final = Router(
        model_list=[
            {"model_name": "gpt-6.1-sol", "litellm_params": {"model": "openai/gpt-6.1-sol", "api_key": "sk-test"}}
        ],
        model_group_alias={"prod": "gpt-6.1-sol"},
    )

    by_name: Final = await _input_tokens(router, "gpt-6.1-sol")

    assert await _input_tokens(router, "prod") == {"prod": by_name["gpt-6.1-sol"]}


@pytest.mark.asyncio
async def test_model_group_alias_of_a_group_no_openai_model_serves_reserves_on_the_alias_as_written():
    router: Final = Router(
        model_list=[
            {
                "model_name": "claude-sonnet-5-5",
                "litellm_params": {"model": "anthropic/claude-sonnet-5-5", "api_key": "sk-test"},
            }
        ],
        model_group_alias={"claude-alias": "claude-sonnet-5-5"},
    )

    assert await _input_tokens(router, "claude-alias") == {
        "claude-alias": litellm.token_counter(model="claude-alias", messages=_MULTILINGUAL_MESSAGES)
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["gpt-6-claude", "o4-alias"])
async def test_release_named_group_or_alias_no_openai_model_serves_reserves_on_an_unknown_names_count(
    name: str,
) -> None:
    router: Final = Router(
        model_list=[
            {"model_name": group, "litellm_params": {"model": "anthropic/claude-sonnet-5-5", "api_key": "sk-test"}}
            for group in ("claude-sonnet-5-5", "gpt-6-claude")
        ],
        model_group_alias={"o4-alias": "claude-sonnet-5-5"},
    )

    assert await _input_tokens(router, name) == {
        name: litellm.token_counter(model="my-model", messages=_MULTILINGUAL_MESSAGES)
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("group", "counted_as"),
    [("gpt-6.1-sol", "my-model"), ("openai/gpt-6.1-sol", "my-model"), ("mixed", "my-model"), ("gpt-4o", "gpt-4o")],
)
async def test_group_mixing_in_a_non_openai_model_reserves_on_its_name_as_written_and_its_openai_models(
    group: str, counted_as: str
):
    served: Final = ("openai/gpt-6.1-sol", "cohere/command-r-plus")
    router: Final = _router_with_groups(*((group, model) for model in served))

    reserved: Final = (await _input_tokens(router, group))[group]

    assert reserved == max(
        litellm.token_counter(model=counted_as, messages=_MULTILINGUAL_MESSAGES),
        litellm.token_counter(model="openai/gpt-6.1-sol", messages=_MULTILINGUAL_MESSAGES),
    )


@pytest.mark.asyncio
async def test_group_reserves_on_its_current_deployments_after_another_vendors_deployment_joins():
    router: Final = _router_with_groups(("prod", "openai/gpt-6.1-sol"))
    openai_only: Final = (await _input_tokens(router, "prod"))["prod"]

    router.add_deployment(
        Deployment(
            model_name="prod",
            litellm_params=LiteLLM_Params(model="anthropic/claude-sonnet-5-5", api_key="sk-test"),
        )
    )

    by_own_name: Final = litellm.token_counter(model="prod", messages=_MULTILINGUAL_MESSAGES)
    assert openai_only != by_own_name
    assert (await _input_tokens(router, "prod"))["prod"] == by_own_name
