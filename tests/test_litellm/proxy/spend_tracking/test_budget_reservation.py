import base64
from typing import Final

import pytest

from litellm.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.spend_tracking.budget_reservation import estimate_request_max_cost, reserve_budget_for_request
from litellm.proxy.utils import ProxyLogging

TOKEN_COUNTING_ROUTES: Final = (
    "/responses/input_tokens",
    "/v1/responses/input_tokens",
    "/openai/v1/responses/input_tokens",
    "/utils/token_counter",
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


def test_input_audio_requests_reserve_at_least_the_serialised_fallback():
    """
    Budget reservation counts an ``input_audio`` block as a size-derived
    estimate at a deliberately low bitrate, priced at the text rate. Before
    #38459 the same request raised inside ``token_counter`` and reserved
    from the serialised-messages fallback, which tokenises the base64
    payload itself. Floor audio-bearing requests at that fallback so a
    caller cannot be admitted against a budget more cheaply than before
    the blocks became countable (compressed audio carries far more duration
    per byte than the estimate assumes).
    """
    from litellm.proxy.spend_tracking.budget_reservation import _count_input_tokens, _count_text_tokens

    model: Final = "gpt-4o-audio-preview"
    audio_b64: Final = base64.b64encode(bytes(range(256)) * 400).decode()
    audio_messages: Final = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this recording."},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
            ],
        }
    ]
    text_messages: Final = [{"role": "user", "content": "Transcribe this recording."}]

    audio_count: Final = _count_input_tokens(request_body={"messages": audio_messages}, model=model)
    fallback: Final = _count_text_tokens(model=model, text=audio_messages)
    text_count: Final = _count_input_tokens(request_body={"messages": text_messages}, model=model)

    assert audio_count is not None, "an audio-bearing request must still be countable"
    assert fallback > 0, "the serialised fallback must see the base64 payload"
    assert audio_count >= fallback, (
        f"audio request reserved {audio_count} tokens, below the pre-#38459 fallback of {fallback}"
    )
    assert text_count is not None and text_count < fallback, "a text-only request must not be floored"
