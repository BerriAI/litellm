from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.bases.openai import GPT_5_6_SOL_TEST_CASE as GPT_5_6_SOL_CHAT_TEST_CASE
from integration.translation.chat_completions.bases.openai_responses import GPT_5_6_SOL_TEST_CASE
from integration.translation.runner import assert_translation

GPT_5_6_SOL_WEB_SEARCH_TEST_CASE: Final = replace(
    GPT_5_6_SOL_TEST_CASE,
    scenario="web_search_options",
    litellm_request={**GPT_5_6_SOL_TEST_CASE.litellm_request, "web_search_options": {"search_context_size": "low"}},
    expected_provider_request={
        **GPT_5_6_SOL_TEST_CASE.expected_provider_request,
        "tools": [{"type": "web_search", "search_context_size": "low"}],
    },
)

GPT_5_6_SOL_WEB_SEARCH_USER_LOCATION_TEST_CASE: Final = replace(
    GPT_5_6_SOL_TEST_CASE,
    scenario="web_search_options_user_location",
    litellm_request={
        **GPT_5_6_SOL_TEST_CASE.litellm_request,
        "web_search_options": {
            "search_context_size": "medium",
            "user_location": {"type": "approximate", "approximate": {"country": "US", "city": "San Francisco"}},
        },
    },
    expected_provider_request={
        **GPT_5_6_SOL_TEST_CASE.expected_provider_request,
        "tools": [
            {
                "type": "web_search",
                "search_context_size": "medium",
                "user_location": {"type": "approximate", "country": "US", "city": "San Francisco"},
            }
        ],
    },
)

# No `responses/` prefix: a gpt-5.6 chat request with function tools and a reasoning effort is bridged by the
# tools-plus-reasoning rule rather than the deployment's mode, and web_search_options must survive that too
GPT_5_6_SOL_LATE_BRIDGE_WEB_SEARCH_TEST_CASE: Final = replace(
    GPT_5_6_SOL_TEST_CASE,
    scenario="late_bridge_web_search_options",
    litellm_request={
        **GPT_5_6_SOL_TEST_CASE.litellm_request,
        "model": "openai/gpt-5.6-sol",
        "reasoning_effort": "low",
        "web_search_options": {"search_context_size": "low"},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                },
            }
        ],
    },
    expected_provider_request={
        **GPT_5_6_SOL_TEST_CASE.expected_provider_request,
        "tools": [
            {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                "strict": None,
                "type": "function",
                "description": "Get weather",
            },
            {"type": "web_search", "search_context_size": "low"},
        ],
        "reasoning": {"effort": "low"},
    },
    expected_litellm_response={**GPT_5_6_SOL_TEST_CASE.expected_litellm_response, "model": "openai/gpt-5.6-sol"},
)

# The rig's deployment has a custom api_base, which the bridge treats as a chat-only endpoint unless the
# request names an effort. An effort the caller drops through additional_drop_params must not pick the
# Responses route, so this request stays on /chat/completions exactly as before the early route decision
GPT_5_6_SOL_DROPPED_EFFORT_STAYS_ON_CHAT_TEST_CASE: Final = replace(
    GPT_5_6_SOL_CHAT_TEST_CASE,
    scenario="dropped_extra_body_effort_stays_on_chat",
    litellm_request={
        **GPT_5_6_SOL_CHAT_TEST_CASE.litellm_request,
        "extra_body": {"reasoning_effort": "low"},
        "additional_drop_params": ["reasoning_effort"],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                },
            }
        ],
    },
    expected_provider_request={
        **GPT_5_6_SOL_CHAT_TEST_CASE.expected_provider_request,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                },
            }
        ],
    },
)

CASES: Final = (
    GPT_5_6_SOL_WEB_SEARCH_TEST_CASE,
    GPT_5_6_SOL_WEB_SEARCH_USER_LOCATION_TEST_CASE,
    GPT_5_6_SOL_LATE_BRIDGE_WEB_SEARCH_TEST_CASE,
    GPT_5_6_SOL_DROPPED_EFFORT_STAYS_ON_CHAT_TEST_CASE,
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_chat_completions_params_openai_responses_web_search(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
