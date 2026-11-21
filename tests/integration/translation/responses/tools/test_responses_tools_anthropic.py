from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import RejectionTestCase, TranslationTestCase
from integration.translation.responses.bases.anthropic import CLAUDE_HAIKU_4_5_TEST_CASE, CLAUDE_HAIKU_5_5_TEST_CASE
from integration.translation.runner import assert_rejection, assert_translation
from pydantic import JsonValue

REQUEST_TOOLS: Final = CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request["tools"]
PROVIDER_TOOLS: Final = CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_request["tools"]
ECHOED_TOOLS: Final = CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response["tools"]
assert isinstance(REQUEST_TOOLS, list)
assert isinstance(PROVIDER_TOOLS, list)
assert isinstance(ECHOED_TOOLS, list)

CODE_INTERPRETER_TOOL: Final[dict[str, JsonValue]] = {"type": "code_interpreter", "container": {"type": "auto"}}
ALLOWED_TOOLS_CHOICE: Final[dict[str, JsonValue]] = {
    "type": "allowed_tools",
    "mode": "required",
    "tools": [{"type": "function", "name": "read_file"}],
}

CLAUDE_HAIKU_5_5_WEB_SEARCH_TOOL_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="web_search_tool",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "tools": [*REQUEST_TOOLS, {"type": "web_search"}]},
    expected_provider_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_request,
        "tools": [*PROVIDER_TOOLS, {"type": "web_search_20250305", "name": "web_search"}],
    },
    expected_litellm_response={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response,
        "tools": [
            *ECHOED_TOOLS,
            {"type": "web_search", "filters": None, "search_context_size": None, "user_location": None},
        ],
    },
)

CLAUDE_HAIKU_5_5_TOOL_CHOICE_AUTO_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="tool_choice_auto",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "tool_choice": "auto"},
    expected_provider_request={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_request, "tool_choice": {"type": "auto"}},
)

CLAUDE_HAIKU_5_5_TOOL_CHOICE_REQUIRED_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="tool_choice_required",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "tool_choice": "required"},
    expected_provider_request={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_request, "tool_choice": {"type": "any"}},
    expected_litellm_response={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response, "tool_choice": "required"},
)

CLAUDE_HAIKU_5_5_TOOL_CHOICE_FUNCTION_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="tool_choice_function",
    litellm_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request,
        "tool_choice": {"type": "function", "name": "read_file"},
    },
    expected_provider_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_request,
        "tool_choice": {"type": "tool", "name": "read_file"},
    },
    expected_litellm_response={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response,
        "tool_choice": {"type": "function", "name": "read_file"},
    },
)

CLAUDE_HAIKU_5_5_PARALLEL_TOOL_CALLS_FALSE_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="parallel_tool_calls_false",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "parallel_tool_calls": False},
    expected_provider_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_request,
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
    },
)

CLAUDE_HAIKU_5_5_PARALLEL_TOOL_CALLS_TRUE_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="parallel_tool_calls_true",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "parallel_tool_calls": True},
    expected_provider_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_request,
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": False},
    },
    expected_litellm_response={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response, "parallel_tool_calls": True},
)

CLAUDE_HAIKU_4_5_PARALLEL_TOOL_CALLS_WITHOUT_TOOLS_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_4_5_TEST_CASE,
    scenario="parallel_tool_calls_without_tools",
    litellm_request={**CLAUDE_HAIKU_4_5_TEST_CASE.litellm_request, "parallel_tool_calls": False},
)

CLAUDE_HAIKU_5_5_MAX_TOOL_CALLS_DROP_PARAMS_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="max_tool_calls_drop_params",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "max_tool_calls": 3, "drop_params": True},
)

CLAUDE_HAIKU_5_5_MAX_TOOL_CALLS_TEST_CASE: Final = RejectionTestCase(
    scenario="max_tool_calls",
    litellm_endpoint="/v1/responses",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "max_tool_calls": 3},
    expected_litellm_status_code=400,
    expected_litellm_response={
        "error": {
            "message": "litellm.UnsupportedParamsError: anthropic does not support parameters: ['max_tool_calls'], "
            "for model=claude-haiku-5-5. The Responses API to chat completions bridge cannot translate them. To "
            "drop these, set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n. "
            "\n If you want to use these params dynamically send allowed_openai_params=['max_tool_calls'] in your "
            "request.\n\nLiteLLM: model group 'anthropic/claude-haiku-5-5' failed with the error above. No fallback "
            "was attempted.",
            "type": "invalid_request_error",
            "param": None,
            "code": "400",
        }
    },
)

CLAUDE_HAIKU_5_5_TOOL_CHOICE_HOSTED_TOOL_TEST_CASE: Final = RejectionTestCase(
    scenario="tool_choice_hosted_tool",
    litellm_endpoint="/v1/responses",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "tool_choice": {"type": "code_interpreter"}},
    expected_litellm_status_code=400,
    expected_litellm_response={
        "error": {
            "message": "litellm.BadRequestError: tool_choice type 'code_interpreter' forces an OpenAI hosted tool "
            "and is only supported for azure, litellm_proxy, openai, got custom_llm_provider=anthropic. drop_params "
            "does not drop it, since that would let the model answer without the tool the request required.\n\n"
            "LiteLLM: model group 'anthropic/claude-haiku-5-5' failed with the error above. No fallback was "
            "attempted.",
            "type": "invalid_request_error",
            "param": None,
            "code": "400",
        }
    },
)

CLAUDE_HAIKU_5_5_CODE_INTERPRETER_TOOL_TEST_CASE: Final = RejectionTestCase(
    scenario="code_interpreter_tool",
    litellm_endpoint="/v1/responses",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "tools": [*REQUEST_TOOLS, CODE_INTERPRETER_TOOL]},
    expected_litellm_status_code=400,
    expected_litellm_response={
        "error": {
            "message": "litellm.UnsupportedParamsError: anthropic does not support parameters: ['tools'], "
            "for model=claude-haiku-5-5. The Responses API to chat completions bridge cannot translate them. To "
            "drop these, set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n. "
            "\n If you want to use these params dynamically send allowed_openai_params=['tools'] in your "
            "request.\n\nLiteLLM: model group 'anthropic/claude-haiku-5-5' failed with the error above. No fallback "
            "was attempted.",
            "type": "invalid_request_error",
            "param": None,
            "code": "400",
        }
    },
)

CLAUDE_HAIKU_5_5_CODE_INTERPRETER_TOOL_DROP_PARAMS_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="code_interpreter_tool_drop_params",
    litellm_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request,
        "tools": [*REQUEST_TOOLS, CODE_INTERPRETER_TOOL],
        "drop_params": True,
    },
    expected_litellm_response={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response,
        "tools": [
            *ECHOED_TOOLS,
            {
                "type": "code_interpreter",
                "container": {"type": "auto", "file_ids": None, "memory_limit": None, "network_policy": None},
            },
        ],
    },
)

CLAUDE_HAIKU_5_5_IMAGE_GENERATION_TOOL_TEST_CASE: Final = RejectionTestCase(
    scenario="image_generation_tool",
    litellm_endpoint="/v1/responses",
    litellm_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request,
        "tools": [*REQUEST_TOOLS, {"type": "image_generation"}],
    },
    expected_litellm_status_code=400,
    expected_litellm_response={
        "error": {
            "message": "litellm.UnsupportedParamsError: anthropic does not support parameters: ['tools'], "
            "for model=claude-haiku-5-5. The Responses API to chat completions bridge cannot translate them. To "
            "drop these, set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n. "
            "\n If you want to use these params dynamically send allowed_openai_params=['tools'] in your "
            "request.\n\nLiteLLM: model group 'anthropic/claude-haiku-5-5' failed with the error above. No fallback "
            "was attempted.",
            "type": "invalid_request_error",
            "param": None,
            "code": "400",
        }
    },
)

CLAUDE_HAIKU_5_5_IMAGE_GENERATION_TOOL_DROP_PARAMS_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="image_generation_tool_drop_params",
    litellm_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request,
        "tools": [*REQUEST_TOOLS, {"type": "image_generation"}],
        "drop_params": True,
    },
    expected_litellm_response={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response,
        "tools": [
            *ECHOED_TOOLS,
            {
                "type": "image_generation",
                "action": None,
                "background": None,
                "input_fidelity": None,
                "input_image_mask": None,
                "model": None,
                "moderation": None,
                "output_compression": None,
                "output_format": None,
                "partial_images": None,
                "quality": None,
                "size": None,
            },
        ],
    },
)

CLAUDE_HAIKU_5_5_PARALLEL_TOOL_CALLS_OMITTED_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="parallel_tool_calls_omitted",
    expected_litellm_response={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response, "parallel_tool_calls": True},
)

CLAUDE_HAIKU_5_5_FILE_SEARCH_TOOL_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="file_search_tool",
    litellm_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request,
        "tools": [*REQUEST_TOOLS, {"type": "file_search", "vector_store_ids": ["vs_synthetic_01"]}],
    },
    expected_provider_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_request,
        "tools": [
            *PROVIDER_TOOLS,
            {
                "name": "litellm_file_search",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "One or more search queries to look up in the vector store. Multiple "
                            "queries help find comprehensive information from different angles.",
                        },
                        "vector_store_id": {
                            "type": "string",
                            "description": "ID of the vector store to search.",
                            "enum": ["vs_synthetic_01"],
                        },
                    },
                    "required": ["queries"],
                },
                "type": "custom",
                "description": "Search the knowledge base for information relevant to the query. Use this whenever "
                "you need to look up specific facts, documents, or content from the vector store. You can provide "
                "multiple queries to search for different aspects of the information.",
            },
        ],
    },
    expected_litellm_response={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response,
        "tools": [
            *ECHOED_TOOLS,
            {
                "type": "file_search",
                "vector_store_ids": ["vs_synthetic_01"],
                "filters": None,
                "max_num_results": None,
                "ranking_options": None,
            },
        ],
    },
)

CLAUDE_HAIKU_5_5_TOOL_CHOICE_ALLOWED_TOOLS_TEST_CASE: Final = RejectionTestCase(
    scenario="tool_choice_allowed_tools",
    litellm_endpoint="/v1/responses",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "tool_choice": ALLOWED_TOOLS_CHOICE},
    expected_litellm_status_code=400,
    expected_litellm_response={
        "error": {
            "message": "litellm.BadRequestError: tool_choice type 'allowed_tools' is only supported for azure, "
            "litellm_proxy, openai, got custom_llm_provider=anthropic. drop_params does not drop it, since that "
            "would let the model call tools the request excluded. Send only the allowed tools in `tools` with "
            "tool_choice='auto' or 'required' instead.\n\nLiteLLM: model group 'anthropic/claude-haiku-5-5' failed "
            "with the error above. No fallback was attempted.",
            "type": "invalid_request_error",
            "param": None,
            "code": "400",
        }
    },
)

CLAUDE_HAIKU_5_5_TOOL_CHOICE_HOSTED_TOOL_DROP_PARAMS_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="tool_choice_hosted_tool_drop_params",
    litellm_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request,
        "tool_choice": {"type": "code_interpreter"},
        "drop_params": True,
    },
    expected_provider_request={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_request, "tool_choice": {"type": "auto"}},
    expected_litellm_response={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response,
        "tool_choice": {"type": "code_interpreter"},
    },
)


@pytest.mark.parametrize(
    "case",
    [
        CLAUDE_HAIKU_5_5_WEB_SEARCH_TOOL_TEST_CASE,
        CLAUDE_HAIKU_5_5_TOOL_CHOICE_AUTO_TEST_CASE,
        CLAUDE_HAIKU_5_5_TOOL_CHOICE_REQUIRED_TEST_CASE,
        CLAUDE_HAIKU_5_5_TOOL_CHOICE_FUNCTION_TEST_CASE,
        CLAUDE_HAIKU_5_5_PARALLEL_TOOL_CALLS_FALSE_TEST_CASE,
        CLAUDE_HAIKU_5_5_PARALLEL_TOOL_CALLS_TRUE_TEST_CASE,
        CLAUDE_HAIKU_4_5_PARALLEL_TOOL_CALLS_WITHOUT_TOOLS_TEST_CASE,
        CLAUDE_HAIKU_5_5_MAX_TOOL_CALLS_DROP_PARAMS_TEST_CASE,
        CLAUDE_HAIKU_5_5_IMAGE_GENERATION_TOOL_DROP_PARAMS_TEST_CASE,
    ],
    ids=lambda case: case.id,
)
def test_responses_tools_anthropic(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)


@pytest.mark.parametrize(
    "case",
    [CLAUDE_HAIKU_5_5_MAX_TOOL_CALLS_TEST_CASE, CLAUDE_HAIKU_5_5_TOOL_CHOICE_HOSTED_TOOL_TEST_CASE],
    ids=lambda case: case.id,
)
def test_responses_tools_anthropic_rejection(
    case: RejectionTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_rejection(case, gateway, provider)


def test_responses_code_interpreter_tool_anthropic(gateway: Gateway, provider: SharedProvider) -> None:
    pytest.skip(
        "BUG: a code_interpreter tool on an Anthropic model fails with a 500 Unsupported tool type instead of a 400 that names drop_params"
    )
    assert_rejection(CLAUDE_HAIKU_5_5_CODE_INTERPRETER_TOOL_TEST_CASE, gateway, provider)


def test_responses_code_interpreter_tool_drop_params_anthropic(gateway: Gateway, provider: SharedProvider) -> None:
    pytest.skip(
        "BUG: drop_params does not drop a code_interpreter tool on an Anthropic model, the request still fails with a 500"
    )
    assert_translation(CLAUDE_HAIKU_5_5_CODE_INTERPRETER_TOOL_DROP_PARAMS_TEST_CASE, gateway, provider)


def test_responses_image_generation_tool_anthropic(gateway: Gateway, provider: SharedProvider) -> None:
    pytest.skip(
        "BUG: an image_generation tool on an Anthropic model is silently dropped instead of a 400 that names drop_params"
    )
    assert_rejection(CLAUDE_HAIKU_5_5_IMAGE_GENERATION_TOOL_TEST_CASE, gateway, provider)


def test_responses_parallel_tool_calls_omitted_anthropic(gateway: Gateway, provider: SharedProvider) -> None:
    pytest.skip(
        "BUG: with parallel_tool_calls omitted Anthropic may call tools in parallel, but the response echoes parallel_tool_calls false"
    )
    assert_translation(CLAUDE_HAIKU_5_5_PARALLEL_TOOL_CALLS_OMITTED_TEST_CASE, gateway, provider)


def test_responses_file_search_tool_anthropic(gateway: Gateway, provider: SharedProvider) -> None:
    pytest.skip(
        "BUG: with a file_search tool the response replaces the model's function call with a fake file_search call and blanks every echoed request field"
    )
    assert_translation(CLAUDE_HAIKU_5_5_FILE_SEARCH_TOOL_TEST_CASE, gateway, provider)


def test_responses_tool_choice_allowed_tools_anthropic(gateway: Gateway, provider: SharedProvider) -> None:
    pytest.skip(
        "BUG: an allowed_tools tool_choice reaches chat completions in the Responses layout and fails as malformed instead of with the allowed_tools support error"
    )
    assert_rejection(CLAUDE_HAIKU_5_5_TOOL_CHOICE_ALLOWED_TOOLS_TEST_CASE, gateway, provider)


def test_responses_tool_choice_hosted_tool_drop_params_anthropic(gateway: Gateway, provider: SharedProvider) -> None:
    pytest.skip(
        "BUG: drop_params does not downgrade a hosted tool tool_choice to auto on an Anthropic model, the request still fails with a 400"
    )
    assert_translation(CLAUDE_HAIKU_5_5_TOOL_CHOICE_HOSTED_TOOL_DROP_PARAMS_TEST_CASE, gateway, provider)
