from __future__ import annotations

from typing import Final

import pytest

from tests.sdk_function_trace.profiler import FunctionTraceEvent
from tests.sdk_function_trace.runtime import trace_diff
from tests.sdk_function_trace.steps import pipeline_issues, pipeline_steps


def test_python_ocr_projection_keeps_pipeline_and_drops_noise() -> None:
    events: Final = (
        FunctionTraceEvent("utils.py:1747 client.<locals>.wrapper_async", 0),
        FunctionTraceEvent("ocr/main.py:331 aocr", 1),
        FunctionTraceEvent("ocr/main.py:70 _prepare_ocr_request", 2),
        FunctionTraceEvent("litellm_core_utils/get_llm_provider_logic.py:142 get_llm_provider", 3),
        FunctionTraceEvent("utils.py:9303 ProviderConfigManager.get_provider_ocr_config", 3),
        FunctionTraceEvent("llms/mistral/ocr/transformation.py:34 MistralOCRConfig.get_supported_ocr_params", 4),
        FunctionTraceEvent("llms/mistral/ocr/transformation.py:72 MistralOCRConfig.map_ocr_params", 4),
        FunctionTraceEvent("llms/mistral/ocr/transformation.py:34 MistralOCRConfig.get_supported_ocr_params", 5),
        FunctionTraceEvent("llms/custom_httpx/llm_http_handler.py:1705 BaseLLMHTTPHandler.async_ocr", 2),
        FunctionTraceEvent("llms/mistral/ocr/transformation.py:94 MistralOCRConfig.validate_environment", 4),
        FunctionTraceEvent("llms/mistral/ocr/transformation.py:124 MistralOCRConfig.get_complete_url", 4),
        FunctionTraceEvent("llms/base_llm/ocr/transformation.py:209 BaseOCRConfig.async_transform_ocr_request", 5),
        FunctionTraceEvent("llms/mistral/ocr/transformation.py:149 MistralOCRConfig.transform_ocr_request", 6),
        FunctionTraceEvent("llms/custom_httpx/http_handler.py:654 AsyncHTTPHandler.post", 6),
        FunctionTraceEvent("llms/base_llm/ocr/transformation.py:255 BaseOCRConfig.async_transform_ocr_response", 4),
        FunctionTraceEvent("llms/mistral/ocr/transformation.py:200 MistralOCRConfig.transform_ocr_response", 5),
        FunctionTraceEvent("cost_calculator.py:1874 ocr_cost", 6),
    )

    assert pipeline_steps("ocr", "python", events) == (
        FunctionTraceEvent("ocr", 0),
        FunctionTraceEvent("prepare_ocr_call", 1),
        FunctionTraceEvent("get_provider_ocr_config", 2),
        FunctionTraceEvent("supported_ocr_params", 3),
        FunctionTraceEvent("map_ocr_params", 3),
        FunctionTraceEvent("execute_ocr_provider_call", 1),
        FunctionTraceEvent("validate_environment", 2),
        FunctionTraceEvent("complete_url", 2),
        FunctionTraceEvent("transform_ocr_request", 3),
        FunctionTraceEvent("http_request", 3),
        FunctionTraceEvent("transform_ocr_response", 2),
    )


def test_rust_ocr_projection_reuses_step_names_and_keeps_unknown_spans() -> None:
    events: Final = (
        FunctionTraceEvent("ocr", 0),
        FunctionTraceEvent("prepare_ocr_call", 1),
        FunctionTraceEvent("map_ocr_params", 2),
        FunctionTraceEvent("supported_ocr_params", 3),
        FunctionTraceEvent("map_ocr_params", 2),
        FunctionTraceEvent("transform_ocr_request", 2),
        FunctionTraceEvent("execute_ocr_provider_call", 2),
        FunctionTraceEvent("transform_ocr_response", 3),
        FunctionTraceEvent("new_uninstrumented_span", 3),
    )

    assert pipeline_steps("ocr", "rust", events) == (
        FunctionTraceEvent("ocr", 0),
        FunctionTraceEvent("prepare_ocr_call", 1),
        FunctionTraceEvent("map_ocr_params", 2),
        FunctionTraceEvent("supported_ocr_params", 3),
        FunctionTraceEvent("transform_ocr_request", 2),
        FunctionTraceEvent("execute_ocr_provider_call", 2),
        FunctionTraceEvent("transform_ocr_response", 3),
        FunctionTraceEvent("new_uninstrumented_span", 3),
    )


def test_projection_resets_depth_on_thread_root() -> None:
    events: Final = (
        FunctionTraceEvent("main.py:387 acompletion", 1),
        FunctionTraceEvent("llms/anthropic/chat/handler.py:255 AnthropicChatCompletion.acompletion_function", 2),
        FunctionTraceEvent(
            "llms/anthropic/experimental_pass_through/messages/handler.py:416 anthropic_messages_handler", 0
        ),
        FunctionTraceEvent(
            "llms/anthropic/experimental_pass_through/messages/transformation.py:575"
            " AnthropicMessagesConfig.transform_anthropic_messages_request",
            4,
        ),
    )

    assert pipeline_steps("chat_completions", "python", events) == (
        FunctionTraceEvent("chat_completions", 0),
        FunctionTraceEvent("execute_chat_completions_provider_call", 1),
    )
    assert pipeline_steps("messages", "python", events) == (
        FunctionTraceEvent("execute_messages_provider_call", 0),
        FunctionTraceEvent("transform_request", 1),
    )


@pytest.mark.parametrize("function", ("completion", "completion_function", "acompletion_function"))
def test_chat_projection_includes_sync_and_async_handlers(function: str) -> None:
    events: Final = (FunctionTraceEvent(f"llms/anthropic/chat/handler.py:100 AnthropicChatCompletion.{function}", 0),)

    assert pipeline_steps("chat_completions", "python", events) == (
        FunctionTraceEvent("execute_chat_completions_provider_call", 0),
    )


def test_trace_diff_reports_no_difference_for_identical_steps() -> None:
    steps: Final = (
        FunctionTraceEvent("ocr", 0),
        FunctionTraceEvent("transform_ocr_request", 1),
    )

    diff: Final = trace_diff(steps, steps)

    assert diff.python_only == ()
    assert diff.rust_only == ()
    assert diff.shared_order_matches


def test_trace_diff_reports_exclusive_steps_and_reordered_shared_steps() -> None:
    python: Final = (
        FunctionTraceEvent("ocr", 0),
        FunctionTraceEvent("supported_ocr_params", 1),
        FunctionTraceEvent("map_ocr_params", 1),
        FunctionTraceEvent("http_request", 2),
    )
    rust: Final = (
        FunctionTraceEvent("ocr", 0),
        FunctionTraceEvent("map_ocr_params", 1),
        FunctionTraceEvent("supported_ocr_params", 2),
        FunctionTraceEvent("transform_ocr_response", 2),
    )

    diff: Final = trace_diff(python, rust)

    assert diff.python_only == ("http_request",)
    assert diff.rust_only == ("transform_ocr_response",)
    assert not diff.shared_order_matches


def test_trace_diff_does_not_claim_empty_or_disjoint_traces_match() -> None:
    assert not trace_diff((), ()).shared_order_matches
    assert not trace_diff((FunctionTraceEvent("ocr", 0),), (FunctionTraceEvent("messages", 0),)).shared_order_matches


def test_projection_uses_actual_ancestors_after_coroutine_resumption() -> None:
    entrypoint: Final = "main.py:387 acompletion"
    handler: Final = "llms/anthropic/chat/handler.py:255 AnthropicChatCompletion.acompletion_function"
    events: Final = (
        FunctionTraceEvent(entrypoint, 0, ()),
        FunctionTraceEvent(handler, 1, (entrypoint,)),
        FunctionTraceEvent("utils.py:100 unrelated_worker", 0, ()),
        FunctionTraceEvent("llms/anthropic/chat/transformation.py:100 transform_response", 1, (handler,)),
    )

    assert pipeline_steps("chat_completions", "python", events) == (
        FunctionTraceEvent("chat_completions", 0),
        FunctionTraceEvent("execute_chat_completions_provider_call", 1),
        FunctionTraceEvent("transform_response", 2),
    )


def test_projection_does_not_nest_siblings_under_a_returned_config_lookup() -> None:
    events: Final = (
        FunctionTraceEvent("main.py:387 completion", 0),
        FunctionTraceEvent("utils.py:100 ProviderConfigManager.get_provider_chat_config", 1),
        FunctionTraceEvent("utils.py:200 unrelated_helper", 1),
        FunctionTraceEvent("llms/anthropic/chat/transformation.py:100 transform_request", 2),
    )

    assert pipeline_steps("chat_completions", "python", events) == (
        FunctionTraceEvent("chat_completions", 0),
        FunctionTraceEvent("get_provider_chat_config", 1),
        FunctionTraceEvent("transform_request", 1),
    )


CHAT_RUST_STEPS: Final = (
    "chat_completions",
    "prepare_chat_completions_call",
    "get_provider_chat_config",
    "transform_request",
    "execute_chat_completions_provider_call",
    "http_request",
    "transform_response",
)


@pytest.mark.parametrize("missing", CHAT_RUST_STEPS)
def test_pipeline_check_rejects_missing_stages(missing: str) -> None:
    steps: Final = tuple(FunctionTraceEvent(name, 0) for name in CHAT_RUST_STEPS if name != missing)

    assert f"missing {missing}" in pipeline_issues("chat_completions", "rust", steps)


def test_pipeline_check_rejects_http_before_request_transformation() -> None:
    steps: Final = tuple(
        FunctionTraceEvent(name, 0)
        for name in (
            "chat_completions",
            "prepare_chat_completions_call",
            "get_provider_chat_config",
            "execute_chat_completions_provider_call",
            "http_request",
            "transform_request",
            "transform_response",
        )
    )

    assert "transform_request must precede http_request" in pipeline_issues("chat_completions", "rust", steps)


def test_pipeline_check_accepts_different_handler_boundaries() -> None:
    rust: Final = tuple(FunctionTraceEvent(name, 0) for name in CHAT_RUST_STEPS)
    python: Final = tuple(
        FunctionTraceEvent(name, 0)
        for name in (
            "chat_completions",
            "get_provider_chat_config",
            "supported_openai_params",
            "execute_chat_completions_provider_call",
            "validate_environment",
            "transform_request",
            "http_request",
            "transform_response",
        )
    )

    assert not trace_diff(python, rust).shared_order_matches
    assert pipeline_issues("chat_completions", "python", python) == ()
    assert pipeline_issues("chat_completions", "rust", rust) == ()
