"""
Unit tests for prompt management support in the Responses API.

Covers:
  A) str input is coerced to a message list before merging with the template
  B) list input is merged with the template
  C) no prompt_id → hook is skipped, input is unchanged
  D) model override from the prompt template is applied
  E) prompt_template_optional_params flow into the request
  F) non-message items in input are filtered out
  G) model override re-resolves provider
  H) async path calls async_get_chat_completion_prompt
  I) async path propagates optional params to downstream handler
"""

from importlib import import_module
import asyncio
import copy
from datetime import datetime
from typing import Final, List, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from fastapi import HTTPException

import litellm
from litellm.integrations.anthropic_cache_control_hook import (
    AnthropicCacheControlHook,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
from litellm.vector_stores.vector_store_registry import VectorStoreRegistry
from litellm.types.llms.openai import (
    AllMessageValues,
    ResponseInputParam,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logging_obj(
    merged_model: str,
    merged_messages: List[AllMessageValues],
    should_run: bool = True,
    merged_optional_params: dict = None,
) -> MagicMock:
    """Return a mock LiteLLMLoggingObj pre-configured for prompt management."""
    if merged_optional_params is None:
        merged_optional_params = {}
    logging_obj = MagicMock()
    logging_obj.__class__ = LiteLLMLoggingObj
    logging_obj.should_run_prompt_management_hooks.return_value = should_run
    prompt_return = (merged_model, merged_messages, merged_optional_params)
    logging_obj.get_chat_completion_prompt.return_value = prompt_return
    logging_obj.async_get_chat_completion_prompt = AsyncMock(return_value=prompt_return)
    logging_obj.async_failure_handler = AsyncMock()
    logging_obj.model_call_details = {}
    return logging_obj


def _provider_by_model(model: str, **_: object) -> tuple[str, str, None, None]:
    provider, _, bare_model = model.partition("/")
    if not bare_model:
        return (model, "anthropic" if "claude" in model else "openai", None, None)
    return (bare_model, provider, None, None)


def _patch_responses_dispatch():
    """Patch everything after the prompt management block so tests stay unit-level."""
    return [
        patch.object(
            import_module("litellm.responses.main").litellm, "get_llm_provider",
            side_effect=_provider_by_model,
        ),
        patch.object(
            import_module("litellm.responses.main").ProviderConfigManager, "get_provider_responses_api_config",
            return_value=None,
        ),
        patch.object(
            import_module("litellm.responses.main").litellm_completion_transformation_handler, "response_api_handler",
            return_value=MagicMock(),
        ),
    ]


def _make_cache_control_case() -> tuple[
    ResponseInputParam,
    list[AllMessageValues],
    dict[str, object],
]:
    system_message = cast(
        AllMessageValues,
        {"role": "system", "content": "Analyze the request"},
    )
    assistant_message = cast(
        AllMessageValues,
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "The code has a bug",
                    "annotations": [],
                }
            ],
        },
    )
    user_message = cast(
        AllMessageValues,
        {"role": "user", "content": "Check for security issues"},
    )
    reasoning_item = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "encrypted_content": "encrypted",
    }
    original_input = cast(
        ResponseInputParam,
        [system_message, reasoning_item, assistant_message, user_message],
    )
    _, merged_messages, _ = AnthropicCacheControlHook().get_chat_completion_prompt(
        model="azure/gpt-5-codex",
        messages=[system_message, assistant_message, user_message],
        non_default_params={"cache_control_injection_points": [{"location": "message", "role": "system"}]},
        prompt_id=None,
        prompt_variables=None,
        dynamic_callback_params={},
    )
    return original_input, merged_messages, reasoning_item


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResponsesAPIPromptManagement:

    def test_str_input_coerced_and_merged(self):
        """[A] str input is wrapped into a message list before being passed to the hook."""
        template_messages: List[AllMessageValues] = [
            {"role": "system", "content": "You are a summariser."},  # type: ignore[list-item]
        ]
        client_message: List[AllMessageValues] = [
            {"role": "user", "content": "Tell me about AI."},  # type: ignore[list-item]
        ]
        expected_merged = template_messages + client_message

        logging_obj = _make_logging_obj(
            merged_model="openai/gpt-4o",
            merged_messages=expected_merged,
        )

        patches = _patch_responses_dispatch()
        with patches[0], patches[1], patches[2]:
            import litellm

            litellm.responses(
                input="Tell me about AI.",
                model="gpt-4o",
                prompt_id="summariser-prompt",
                prompt_variables={},
                litellm_logging_obj=logging_obj,
            )

        logging_obj.get_chat_completion_prompt.assert_called_once()
        call_kwargs = logging_obj.get_chat_completion_prompt.call_args.kwargs
        # str was coerced to a single user message before being passed to the hook
        assert call_kwargs["messages"] == [
            {"role": "user", "content": "Tell me about AI."}
        ]
        assert call_kwargs["prompt_id"] == "summariser-prompt"

    def test_list_input_merged_with_template(self):
        """[B] list input is passed directly to the hook and merged with the template."""
        template_messages: List[AllMessageValues] = [
            {"role": "system", "content": "You are helpful."},  # type: ignore[list-item]
        ]
        client_messages = [
            {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
        ]
        expected_merged = template_messages + client_messages  # type: ignore[operator]

        logging_obj = _make_logging_obj(
            merged_model="openai/gpt-4o",
            merged_messages=expected_merged,  # type: ignore[arg-type]
        )

        patches = _patch_responses_dispatch()
        with patches[0], patches[1], patches[2]:
            import litellm

            litellm.responses(
                input=client_messages,  # type: ignore[arg-type]
                model="gpt-4o",
                prompt_id="helper-prompt",
                litellm_logging_obj=logging_obj,
            )

        logging_obj.get_chat_completion_prompt.assert_called_once()
        call_kwargs = logging_obj.get_chat_completion_prompt.call_args.kwargs
        assert call_kwargs["messages"] == client_messages

    def test_no_prompt_id_skips_hook(self):
        """[C] When prompt_id is absent, prompt management hooks are not called."""
        logging_obj = _make_logging_obj(
            merged_model="openai/gpt-4o",
            merged_messages=[],
            should_run=False,
        )

        patches = _patch_responses_dispatch()
        with patches[0], patches[1], patches[2]:
            import litellm

            litellm.responses(
                input="Hello",
                model="gpt-4o",
                litellm_logging_obj=logging_obj,
            )

        logging_obj.get_chat_completion_prompt.assert_not_called()

    def test_optional_params_from_template_applied(self):
        """[E] prompt_template_optional_params (e.g. temperature) flow into the request."""
        template_messages: List[AllMessageValues] = [
            {"role": "user", "content": "Hello"},  # type: ignore[list-item]
        ]
        # Simulate get_chat_completion_prompt returning merged optional params
        # that include a template-defined temperature
        merged_kwargs = {"temperature": 0.2}

        logging_obj = MagicMock()
        logging_obj.__class__ = LiteLLMLoggingObj
        logging_obj.should_run_prompt_management_hooks.return_value = True
        logging_obj.get_chat_completion_prompt.return_value = (
            "openai/gpt-4o",
            template_messages,
            merged_kwargs,
        )
        logging_obj.model_call_details = {}

        patches = _patch_responses_dispatch()
        with patches[0], patches[1], patches[2] as mock_handler:
            import litellm

            litellm.responses(
                input="Hello",
                model="gpt-4o",
                prompt_id="t",
                litellm_logging_obj=logging_obj,
            )

        # temperature from the template should reach the downstream handler via local_vars
        handler_call_kwargs = mock_handler.call_args.kwargs
        request_params = handler_call_kwargs.get("responses_api_request", {})
        assert request_params.get("temperature") == 0.2

    def test_model_override_from_template(self):
        """[D] Model returned by the prompt hook overrides the original request model."""
        template_messages: List[AllMessageValues] = [
            {"role": "user", "content": "{{query}}"},  # type: ignore[list-item]
        ]
        logging_obj = _make_logging_obj(
            merged_model="openai/gpt-4o-mini",  # overridden model from template
            merged_messages=template_messages,
        )

        patches = _patch_responses_dispatch()
        with patches[0], patches[1], patches[2] as mock_handler:
            import litellm

            litellm.responses(
                input="What is AI?",
                model="gpt-4o",
                prompt_id="query-prompt",
                prompt_variables={"query": "What is AI?"},
                litellm_logging_obj=logging_obj,
            )

        # The model passed to the downstream handler should be the overridden one
        handler_call_kwargs = mock_handler.call_args.kwargs
        assert handler_call_kwargs.get("model") == "gpt-4o-mini"

    def test_non_message_input_items_filtered(self):
        """[F] Non-message items in ResponseInputParam (e.g. function_call_output) are
        filtered out before being passed to the prompt hook, avoiding malformed merges.
        """
        template_messages: List[AllMessageValues] = [
            {"role": "system", "content": "You are helpful."},  # type: ignore[list-item]
        ]
        mixed_input = [
            {"role": "user", "content": "Hello"},
            {"type": "function_call_output", "call_id": "abc", "output": "42"},
        ]
        logging_obj = _make_logging_obj(
            merged_model="openai/gpt-4o",
            merged_messages=template_messages + [{"role": "user", "content": "Hello"}],  # type: ignore[operator]
        )

        patches = _patch_responses_dispatch()
        with patches[0], patches[1], patches[2]:
            import litellm

            litellm.responses(
                input=mixed_input,  # type: ignore[arg-type]
                model="gpt-4o",
                prompt_id="filter-test",
                litellm_logging_obj=logging_obj,
            )

        call_kwargs = logging_obj.get_chat_completion_prompt.call_args.kwargs
        passed_messages = call_kwargs["messages"]
        assert all(isinstance(m, dict) and "role" in m for m in passed_messages)
        assert len(passed_messages) == 1

    def test_cache_control_hook_preserves_reasoning_items(self):
        original_input, merged_messages, reasoning_item = _make_cache_control_case()
        logging_obj = _make_logging_obj(
            merged_model="azure/gpt-5-codex",
            merged_messages=merged_messages,
        )

        patches = _patch_responses_dispatch()
        with patches[0], patches[1], patches[2] as mock_handler:
            import litellm

            litellm.responses(
                input=original_input,
                model="azure/gpt-5-codex",
                litellm_logging_obj=logging_obj,
                cache_control_injection_points=[{"location": "message", "role": "system"}],
            )

        sent_input = mock_handler.call_args.kwargs["input"]
        assert [item.get("type") for item in sent_input] == [
            None,
            "reasoning",
            "message",
            None,
        ]
        assert sent_input[0]["cache_control"] == {"type": "ephemeral"}
        assert sent_input[1] == reasoning_item
        assert sent_input[2]["id"] == "msg_1"

    def test_all_non_message_input_items_remain_unchanged(self):
        reasoning_item = {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "encrypted_content": "encrypted",
        }
        original_input = cast(ResponseInputParam, [reasoning_item])
        logging_obj = _make_logging_obj(
            merged_model="openai/gpt-4o",
            merged_messages=[
                cast(
                    AllMessageValues,
                    {"role": "system", "content": "Analyze the request"},
                )
            ],
        )

        patches = _patch_responses_dispatch()
        with patches[0], patches[1], patches[2] as mock_handler:
            import litellm

            litellm.responses(
                input=original_input,
                model="gpt-4o",
                prompt_id="all-non-message",
                litellm_logging_obj=logging_obj,
            )

        assert mock_handler.call_args.kwargs["input"] == original_input

    def test_model_override_re_resolves_provider(self):
        """[G] When the prompt template overrides the model to a different provider,
        custom_llm_provider is re-resolved so downstream routing uses the correct provider.
        """
        template_messages: List[AllMessageValues] = [
            {"role": "user", "content": "Hi"},  # type: ignore[list-item]
        ]
        logging_obj = _make_logging_obj(
            merged_model="anthropic/claude-3-5-sonnet",
            merged_messages=template_messages,
        )

        patches = _patch_responses_dispatch()
        with (
            patch.object(
                import_module("litellm.responses.main").litellm, "get_llm_provider",
                side_effect=_provider_by_model,
            ),
            patches[1],
            patches[2] as mock_handler,
        ):
            import litellm

            litellm.responses(
                input="Hi",
                model="gpt-4o",
                prompt_id="cross-provider",
                litellm_logging_obj=logging_obj,
            )

        handler_call_kwargs = mock_handler.call_args.kwargs
        assert handler_call_kwargs.get("custom_llm_provider") == "anthropic"


class TestAsyncResponsesAPIPromptManagement:
    """Tests for the async aresponses() prompt management path.

    aresponses() calls async_get_chat_completion_prompt at the outer async
    level, then pops prompt_id from kwargs and passes merged_optional_params
    via an internal kwarg. The sync responses() path sees no prompt_id and
    skips the sync hook entirely — preventing double-merge of template messages.
    """

    @pytest.mark.asyncio
    async def test_async_calls_async_hook_not_sync(self):
        """[H] aresponses() invokes async_get_chat_completion_prompt and the
        sync get_chat_completion_prompt is NOT called (no double-merge)."""
        template_messages: List[AllMessageValues] = [
            {"role": "system", "content": "You are helpful."},  # type: ignore[list-item]
        ]
        logging_obj = _make_logging_obj(
            merged_model="openai/gpt-4o",
            merged_messages=template_messages + [{"role": "user", "content": "Hi"}],  # type: ignore[list-item]
        )

        patches = _patch_responses_dispatch()
        with patches[0], patches[1], patches[2]:
            import litellm

            await litellm.aresponses(
                input="Hi",
                model="gpt-4o",
                prompt_id="async-test",
                prompt_variables={},
                litellm_logging_obj=logging_obj,
            )

        logging_obj.async_get_chat_completion_prompt.assert_called_once()
        logging_obj.get_chat_completion_prompt.assert_not_called()
        call_kwargs = logging_obj.async_get_chat_completion_prompt.call_args.kwargs
        assert call_kwargs["prompt_id"] == "async-test"

    @pytest.mark.asyncio
    async def test_async_optional_params_propagated(self):
        """[I] Template-defined optional params (e.g. temperature) from the async
        hook reach the downstream handler — they are NOT silently discarded."""
        template_messages: List[AllMessageValues] = [
            {"role": "user", "content": "Hello"},  # type: ignore[list-item]
        ]
        logging_obj = _make_logging_obj(
            merged_model="openai/gpt-4o",
            merged_messages=template_messages,
            merged_optional_params={"temperature": 0.7},
        )

        patches = _patch_responses_dispatch()
        with patches[0], patches[1], patches[2] as mock_handler:
            import litellm

            await litellm.aresponses(
                input="Hello",
                model="gpt-4o",
                prompt_id="async-temp",
                litellm_logging_obj=logging_obj,
            )

        logging_obj.get_chat_completion_prompt.assert_not_called()
        handler_call_kwargs = mock_handler.call_args.kwargs
        request_params = handler_call_kwargs.get("responses_api_request", {})
        assert request_params.get("temperature") == 0.7

    @pytest.mark.asyncio
    async def test_async_non_message_items_filtered(self):
        """[J] Non-message items are filtered in the async path too."""
        template_messages: List[AllMessageValues] = [
            {"role": "system", "content": "Be helpful."},  # type: ignore[list-item]
        ]
        mixed_input = [
            {"role": "user", "content": "Hello"},
            {"type": "function_call_output", "call_id": "abc", "output": "42"},
        ]
        logging_obj = _make_logging_obj(
            merged_model="openai/gpt-4o",
            merged_messages=template_messages + [{"role": "user", "content": "Hello"}],  # type: ignore[operator]
        )

        patches = _patch_responses_dispatch()
        with patches[0], patches[1], patches[2]:
            import litellm

            await litellm.aresponses(
                input=mixed_input,  # type: ignore[arg-type]
                model="gpt-4o",
                prompt_id="async-filter",
                litellm_logging_obj=logging_obj,
            )

        logging_obj.async_get_chat_completion_prompt.assert_called_once()
        logging_obj.get_chat_completion_prompt.assert_not_called()
        call_kwargs = logging_obj.async_get_chat_completion_prompt.call_args.kwargs
        passed_messages = call_kwargs["messages"]
        assert all(isinstance(m, dict) and "role" in m for m in passed_messages)
        assert len(passed_messages) == 1

    @pytest.mark.asyncio
    async def test_async_cache_control_hook_preserves_reasoning_items(self):
        original_input, merged_messages, reasoning_item = _make_cache_control_case()
        logging_obj = _make_logging_obj(
            merged_model="azure/gpt-5-codex",
            merged_messages=merged_messages,
        )

        patches = _patch_responses_dispatch()
        with patches[0], patches[1], patches[2] as mock_handler:
            import litellm

            await litellm.aresponses(
                input=original_input,
                model="azure/gpt-5-codex",
                litellm_logging_obj=logging_obj,
                cache_control_injection_points=[{"location": "message", "role": "system"}],
            )

        sent_input = mock_handler.call_args.kwargs["input"]
        assert [item.get("type") for item in sent_input] == [
            None,
            "reasoning",
            "message",
            None,
        ]
        assert sent_input[0]["cache_control"] == {"type": "ephemeral"}
        assert sent_input[1] == reasoning_item
        assert sent_input[2]["id"] == "msg_1"


# ---------------------------------------------------------------------------
# Cross-provider model swap guard (prompt swaps model after credential resolution)
# ---------------------------------------------------------------------------


def test_resolve_prompt_swapped_provider_raises_cross_provider_with_credentials():
    import litellm
    from litellm.responses.main import _resolve_prompt_swapped_provider

    with pytest.raises(litellm.BadRequestError, match="Refusing to send"):
        _resolve_prompt_swapped_provider(
            original_model="anthropic/claude-haiku-4-5",
            swapped_model="gpt-4o-mini",
            custom_llm_provider="anthropic",
            kwargs={"api_key": "sk-ant-test"},
            prompt_id="p1",
        )


def test_resolve_prompt_swapped_provider_allows_swap_without_credentials():
    from litellm.responses.main import _resolve_prompt_swapped_provider

    assert (
        _resolve_prompt_swapped_provider(
            original_model="anthropic/claude-haiku-4-5",
            swapped_model="gpt-4o-mini",
            custom_llm_provider="anthropic",
            kwargs={},
            prompt_id="p1",
        )
        == "openai"
    )


def test_resolve_prompt_swapped_provider_allows_same_provider_swap_with_credentials():
    from litellm.responses.main import _resolve_prompt_swapped_provider

    assert (
        _resolve_prompt_swapped_provider(
            original_model="openai/gpt-4o",
            swapped_model="gpt-4o-mini",
            custom_llm_provider="openai",
            kwargs={"api_key": "sk-test", "api_base": "https://api.openai.com/v1"},
            prompt_id="p1",
        )
        == "openai"
    )


def test_sync_prompt_swap_resolves_credentials_for_swapped_provider(monkeypatch: pytest.MonkeyPatch):
    import litellm

    monkeypatch.setenv("XAI_API_KEY", "sk-xai-test")
    logging_obj = _make_logging_obj("gpt-4o-mini", [{"role": "user", "content": "hi"}])
    with patch.object(  # test-quality-ok: handler boundary stub proves creds resolve for the swapped provider without network
        import_module("litellm.responses.main").base_llm_http_handler, "response_api_handler", return_value=MagicMock()
    ) as mock_handler:
        litellm.responses(input="hi", model="xai/grok-4", prompt_id="p1", litellm_logging_obj=logging_obj)

    handler_kwargs = mock_handler.call_args.kwargs
    assert handler_kwargs["model"] == "gpt-4o-mini"
    assert handler_kwargs["custom_llm_provider"] == "openai"
    assert handler_kwargs["litellm_params"].api_base is None
    assert handler_kwargs["litellm_params"].api_key != "sk-xai-test"


def test_sync_prompt_swap_cross_provider_with_credentials_raises():
    import litellm
    from litellm.responses.main import _apply_prompt_management_to_responses_call

    logging_obj = _make_logging_obj("gpt-4o-mini", [{"role": "user", "content": "hi"}])
    with pytest.raises(litellm.BadRequestError, match="Refusing to send"):
        _apply_prompt_management_to_responses_call(
            input="hi",
            tools=None,
            model="anthropic/claude-haiku-4-5",
            custom_llm_provider="anthropic",
            litellm_logging_obj=logging_obj,
            kwargs={"prompt_id": "p1", "api_key": "sk-ant-test"},
            local_vars={},
            use_chat_completions_api=False,
        )


@pytest.mark.asyncio
async def test_aresponses_prompt_swap_cross_provider_with_credentials_raises():
    import litellm

    logging_obj = _make_logging_obj("gpt-4o-mini", [{"role": "user", "content": "hi"}])
    logging_obj.async_failure_handler = AsyncMock()
    with pytest.raises(litellm.BadRequestError, match="Refusing to send"):
        await litellm.aresponses(
            input="hi",
            model="anthropic/claude-haiku-4-5",
            litellm_logging_obj=logging_obj,
            prompt_id="p1",
            api_key="sk-ant-test",
        )


def _guardrail_block() -> HTTPException:
    return HTTPException(status_code=400, detail={"error": "Violated guardrail policy"})


@pytest.mark.asyncio
async def test_async_guardrail_block_from_prompt_hook_reaches_caller_unwrapped():
    block = _guardrail_block()
    logging_obj = _make_logging_obj(merged_model="openai/gpt-4o", merged_messages=[])
    logging_obj.async_get_chat_completion_prompt = AsyncMock(side_effect=block)

    patches = _patch_responses_dispatch()
    with patches[0], patches[1], patches[2], pytest.raises(HTTPException) as exc_info:
        await litellm.aresponses(input="Hi", model="gpt-4o", prompt_id="blocked", litellm_logging_obj=logging_obj)

    assert exc_info.value is block


def test_sync_guardrail_block_from_prompt_hook_reaches_caller_unwrapped():
    block = _guardrail_block()
    logging_obj = _make_logging_obj(merged_model="openai/gpt-4o", merged_messages=[])
    logging_obj.get_chat_completion_prompt.side_effect = block

    patches = _patch_responses_dispatch()
    with patches[0], patches[1], patches[2], pytest.raises(HTTPException) as exc_info:
        litellm.responses(input="Hi", model="gpt-4o", prompt_id="blocked", litellm_logging_obj=logging_obj)

    assert exc_info.value is block


_TOOL_TURN_INPUT: Final = [
    {"role": "system", "content": "You are a weather assistant"},
    {"role": "user", "content": "What is the weather in Bogota?"},
    {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": '{"city": "Bogota"}'},
    {"type": "function_call_output", "call_id": "call_1", "output": '{"temp_c": 14}'},
]


def _real_logging_obj() -> LiteLLMLoggingObj:
    return LiteLLMLoggingObj(
        model="gpt-4o",
        messages=copy.deepcopy(_TOOL_TURN_INPUT),
        stream=False,
        call_type="aresponses",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="call-id",
        function_id="function-id",
    )


_TOOL_TURN_INPUT_WITH_INJECTED_MARKER: Final = [
    {**_TOOL_TURN_INPUT[0], "cache_control": {"type": "ephemeral"}},
    *_TOOL_TURN_INPUT[1:],
]


_OPENAI_RESPONSE: Final = {
    "id": "resp_1",
    "object": "response",
    "created_at": 1767225600,
    "status": "completed",
    "model": "gpt-4o",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Bogota is 14C", "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
}


def _mock_openai_responses(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(200, json=_OPENAI_RESPONSE)
    )


@pytest.mark.asyncio
async def test_async_cache_control_injection_logs_role_less_input_items(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    responses_route: Final = _mock_openai_responses(respx_mock)
    logging_obj: Final = _real_logging_obj()

    await litellm.aresponses(
        input=copy.deepcopy(_TOOL_TURN_INPUT),
        model="gpt-4o",
        api_key="sk-test",
        cache_control_injection_points=[{"location": "message", "role": "system"}],
        litellm_logging_obj=logging_obj,
    )

    assert responses_route.call_count == 1
    assert logging_obj.model_call_details["messages"] == _TOOL_TURN_INPUT_WITH_INJECTED_MARKER
    assert logging_obj.messages == _TOOL_TURN_INPUT_WITH_INJECTED_MARKER


def test_sync_cache_control_injection_logs_role_less_input_items(respx_mock: respx.MockRouter) -> None:
    responses_route: Final = _mock_openai_responses(respx_mock)
    logging_obj: Final = _real_logging_obj()

    litellm.responses(
        input=copy.deepcopy(_TOOL_TURN_INPUT),
        model="gpt-4o",
        api_key="sk-test",
        cache_control_injection_points=[{"location": "message", "role": "system"}],
        litellm_logging_obj=logging_obj,
    )

    assert responses_route.call_count == 1
    assert logging_obj.model_call_details["messages"] == _TOOL_TURN_INPUT_WITH_INJECTED_MARKER
    assert logging_obj.messages == _TOOL_TURN_INPUT_WITH_INJECTED_MARKER


@pytest.mark.asyncio
async def test_async_vector_store_context_logs_role_less_input_items(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(
        litellm,
        "vector_store_registry",
        VectorStoreRegistry(
            vector_stores=[LiteLLM_ManagedVectorStore(vector_store_id="vs_weather", custom_llm_provider="openai")]
        ),
    )
    search_route: Final = respx_mock.post("https://api.openai.com/v1/vector_stores/vs_weather/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "vector_store.search_results.page",
                "search_query": "What is the weather in Bogota?",
                "data": [{"file_id": "file_1", "filename": "bogota.txt", "score": 0.9, "attributes": {},
                          "content": [{"type": "text", "text": "Bogota averages 14C"}]}],
                "has_more": False,
                "next_page": None,
            },
        )
    )
    responses_route: Final = _mock_openai_responses(respx_mock)
    logging_obj: Final = _real_logging_obj()

    await litellm.aresponses(
        input=copy.deepcopy(_TOOL_TURN_INPUT),
        model="gpt-4o",
        api_key="sk-test",
        vector_store_ids=["vs_weather"],
        litellm_logging_obj=logging_obj,
    )

    assert search_route.call_count == 1
    assert responses_route.call_count == 1
    assert logging_obj.model_call_details["messages"] == [
        _TOOL_TURN_INPUT[0],
        {"role": "user", "content": "Context:\n\nBogota averages 14C\n\n"},
        *_TOOL_TURN_INPUT[1:],
    ]
