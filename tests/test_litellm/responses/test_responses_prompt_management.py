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

import asyncio
import copy
import time
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.responses.main import (
    _merge_prompt_management_messages_into_input,
)
from litellm.types.llms.openai import AllMessageValues

# ---------------------------------------------------------------------------
# Shared fixtures for the reasoning/non-message preservation tests (#32335)
# ---------------------------------------------------------------------------

REASONING_ITEM = {
    "type": "reasoning",
    "id": "rs_abc123",
    "summary": [],
    "encrypted_content": "gAAAA_fake_encrypted",
}
ASSISTANT_MESSAGE = {
    "type": "message",
    "id": "msg_def456",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "Here is the analysis.", "annotations": []}],
}
USER_MESSAGE = {"role": "user", "content": "Now check for security issues too"}
FUNCTION_CALL = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call_1",
    "name": "get_weather",
    "arguments": "{}",
}
FUNCTION_CALL_OUTPUT = {
    "type": "function_call_output",
    "call_id": "call_1",
    "output": "sunny",
}

# Pristine snapshot of the shared items, taken once at import.
_CANONICAL_SHARED_ITEMS = {
    "REASONING_ITEM": copy.deepcopy(REASONING_ITEM),
    "ASSISTANT_MESSAGE": copy.deepcopy(ASSISTANT_MESSAGE),
    "USER_MESSAGE": copy.deepcopy(USER_MESSAGE),
    "FUNCTION_CALL": copy.deepcopy(FUNCTION_CALL),
    "FUNCTION_CALL_OUTPUT": copy.deepcopy(FUNCTION_CALL_OUTPUT),
}


@pytest.fixture(autouse=True)
def _fresh_shared_items():
    """Give every test its own fresh copies of the shared message/non-message items.

    The merge logic is identity/id based, so tests intentionally reuse the *same*
    objects within a test to simulate identity-preserving hooks. Rebinding fresh
    deep-copies of the module-level items before each test preserves that within-test
    identity while isolating tests from one another, so a future test that mutates a
    shared item can never leak state into another test.
    """
    module_globals = globals()
    for name, template in _CANONICAL_SHARED_ITEMS.items():
        module_globals[name] = copy.deepcopy(template)
    yield


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
    logging_obj.model_call_details = {}
    return logging_obj


def _patch_responses_dispatch():
    """Patch everything after the prompt management block so tests stay unit-level."""
    return [
        patch(
            "litellm.responses.main.litellm.get_llm_provider",
            return_value=("gpt-4o", "openai", None, None),
        ),
        patch(
            "litellm.responses.mcp.litellm_proxy_mcp_handler."
            "LiteLLM_Proxy_MCP_Handler._should_use_litellm_mcp_gateway",
            return_value=False,
        ),
        patch(
            "litellm.responses.main.ProviderConfigManager"
            ".get_provider_responses_api_config",
            return_value=None,
        ),
        patch(
            "litellm.responses.main.litellm_completion_transformation_handler"
            ".response_api_handler",
            return_value=MagicMock(),
        ),
    ]


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
        with patches[0], patches[1], patches[2], patches[3]:
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
        with patches[0], patches[1], patches[2], patches[3]:
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
        with patches[0], patches[1], patches[2], patches[3]:
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
        with patches[0], patches[1], patches[2], patches[3] as mock_handler:
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
        with patches[0], patches[1], patches[2], patches[3] as mock_handler:
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
        assert handler_call_kwargs.get("model") == "openai/gpt-4o-mini"

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
        with patches[0], patches[1], patches[2], patches[3]:
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
            patch(
                "litellm.responses.main.litellm.get_llm_provider",
                side_effect=[
                    ("gpt-4o", "openai", None, None),
                    ("claude-3-5-sonnet", "anthropic", None, None),
                ],
            ),
            patches[1],
            patches[2],
            patches[3] as mock_handler,
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
        with patches[0], patches[1], patches[2], patches[3]:
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
        with patches[0], patches[1], patches[2], patches[3] as mock_handler:
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
        with patches[0], patches[1], patches[2], patches[3]:
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


# ---------------------------------------------------------------------------
# Regression tests for issue #32335:
# non-message items (reasoning / function_call / function_call_output) must NOT
# be dropped when a prompt-management hook runs on a Responses API call.
# ---------------------------------------------------------------------------


class TestMergePromptManagementMessagesIntoInput:
    """Direct unit tests for the merge helper that recombines the hook's message
    output with the non-message items that were filtered out before the hook ran."""

    def test_passthrough_hook_preserves_reasoning_in_place(self):
        """[#32335] Hook returns the messages unchanged -> reasoning item stays put,
        immediately preceding its paired assistant message."""
        original = [REASONING_ITEM, ASSISTANT_MESSAGE, USER_MESSAGE]
        merged = [ASSISTANT_MESSAGE, USER_MESSAGE]  # message subset, unchanged

        result = _merge_prompt_management_messages_into_input(original, merged)

        assert result == [REASONING_ITEM, ASSISTANT_MESSAGE, USER_MESSAGE]
        # reasoning immediately precedes its message
        assert result.index(REASONING_ITEM) == result.index(ASSISTANT_MESSAGE) - 1

    def test_prepended_template_message_keeps_reasoning_paired(self):
        """[#32335] Hook prepends a system/template message -> preamble goes to the
        front, and reasoning still immediately precedes its assistant message."""
        original = [REASONING_ITEM, ASSISTANT_MESSAGE, USER_MESSAGE]
        system_msg = {"role": "system", "content": "You are helpful."}
        merged = [system_msg, ASSISTANT_MESSAGE, USER_MESSAGE]

        result = _merge_prompt_management_messages_into_input(original, merged)

        assert result == [system_msg, REASONING_ITEM, ASSISTANT_MESSAGE, USER_MESSAGE]
        # reasoning is not duplicated and still directly precedes its message
        assert result.count(REASONING_ITEM) == 1
        assert result.index(REASONING_ITEM) == result.index(ASSISTANT_MESSAGE) - 1

    def test_function_call_items_preserved(self):
        """[#32335] function_call and function_call_output items are preserved in
        their original positions alongside reasoning."""
        original = [
            REASONING_ITEM,
            ASSISTANT_MESSAGE,
            FUNCTION_CALL,
            FUNCTION_CALL_OUTPUT,
            USER_MESSAGE,
        ]
        merged = [ASSISTANT_MESSAGE, USER_MESSAGE]  # only the two messages are seen by the hook

        result = _merge_prompt_management_messages_into_input(original, merged)

        assert result == [
            REASONING_ITEM,
            ASSISTANT_MESSAGE,
            FUNCTION_CALL,
            FUNCTION_CALL_OUTPUT,
            USER_MESSAGE,
        ]

    def test_transformed_messages_are_used(self):
        """The hook's *transformed* message content is what ends up in the merged
        input (we splice the hook output in, not the originals)."""
        original = [REASONING_ITEM, ASSISTANT_MESSAGE, USER_MESSAGE]
        transformed_user = {"role": "user", "content": "REWRITTEN by template"}
        merged = [ASSISTANT_MESSAGE, transformed_user]

        result = _merge_prompt_management_messages_into_input(original, merged)

        assert result == [REASONING_ITEM, ASSISTANT_MESSAGE, transformed_user]

    def test_pure_chat_input_unchanged(self):
        """No non-message items -> the hook output is authoritative (unchanged
        behavior, including any prepended template messages)."""
        original = [USER_MESSAGE]
        system_msg = {"role": "system", "content": "template"}
        merged = [system_msg, USER_MESSAGE]

        result = _merge_prompt_management_messages_into_input(original, merged)

        assert result == [system_msg, USER_MESSAGE]

    def test_str_input_returns_hook_output(self):
        """A plain-string input has no non-message items to preserve."""
        system_msg = {"role": "system", "content": "template"}
        merged = [system_msg, {"role": "user", "content": "hi"}]

        result = _merge_prompt_management_messages_into_input("hi", merged)

        assert result == merged

    def test_anchor_message_removed_drops_orphan_and_warns(self):
        """If the hook REMOVES the message a non-message item was anchored to, the
        orphan is dropped (with a warning) rather than re-attached to an unrelated
        message. This is the safe choice: it never produces a mis-paired request
        (the failure mode #32335 is about), it just can't preserve an item whose
        partner the hook deleted."""
        original = [REASONING_ITEM, ASSISTANT_MESSAGE, USER_MESSAGE]
        # Hook dropped the assistant message (reasoning's anchor) but kept the user one.
        merged = [USER_MESSAGE]

        with patch("litellm.responses.main.verbose_logger.warning") as mock_warn:
            result = _merge_prompt_management_messages_into_input(original, merged)

        # reasoning is NOT silently mis-placed before the surviving (unrelated) user msg
        assert REASONING_ITEM not in result
        assert result == [USER_MESSAGE]
        mock_warn.assert_called_once()

    def test_vector_store_insert_at_second_to_last_keeps_pairing(self):
        """Mirrors VectorStorePreCallHook: context inserted via ``messages.insert(-1)``
        with the ORIGINAL message objects preserved by identity. Reasoning must stay
        with its assistant message even though the injected message is NOT a prefix."""
        original = [REASONING_ITEM, ASSISTANT_MESSAGE, USER_MESSAGE]
        context_msg = {"role": "user", "content": "Context:\n\n...retrieved..."}
        merged = [ASSISTANT_MESSAGE, USER_MESSAGE]  # same objects (list.copy preserves refs)
        merged.insert(-1, context_msg)  # -> [ASSISTANT, context, USER]

        result = _merge_prompt_management_messages_into_input(original, merged)

        assert result == [REASONING_ITEM, ASSISTANT_MESSAGE, context_msg, USER_MESSAGE]
        # reasoning immediately precedes its OWN assistant message, not the context msg
        assert result.index(REASONING_ITEM) == result.index(ASSISTANT_MESSAGE) - 1

    def test_cache_control_deepcopy_reattaches_via_message_id(self):
        """Mirrors AnthropicCacheControlHook: messages are deep-copied (object identity
        lost) but their message ids are preserved. The reasoning item is re-attached to
        its message via the id fallback, not object identity."""
        original = [REASONING_ITEM, ASSISTANT_MESSAGE, USER_MESSAGE]
        merged = copy.deepcopy([ASSISTANT_MESSAGE, USER_MESSAGE])  # new objects, same ids

        with patch("litellm.responses.main.verbose_logger.warning") as mock_warn:
            result = _merge_prompt_management_messages_into_input(original, merged)

        # value-equal to the paired ordering; reasoning stays in front of its message
        assert result == [REASONING_ITEM, ASSISTANT_MESSAGE, USER_MESSAGE]
        assert result[0] is REASONING_ITEM
        assert result[1] is merged[0]  # the deep-copied assistant message (matched by id)
        mock_warn.assert_not_called()

    def test_deepcopy_without_ids_maps_positionally(self):
        """When messages carry NO ids and are deep-copied (neither identity nor id can
        locate them) but the count is preserved, fall back to positional slot-for-slot
        mapping so the reasoning item still precedes its message."""
        reasoning = {"type": "reasoning", "id": "rs_x", "summary": [], "encrypted_content": "e"}
        assistant_no_id = {"role": "assistant", "content": "hi"}  # no id
        user_no_id = {"role": "user", "content": "next"}  # no id
        original = [reasoning, assistant_no_id, user_no_id]
        merged = copy.deepcopy([assistant_no_id, user_no_id])  # no ids, no identity

        with patch("litellm.responses.main.verbose_logger.warning") as mock_warn:
            result = _merge_prompt_management_messages_into_input(original, merged)

        assert result[0] is reasoning
        assert result == [reasoning, merged[0], merged[1]]
        mock_warn.assert_not_called()

    def test_partial_replacement_reattaches_surviving_items_via_id(self):
        """The partial-identity + equal-count case: the hook keeps some messages by
        identity and deep-copies another (new object, SAME id), preserving count. The id
        fallback re-attaches BOTH reasoning items correctly -- under identity-only
        matching the second one would have been dropped."""
        a1 = {
            "type": "message", "id": "msg_1", "role": "assistant",
            "content": [{"type": "output_text", "text": "one", "annotations": []}],
        }
        a2 = {
            "type": "message", "id": "msg_2", "role": "assistant",
            "content": [{"type": "output_text", "text": "two", "annotations": []}],
        }
        r1 = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "e1"}
        r2 = {"type": "reasoning", "id": "rs_2", "summary": [], "encrypted_content": "e2"}
        u = {"role": "user", "content": "go"}
        original = [r1, a1, r2, a2, u]
        # a1 kept by identity, a2 deep-copied (same id), u kept by identity; count preserved
        merged = [a1, copy.deepcopy(a2), u]

        with patch("litellm.responses.main.verbose_logger.warning") as mock_warn:
            result = _merge_prompt_management_messages_into_input(original, merged)

        assert result == [r1, a1, r2, merged[1], u]
        assert r2 in result  # preserved via id fallback (would be dropped under identity-only)
        mock_warn.assert_not_called()

    def test_partial_unlocatable_equal_count_uses_positional_fallback(self):
        """The partial + equal-count case where a message is GENUINELY unlocatable
        (replaced by a different message, no matching identity or id) but the count is
        preserved. Rather than drop the item anchored to it, fall back to positional
        substitution so it is preserved. (Order-preserving equal-count transform.)"""
        a1 = {
            "type": "message", "id": "msg_1", "role": "assistant",
            "content": [{"type": "output_text", "text": "one", "annotations": []}],
        }
        a2 = {
            "type": "message", "id": "msg_2", "role": "assistant",
            "content": [{"type": "output_text", "text": "two", "annotations": []}],
        }
        r1 = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "e1"}
        r2 = {"type": "reasoning", "id": "rs_2", "summary": [], "encrypted_content": "e2"}
        u = {"role": "user", "content": "go"}
        original = [r1, a1, r2, a2, u]
        # a1 kept, a2 REPLACED by a different message (id msg_99 -> not locatable), u kept;
        # count preserved (3 messages).
        replacement = {
            "type": "message", "id": "msg_99", "role": "assistant",
            "content": [{"type": "output_text", "text": "replaced", "annotations": []}],
        }
        merged = [a1, replacement, u]

        with patch("litellm.responses.main.verbose_logger.warning") as mock_warn:
            result = _merge_prompt_management_messages_into_input(original, merged)

        # positional fallback maps a2's slot to the replacement, preserving r2
        assert result == [r1, a1, r2, replacement, u]
        assert r2 in result  # preserved (previously dropped-and-warned)
        mock_warn.assert_not_called()

    def test_full_template_replacement_drops_non_message_items_with_warning(self):
        """Mirrors a prompt template that REPLACES the incoming conversation with
        entirely new messages (different objects, different count). Non-message items
        cannot be anchored to any surviving message, so they are dropped with a
        warning rather than mis-placed onto an unrelated template message."""
        original = [REASONING_ITEM, ASSISTANT_MESSAGE, USER_MESSAGE]
        merged = [
            {"role": "system", "content": "Rendered template."},
            {"role": "user", "content": "Rendered a."},
            {"role": "user", "content": "Rendered b."},
        ]  # 3 brand-new messages, none is an original object

        with patch("litellm.responses.main.verbose_logger.warning") as mock_warn:
            result = _merge_prompt_management_messages_into_input(original, merged)

        assert result == merged
        assert REASONING_ITEM not in result
        mock_warn.assert_called_once()

    def test_trailing_non_message_items_reattached_after_last_message(self):
        """A tool-result turn: input ENDS with non-message items (function_call +
        function_call_output) that trail the last message. They must be re-attached
        immediately after that surviving message, in order."""
        original = [USER_MESSAGE, ASSISTANT_MESSAGE, FUNCTION_CALL, FUNCTION_CALL_OUTPUT]
        merged = [USER_MESSAGE, ASSISTANT_MESSAGE]  # passthrough; same objects

        with patch("litellm.responses.main.verbose_logger.warning") as mock_warn:
            result = _merge_prompt_management_messages_into_input(original, merged)

        assert result == [
            USER_MESSAGE,
            ASSISTANT_MESSAGE,
            FUNCTION_CALL,
            FUNCTION_CALL_OUTPUT,
        ]
        # trailing items kept their order and stayed after the last message
        assert result.index(FUNCTION_CALL) == result.index(ASSISTANT_MESSAGE) + 1
        assert result.index(FUNCTION_CALL_OUTPUT) == result.index(FUNCTION_CALL) + 1
        mock_warn.assert_not_called()

    def test_trailing_reasoning_prepended_by_template_stays_after_its_message(self):
        """Trailing item + a hook that PREPENDS a template message (identity of the
        originals preserved). The trailing item still lands after the last original
        message, and the injected preamble stays at the front."""
        system_msg = {"role": "system", "content": "Template preamble."}
        original = [USER_MESSAGE, ASSISTANT_MESSAGE, FUNCTION_CALL_OUTPUT]
        merged = [system_msg, USER_MESSAGE, ASSISTANT_MESSAGE]  # prepend, same originals

        result = _merge_prompt_management_messages_into_input(original, merged)

        assert result == [system_msg, USER_MESSAGE, ASSISTANT_MESSAGE, FUNCTION_CALL_OUTPUT]

    def test_trailing_item_dropped_and_warns_when_last_message_removed(self):
        """If the last message (the trailing item's anchor) is removed/replaced by the
        hook, the trailing orphan is dropped with a warning rather than re-attached to
        an unrelated message."""
        original = [ASSISTANT_MESSAGE, FUNCTION_CALL_OUTPUT]  # trailing FCO anchored to ASSISTANT
        merged = [
            {"role": "system", "content": "New a."},
            {"role": "user", "content": "New b."},
        ]  # 2 new messages (count differs from 1 original) -> best-effort identity path

        with patch("litellm.responses.main.verbose_logger.warning") as mock_warn:
            result = _merge_prompt_management_messages_into_input(original, merged)

        assert result == merged
        assert FUNCTION_CALL_OUTPUT not in result
        mock_warn.assert_called_once()

    def test_hook_output_used_when_original_has_no_messages(self):
        """Edge case: the original input is entirely non-message items (e.g. a pure
        tool-output turn with no role-bearing message), but the hook still returns a
        template message (a prompt_id template that unconditionally prepends a system
        message). Prompt management still runs -- it does not silently no-op just
        because the turn had no client messages -- but the orphaned non-message item
        has no message to anchor to, so it is dropped with a warning rather than left
        unpaired at the front of the request."""
        original = [FUNCTION_CALL_OUTPUT]
        template_msg = {"role": "system", "content": "template preamble"}
        merged = [template_msg]

        with patch("litellm.responses.main.verbose_logger.warning") as mock_warn:
            result = _merge_prompt_management_messages_into_input(original, merged)

        assert result == [template_msg]
        mock_warn.assert_called_once()


class TestResponsesReasoningPreservationEndToEnd:
    """[#32335] Reasoning/non-message items survive all the way to the downstream
    handler when a prompt-management hook is active on a /responses call."""

    def _final_input(self, mock_handler):
        return mock_handler.call_args.kwargs["input"]

    def test_sync_reasoning_preserved_multi_turn(self):
        """Reporter's exact scenario through the sync responses() path."""
        original = [REASONING_ITEM, ASSISTANT_MESSAGE, USER_MESSAGE]
        # passthrough hook: returns the role-bearing subset unchanged
        logging_obj = _make_logging_obj(
            merged_model="openai/gpt-4o",
            merged_messages=[ASSISTANT_MESSAGE, USER_MESSAGE],
        )

        patches = _patch_responses_dispatch()
        with patches[0], patches[1], patches[2], patches[3] as mock_handler:
            import litellm

            litellm.responses(
                input=original,  # type: ignore[arg-type]
                model="gpt-4o",
                prompt_id="p",
                litellm_logging_obj=logging_obj,
            )

        final_input = self._final_input(mock_handler)
        assert REASONING_ITEM in final_input
        assert final_input == [REASONING_ITEM, ASSISTANT_MESSAGE, USER_MESSAGE]
        assert final_input.index(REASONING_ITEM) == final_input.index(ASSISTANT_MESSAGE) - 1

    def test_sync_prepended_template_preserves_reasoning(self):
        """Hook injects a system template message; reasoning still preserved/paired."""
        original = [REASONING_ITEM, ASSISTANT_MESSAGE, USER_MESSAGE]
        system_msg = {"role": "system", "content": "You are helpful."}
        logging_obj = _make_logging_obj(
            merged_model="openai/gpt-4o",
            merged_messages=[system_msg, ASSISTANT_MESSAGE, USER_MESSAGE],
        )

        patches = _patch_responses_dispatch()
        with patches[0], patches[1], patches[2], patches[3] as mock_handler:
            import litellm

            litellm.responses(
                input=original,  # type: ignore[arg-type]
                model="gpt-4o",
                prompt_id="p",
                litellm_logging_obj=logging_obj,
            )

        final_input = self._final_input(mock_handler)
        assert final_input == [system_msg, REASONING_ITEM, ASSISTANT_MESSAGE, USER_MESSAGE]
        assert final_input.count(REASONING_ITEM) == 1

    def test_sync_function_call_items_preserved(self):
        """function_call / function_call_output items survive the same way."""
        original = [
            REASONING_ITEM,
            ASSISTANT_MESSAGE,
            FUNCTION_CALL,
            FUNCTION_CALL_OUTPUT,
            USER_MESSAGE,
        ]
        logging_obj = _make_logging_obj(
            merged_model="openai/gpt-4o",
            merged_messages=[ASSISTANT_MESSAGE, USER_MESSAGE],
        )

        patches = _patch_responses_dispatch()
        with patches[0], patches[1], patches[2], patches[3] as mock_handler:
            import litellm

            litellm.responses(
                input=original,  # type: ignore[arg-type]
                model="gpt-4o",
                prompt_id="p",
                litellm_logging_obj=logging_obj,
            )

        final_input = self._final_input(mock_handler)
        assert final_input == original

    @pytest.mark.asyncio
    async def test_async_reasoning_preserved_multi_turn(self):
        """Reporter's exact scenario through the async aresponses() path."""
        original = [REASONING_ITEM, ASSISTANT_MESSAGE, USER_MESSAGE]
        logging_obj = _make_logging_obj(
            merged_model="openai/gpt-4o",
            merged_messages=[ASSISTANT_MESSAGE, USER_MESSAGE],
        )

        patches = _patch_responses_dispatch()
        with patches[0], patches[1], patches[2], patches[3] as mock_handler:
            import litellm

            await litellm.aresponses(
                input=original,  # type: ignore[arg-type]
                model="gpt-4o",
                prompt_id="p",
                litellm_logging_obj=logging_obj,
            )

        # sync hook must not re-run (async path already merged)
        logging_obj.get_chat_completion_prompt.assert_not_called()
        final_input = self._final_input(mock_handler)
        assert REASONING_ITEM in final_input
        assert final_input == [REASONING_ITEM, ASSISTANT_MESSAGE, USER_MESSAGE]
        assert final_input.index(REASONING_ITEM) == final_input.index(ASSISTANT_MESSAGE) - 1


class TestAzureResponsesBodyRetainsReasoning:
    """Provider-level check: the request body reaching the Azure Responses transform
    retains reasoning items when prompt management is active (#32335)."""

    def test_azure_transform_receives_reasoning_item(self):
        from litellm.llms.azure.responses.transformation import (
            AzureOpenAIResponsesAPIConfig,
        )

        original = [REASONING_ITEM, ASSISTANT_MESSAGE, USER_MESSAGE]
        logging_obj = _make_logging_obj(
            merged_model="azure/gpt-5.3-codex",
            merged_messages=[ASSISTANT_MESSAGE, USER_MESSAGE],
        )

        captured = {}

        orig_transform = AzureOpenAIResponsesAPIConfig.transform_responses_api_request

        def spy_transform(self, model, input, response_api_optional_request_params, litellm_params, headers):
            # capture the fully-built request body, then stop before any HTTP call.
            # (litellm wraps the raised exception in APIConnectionError, so the caller
            # catches broadly — the capture above is what the assertions rely on.)
            body = orig_transform(
                self, model, input, response_api_optional_request_params, litellm_params, headers
            )
            captured["body_input"] = body.get("input")
            raise RuntimeError("stop-before-network")

        import litellm

        with patch.object(
            AzureOpenAIResponsesAPIConfig,
            "transform_responses_api_request",
            spy_transform,
        ):
            try:
                litellm.responses(
                    model="azure/gpt-5.3-codex",
                    input=original,  # type: ignore[arg-type]
                    prompt_id="p",
                    litellm_logging_obj=logging_obj,
                    api_base="https://fake.openai.azure.com",
                    api_key="fake-key",
                    api_version="2024-05-01-preview",
                )
            except Exception:
                pass

        assert "body_input" in captured, "Azure transform was never reached"
        body_input = captured["body_input"]
        types = [
            (i.get("type") or ("message" if "role" in i else None)) for i in body_input
        ]
        assert "reasoning" in types, f"reasoning dropped from Azure request body: {types}"
        # reasoning precedes its paired assistant message
        reasoning_idx = next(i for i, it in enumerate(body_input) if it.get("type") == "reasoning")
        assistant_idx = next(
            i for i, it in enumerate(body_input) if it.get("type") == "message" and it.get("role") == "assistant"
        )
        assert reasoning_idx == assistant_idx - 1


class TestMergePromptManagementPerformance:
    """[#32335] The merge helper must not degrade to quadratic time on large inputs.

    A caller with a `prompt_id` configured controls both the number of role-bearing
    messages and their ids, so a large payload (e.g. many messages plus one trailing
    non-message item) previously drove ~26s of CPU time at 20k messages via repeated
    linear anchor scans and list.insert() shifting -- a request-processing resource
    exhaustion vector reachable before any provider call is made.
    """

    @staticmethod
    def _build_case(num_messages: int) -> tuple[list, list]:
        original = [{"role": "user", "content": f"msg {i}", "id": f"m{i}"} for i in range(num_messages)]
        original.append({"type": "function_call_output", "call_id": "c1", "output": "ok"})
        # deep-copied (identity lost, ids preserved) -> exercises the id-fallback path,
        # the most expensive branch under the old O(n^2) implementation
        merged = copy.deepcopy(original[:num_messages])
        return original, merged

    def test_large_input_scales_sub_quadratically(self):
        small_original, small_merged = self._build_case(1000)
        large_original, large_merged = self._build_case(8000)  # 8x the messages

        start = time.perf_counter()
        small_result = _merge_prompt_management_messages_into_input(small_original, small_merged)
        small_elapsed = time.perf_counter() - start

        start = time.perf_counter()
        large_result = _merge_prompt_management_messages_into_input(large_original, large_merged)
        large_elapsed = time.perf_counter() - start

        assert len(small_result) == 1001
        assert len(large_result) == 8001

        # O(n^2) would grow ~64x (8^2) for an 8x input; O(n) grows ~8x. Allow generous
        # headroom over linear (well below quadratic) to avoid CI-noise flakiness while
        # still catching a regression to quadratic behavior.
        max_expected = max(small_elapsed * 8 * 4, 0.5)
        assert large_elapsed < max_expected, (
            f"merge helper scaled worse than expected: {small_elapsed:.4f}s at 1k messages, "
            f"{large_elapsed:.4f}s at 8k messages (limit {max_expected:.4f}s) -- looks quadratic again"
        )
