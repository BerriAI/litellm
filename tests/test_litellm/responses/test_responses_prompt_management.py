"""
Unit tests for prompt management support in the Responses API.

Covers:
  A) str input is coerced to a message list before merging with the template
  B) list input is merged with the template
  C) no prompt_id → hook is skipped, input is unchanged
  D) model override from the prompt template is applied
"""

from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.llms.openai import AllMessageValues

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_logging_obj(
    merged_model: str,
    merged_messages: List[AllMessageValues],
    should_run: bool = True,
) -> MagicMock:
    """Return a mock LiteLLMLoggingObj pre-configured for prompt management."""
    logging_obj = MagicMock()
    # Make isinstance(logging_obj, LiteLLMLoggingObj) return True
    logging_obj.__class__ = LiteLLMLoggingObj
    logging_obj.should_run_prompt_management_hooks.return_value = should_run
    logging_obj.get_chat_completion_prompt.return_value = (
        merged_model,
        merged_messages,
        {},
    )
    logging_obj.async_get_chat_completion_prompt = AsyncMock(
        return_value=(merged_model, merged_messages, {})
    )
    # Instance attribute accessed by post-call metadata utilities
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

    def test_non_message_items_are_filtered_before_prompt_hook(self):
        """Only role-based message items should be passed to prompt hooks."""
        logging_obj = _make_logging_obj(
            merged_model="openai/gpt-4o",
            merged_messages=[{"role": "user", "content": "Hi"}],  # type: ignore[list-item]
        )

        input_items = [
            {"type": "function_call_output", "call_id": "abc", "output": "done"},
            {"role": "user", "content": "Hi"},
        ]

        patches = _patch_responses_dispatch()
        with patches[0], patches[1], patches[2], patches[3]:
            import litellm

            litellm.responses(
                input=input_items,  # type: ignore[arg-type]
                model="gpt-4o",
                prompt_id="helper-prompt",
                litellm_logging_obj=logging_obj,
            )

        logging_obj.get_chat_completion_prompt.assert_called_once()
        call_kwargs = logging_obj.get_chat_completion_prompt.call_args.kwargs
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hi"}]

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
        merged_kwargs = {"temperature": 0.2, "prompt_id": "t", "litellm_logging_obj": None}

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

    @pytest.mark.asyncio
    async def test_aresponses_uses_async_prompt_hook(self):
        """Async responses path should call async prompt hook before executor dispatch."""
        logging_obj = _make_logging_obj(
            merged_model="openai/gpt-4o-mini",
            merged_messages=[{"role": "user", "content": "Hi"}],  # type: ignore[list-item]
        )
        logging_obj.get_chat_completion_prompt.reset_mock()
        logging_obj.async_get_chat_completion_prompt.reset_mock()

        with patch(
            "litellm.responses.main.litellm.get_llm_provider",
            return_value=("gpt-4o", "openai", None, None),
        ), patch(
            "litellm.responses.main.responses", return_value=MagicMock()
        ) as mock_responses:
            import litellm

            await litellm.aresponses(
                input=[
                    {"type": "function_call_output", "call_id": "abc", "output": "done"},
                    {"role": "user", "content": "Hi"},
                ],  # type: ignore[arg-type]
                model="gpt-4o",
                prompt_id="helper-prompt",
                litellm_logging_obj=logging_obj,
            )

        logging_obj.async_get_chat_completion_prompt.assert_awaited_once()
        async_call_kwargs = logging_obj.async_get_chat_completion_prompt.call_args.kwargs
        assert async_call_kwargs["messages"] == [{"role": "user", "content": "Hi"}]
        logging_obj.get_chat_completion_prompt.assert_not_called()
        assert mock_responses.call_args.kwargs.get(
            "_skip_prompt_management_for_responses"
        ) is True
