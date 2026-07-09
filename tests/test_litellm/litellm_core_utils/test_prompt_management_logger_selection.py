"""
Regression tests for https://github.com/BerriAI/litellm/issues/31887

Toggling `cache_control_injection_points` (Cache Control Injection Points in the
proxy UI) crashed every request with
`ValueError: prompt_id is required for Prompt Management Base class` whenever a
prompt-id-based prompt management logger was registered: the generic
CustomPromptManagement fallback in
`Logging.get_custom_logger_for_prompt_management` was checked before the
AnthropicCacheControlHook, so the wrong logger handled the (prompt_id-less)
cache-control call and raised — and the injection points were never applied.
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath("../../.."))  # Adds the parent directory to the system path

import litellm
from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook
from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.integrations.prompt_management_base import PromptManagementBase
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging


class _PromptIdRequiringPromptManager(CustomPromptManagement):
    """Mimics prompt managers (dotprompt, arize, Langfuse, ...) that require a prompt_id.

    Reuses `PromptManagementBase.get_chat_completion_prompt`, which is the code
    path that raised `ValueError` when selected for a prompt_id-less call.
    """

    get_chat_completion_prompt = PromptManagementBase.get_chat_completion_prompt

    @property
    def integration_name(self) -> str:
        return "prompt-id-requiring-prompt-manager"

    def _compile_prompt_helper(self, *args, **kwargs):
        raise NotImplementedError("should never be reached in these tests")


@pytest.fixture
def logging_obj():
    return LitellmLogging(
        model="anthropic/claude-sonnet-4-5",
        messages=[{"role": "user", "content": "Hey"}],
        stream=False,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="12345",
        function_id="1245",
    )


@pytest.fixture
def registered_prompt_manager():
    """Register a prompt-id-requiring prompt manager and restore global state after."""
    saved_callbacks = list(litellm.callbacks)
    prompt_manager = _PromptIdRequiringPromptManager()
    litellm.logging_callback_manager.add_litellm_callback(prompt_manager)
    try:
        yield prompt_manager
    finally:
        litellm.callbacks = saved_callbacks


INJECTION_POINTS = [{"location": "message", "role": "user", "index": None, "control": None}]


def test_cache_control_hook_selected_over_prompt_id_requiring_manager(logging_obj, registered_prompt_manager):
    """Without a prompt_id, a cache-control-triggered call must select the hook,
    not the registered prompt-id-based manager."""
    custom_logger = logging_obj.get_custom_logger_for_prompt_management(
        model="anthropic/claude-sonnet-4-5",
        non_default_params={"cache_control_injection_points": INJECTION_POINTS},
        prompt_id=None,
        dynamic_callback_params={},
    )

    assert isinstance(custom_logger, AnthropicCacheControlHook)


def test_cache_control_injection_applied_when_prompt_manager_registered(logging_obj, registered_prompt_manager):
    """The full prompt path must not raise AND must actually apply the injection points.

    A guard that only skips the wrongly-selected prompt manager would pass the
    no-crash half of this test but leave the messages without cache_control.
    """
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

    model, result_messages, non_default_params = logging_obj.get_chat_completion_prompt(
        model="anthropic/claude-sonnet-4-5",
        messages=messages,
        non_default_params={"cache_control_injection_points": INJECTION_POINTS},
        prompt_variables=None,
        prompt_id=None,
    )

    assert model == "anthropic/claude-sonnet-4-5"
    user_message = result_messages[0]
    assert any(isinstance(block, dict) and block.get("cache_control") for block in user_message["content"]), (
        f"cache_control was not injected: {user_message}"
    )
    # the hook consumes the message-level injection points
    assert not non_default_params.get("cache_control_injection_points")


def test_prompt_id_flow_still_selects_registered_prompt_manager(logging_obj, registered_prompt_manager):
    """Calls that carry a prompt_id keep the existing selection behavior."""
    custom_logger = logging_obj.get_custom_logger_for_prompt_management(
        model="anthropic/claude-sonnet-4-5",
        non_default_params={},
        prompt_id="my-prompt",
        dynamic_callback_params={},
    )

    assert custom_logger is registered_prompt_manager


def test_prompt_management_base_skips_without_prompt_id():
    """`PromptManagementBase.get_chat_completion_prompt` must skip, not raise,
    when there is no prompt_id to compile."""
    prompt_manager = _PromptIdRequiringPromptManager()
    messages = [{"role": "user", "content": "Hello"}]
    non_default_params = {"temperature": 0.5}

    model, result_messages, result_params = prompt_manager.get_chat_completion_prompt(
        model="anthropic/claude-sonnet-4-5",
        messages=messages,
        non_default_params=non_default_params,
        prompt_id=None,
        prompt_variables=None,
        dynamic_callback_params={},
    )

    assert model == "anthropic/claude-sonnet-4-5"
    assert result_messages == messages
    assert result_params == non_default_params
