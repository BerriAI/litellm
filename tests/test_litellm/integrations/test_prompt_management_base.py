"""
Tests for PromptManagementBase.get_chat_completion_prompt behavior when prompt_id is None.

Verifies that the sync method returns (model, messages, non_default_params) unchanged
when prompt_id is None, matching the async method's behavior.

Regression test for https://github.com/BerriAI/litellm/issues/25425
"""

import os
import sys
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

from litellm.integrations.prompt_management_base import (
    PromptManagementBase,
    PromptManagementClient,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.prompts.init_prompts import PromptSpec
from litellm.types.utils import StandardCallbackDynamicParams


class ConcretePromptManager(PromptManagementBase):
    """Minimal concrete implementation for testing the base class."""

    @property
    def integration_name(self) -> str:
        return "test"

    def should_run_prompt_management(
        self,
        prompt_id: Optional[str],
        prompt_spec: Optional[PromptSpec],
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> bool:
        if prompt_id is None:
            return False
        return True

    def _compile_prompt_helper(
        self,
        prompt_id: Optional[str],
        prompt_spec: Optional[PromptSpec],
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_label: Optional[str] = None,
        prompt_version: Optional[int] = None,
    ) -> PromptManagementClient:
        return PromptManagementClient(
            prompt_id=prompt_id,
            prompt_template=[{"role": "system", "content": "You are a test assistant"}],
            prompt_template_model=None,
            prompt_template_optional_params=None,
            completed_messages=None,
        )

    async def async_compile_prompt_helper(
        self,
        prompt_id: Optional[str],
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_spec: Optional[PromptSpec] = None,
        prompt_label: Optional[str] = None,
        prompt_version: Optional[int] = None,
    ) -> PromptManagementClient:
        return self._compile_prompt_helper(
            prompt_id=prompt_id,
            prompt_spec=prompt_spec,
            prompt_variables=prompt_variables,
            dynamic_callback_params=dynamic_callback_params,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
        )


@pytest.fixture
def manager():
    return ConcretePromptManager()


@pytest.fixture
def dynamic_callback_params():
    return StandardCallbackDynamicParams()


def test_sync_get_chat_completion_prompt_none_prompt_id_returns_passthrough(
    manager, dynamic_callback_params
):
    """Sync get_chat_completion_prompt should return unchanged params when prompt_id is None."""
    model = "gpt-4"
    messages: List[AllMessageValues] = [{"role": "user", "content": "Hello"}]
    non_default_params = {"temperature": 0.7}

    result_model, result_messages, result_params = manager.get_chat_completion_prompt(
        model=model,
        messages=messages,
        non_default_params=non_default_params,
        prompt_id=None,
        prompt_variables=None,
        dynamic_callback_params=dynamic_callback_params,
    )

    assert result_model == model
    assert result_messages == messages
    assert result_params == non_default_params


@pytest.mark.asyncio
async def test_async_get_chat_completion_prompt_none_prompt_id_returns_passthrough(
    manager, dynamic_callback_params
):
    """Async get_chat_completion_prompt should return unchanged params when prompt_id is None."""
    model = "gpt-4"
    messages: List[AllMessageValues] = [{"role": "user", "content": "Hello"}]
    non_default_params = {"temperature": 0.7}

    litellm_logging_obj = MagicMock()

    result_model, result_messages, result_params = (
        await manager.async_get_chat_completion_prompt(
            model=model,
            messages=messages,
            non_default_params=non_default_params,
            prompt_id=None,
            prompt_variables=None,
            dynamic_callback_params=dynamic_callback_params,
            litellm_logging_obj=litellm_logging_obj,
        )
    )

    assert result_model == model
    assert result_messages == messages
    assert result_params == non_default_params


def test_sync_get_chat_completion_prompt_with_valid_prompt_id(
    manager, dynamic_callback_params
):
    """Sync get_chat_completion_prompt should process prompt when prompt_id is provided."""
    model = "gpt-4"
    messages: List[AllMessageValues] = [{"role": "user", "content": "Hello"}]
    non_default_params = {"temperature": 0.7}

    result_model, result_messages, result_params = manager.get_chat_completion_prompt(
        model=model,
        messages=messages,
        non_default_params=non_default_params,
        prompt_id="test-prompt",
        prompt_variables=None,
        dynamic_callback_params=dynamic_callback_params,
    )

    # When prompt_id is valid, messages should be modified (prompt template + client messages)
    assert len(result_messages) == 2  # system prompt + user message
    assert result_messages[0]["role"] == "system"
