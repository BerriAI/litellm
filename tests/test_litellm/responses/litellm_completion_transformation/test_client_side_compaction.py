import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
    _COMPACT_TARGET_RATIO,
)
from litellm.types.llms.openai import ContextManagementEntry


MESSAGES = [{"role": "user", "content": "Hello"}]
MODEL = "gpt-4o"


class TestApplyClientSideCompaction:
    def test_no_context_management_entry_returns_unchanged(self):
        """When context_management has no compaction entry, messages and list are returned unchanged."""
        context_management = [{"type": "other_type"}]

        msgs, cm = LiteLLMCompletionResponsesConfig._apply_client_side_compaction(
            messages=MESSAGES,
            model=MODEL,
            context_management=context_management,
        )

        assert msgs is MESSAGES
        assert cm is context_management

    def test_compaction_entry_calls_trim_messages(self):
        """When a compaction entry is present, trim_messages is called with the correct target."""
        compact_threshold = 10000
        expected_target = int(compact_threshold * _COMPACT_TARGET_RATIO)
        trimmed_messages = [{"role": "user", "content": "trimmed"}]

        with patch(
            "litellm.responses.litellm_completion_transformation.transformation.trim_messages",
            return_value=trimmed_messages,
        ) as mock_trim:
            msgs, cm = LiteLLMCompletionResponsesConfig._apply_client_side_compaction(
                messages=MESSAGES,
                model=MODEL,
                context_management=[
                    {"type": "compaction", "compact_threshold": compact_threshold}
                ],
            )

        mock_trim.assert_called_once_with(
            messages=MESSAGES, model=MODEL, max_tokens=expected_target
        )
        assert msgs is trimmed_messages
        assert cm is None  # context_management must not be forwarded

    def test_compaction_entry_without_threshold_does_not_compact(self):
        """When compact_threshold is absent there is no target to trim to, so messages are left unchanged."""
        context_management = [{"type": "compaction"}]

        with patch(
            "litellm.responses.litellm_completion_transformation.transformation.trim_messages",
        ) as mock_trim:
            msgs, cm = LiteLLMCompletionResponsesConfig._apply_client_side_compaction(
                messages=MESSAGES,
                model=MODEL,
                context_management=context_management,
            )

        mock_trim.assert_not_called()
        assert msgs is MESSAGES
        assert cm is context_management

    def test_compact_target_ratio_is_25_percent(self):
        """_COMPACT_TARGET_RATIO should be 0.25 so trimming targets 25% of the threshold."""
        assert _COMPACT_TARGET_RATIO == 0.25

    def test_target_tokens_computed_correctly(self):
        """Target tokens = floor(compact_threshold * 0.25)."""
        compact_threshold = 8000
        expected_target = 2000  # 8000 * 0.25

        with patch(
            "litellm.responses.litellm_completion_transformation.transformation.trim_messages",
            return_value=MESSAGES,
        ) as mock_trim:
            LiteLLMCompletionResponsesConfig._apply_client_side_compaction(
                messages=MESSAGES,
                model=MODEL,
                context_management=[
                    {"type": "compaction", "compact_threshold": compact_threshold}
                ],
            )

        _, call_kwargs = mock_trim.call_args
        assert call_kwargs["max_tokens"] == expected_target

    def test_non_dict_entry_in_context_management_is_skipped(self):
        """Non-dict entries in context_management must not cause an error and are skipped."""
        context_management = ["not_a_dict", {"type": "something_else"}]

        msgs, cm = LiteLLMCompletionResponsesConfig._apply_client_side_compaction(
            messages=MESSAGES,
            model=MODEL,
            context_management=context_management,
        )

        assert msgs is MESSAGES
        assert cm is context_management
