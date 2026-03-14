import os
import sys
from unittest.mock import patch

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
    _COMPACT_TARGET_RATIO,
)


# Single-message list used by tests that only need a minimal input.
MESSAGES = [{"role": "user", "content": "Hello"}]
MODEL = "gpt-4o"

# Multi-turn history used by tests that need a non-trivial message list.
HISTORY = [
    {"role": "user", "content": "Previous question"},
    {"role": "assistant", "content": "Previous answer"},
]
LATEST = {"role": "user", "content": "Current question"}
MULTI_MESSAGES = HISTORY + [LATEST]


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
        """When token count exceeds compact_threshold, trim_messages is called with history only."""
        compact_threshold = 10000
        latest_tokens = 5
        expected_history_target = int(compact_threshold * _COMPACT_TARGET_RATIO) - latest_tokens
        trimmed_history = [{"role": "user", "content": "trimmed"}]

        with patch(
            "litellm.responses.litellm_completion_transformation.transformation.cheap_token_counter",
            side_effect=[compact_threshold + 1, latest_tokens],
        ), patch(
            "litellm.responses.litellm_completion_transformation.transformation.trim_messages",
            return_value=trimmed_history,
        ) as mock_trim:
            msgs, cm = LiteLLMCompletionResponsesConfig._apply_client_side_compaction(
                messages=MULTI_MESSAGES,
                model=MODEL,
                context_management=[
                    {"type": "compaction", "compact_threshold": compact_threshold}
                ],
            )

        mock_trim.assert_called_once_with(
            messages=HISTORY, model=MODEL, max_tokens=expected_history_target
        )
        assert msgs == trimmed_history + [LATEST]
        assert cm is None  # context_management must not be forwarded

    def test_below_threshold_does_not_compact(self):
        """When token count is below compact_threshold, messages are returned unchanged."""
        compact_threshold = 10000
        context_management = [{"type": "compaction", "compact_threshold": compact_threshold}]

        with patch(
            "litellm.responses.litellm_completion_transformation.transformation.cheap_token_counter",
            return_value=compact_threshold - 1,
        ), patch(
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
        """History target = floor(compact_threshold * 0.25) - latest_message_tokens."""
        compact_threshold = 8000
        latest_tokens = 10
        expected_history_target = 2000 - latest_tokens  # 8000 * 0.25 - 10 = 1990

        with patch(
            "litellm.responses.litellm_completion_transformation.transformation.cheap_token_counter",
            side_effect=[compact_threshold + 1, latest_tokens],
        ), patch(
            "litellm.responses.litellm_completion_transformation.transformation.trim_messages",
            return_value=HISTORY,
        ) as mock_trim:
            LiteLLMCompletionResponsesConfig._apply_client_side_compaction(
                messages=MULTI_MESSAGES,
                model=MODEL,
                context_management=[
                    {"type": "compaction", "compact_threshold": compact_threshold}
                ],
            )

        _, call_kwargs = mock_trim.call_args
        assert call_kwargs["max_tokens"] == expected_history_target

    def test_empty_trim_result_preserves_latest_message(self):
        """When trim_messages returns [] for history, the latest message is still forwarded."""
        compact_threshold = 10000
        latest_tokens = 5

        with patch(
            "litellm.responses.litellm_completion_transformation.transformation.cheap_token_counter",
            side_effect=[compact_threshold + 1, latest_tokens],
        ), patch(
            "litellm.responses.litellm_completion_transformation.transformation.trim_messages",
            return_value=[],
        ):
            msgs, cm = LiteLLMCompletionResponsesConfig._apply_client_side_compaction(
                messages=MULTI_MESSAGES,
                model=MODEL,
                context_management=[
                    {"type": "compaction", "compact_threshold": compact_threshold}
                ],
            )

        assert msgs == [LATEST]
        assert cm is None  # compaction was applied

    def test_latest_message_always_preserved_after_compaction(self):
        """The current-turn (last) message must survive compaction regardless of token pressure."""
        compact_threshold = 10000
        latest_tokens = 5
        trimmed_history = [{"role": "user", "content": "kept old message"}]

        with patch(
            "litellm.responses.litellm_completion_transformation.transformation.cheap_token_counter",
            side_effect=[compact_threshold + 1, latest_tokens],
        ), patch(
            "litellm.responses.litellm_completion_transformation.transformation.trim_messages",
            return_value=trimmed_history,
        ):
            msgs, cm = LiteLLMCompletionResponsesConfig._apply_client_side_compaction(
                messages=MULTI_MESSAGES,
                model=MODEL,
                context_management=[
                    {"type": "compaction", "compact_threshold": compact_threshold}
                ],
            )

        assert msgs[-1] is LATEST
        assert cm is None

    def test_trim_is_called_with_history_not_full_messages(self):
        """trim_messages must receive only history (messages[:-1]), not the full list."""
        compact_threshold = 10000
        latest_tokens = 5

        with patch(
            "litellm.responses.litellm_completion_transformation.transformation.cheap_token_counter",
            side_effect=[compact_threshold + 1, latest_tokens],
        ), patch(
            "litellm.responses.litellm_completion_transformation.transformation.trim_messages",
            return_value=HISTORY,
        ) as mock_trim:
            LiteLLMCompletionResponsesConfig._apply_client_side_compaction(
                messages=MULTI_MESSAGES,
                model=MODEL,
                context_management=[
                    {"type": "compaction", "compact_threshold": compact_threshold}
                ],
            )

        call_kwargs = mock_trim.call_args.kwargs
        assert call_kwargs["messages"] == HISTORY
        assert LATEST not in call_kwargs["messages"]

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
