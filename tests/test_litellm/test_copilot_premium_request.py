"""
Tests for GitHub Copilot X-Initiator header logic.

The X-Initiator header determines premium-request billing:
  - "user"  → counts as a new premium request
  - "agent" → free follow-up within the same turn

Only the *last* message/item should decide, not any message in the array.

Fixes: https://github.com/BerriAI/litellm/issues/18155
"""

import pytest

from litellm.llms.github_copilot.chat.transformation import GithubCopilotConfig
from litellm.llms.github_copilot.responses.transformation import (
    GithubCopilotResponsesAPIConfig,
)


class TestChatInitiator:
    """Tests for GithubCopilotConfig._determine_initiator."""

    def setup_method(self):
        self.config = GithubCopilotConfig()

    def test_single_user_message(self):
        msgs = [{"role": "user", "content": "Hello"}]
        assert self.config._determine_initiator(msgs) == "user"

    def test_single_assistant_message(self):
        msgs = [{"role": "assistant", "content": "Hi"}]
        assert self.config._determine_initiator(msgs) == "agent"

    def test_single_tool_message(self):
        msgs = [{"role": "tool", "content": "result", "tool_call_id": "tc1"}]
        assert self.config._determine_initiator(msgs) == "agent"

    def test_multi_turn_ending_with_user(self):
        """
        Key scenario: multi-turn conversation where user sends new message.
        Even though earlier messages include assistant/tool, the LAST message
        is from the user → should be "user" (premium request).
        """
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi, how can I help?"},
            {"role": "user", "content": "What's the weather?"},
        ]
        assert self.config._determine_initiator(msgs) == "user"

    def test_multi_turn_ending_with_assistant(self):
        """Agent continuation — last message is assistant."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Let me check..."},
        ]
        assert self.config._determine_initiator(msgs) == "agent"

    def test_multi_turn_ending_with_tool(self):
        """Tool result follow-up — should be "agent" (free)."""
        msgs = [
            {"role": "user", "content": "Run the function"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1"}]},
            {"role": "tool", "content": "42", "tool_call_id": "tc1"},
        ]
        assert self.config._determine_initiator(msgs) == "agent"

    def test_complex_agentic_loop_ending_with_user(self):
        """
        Realistic agentic loop: user → assistant → tool → assistant → user.
        Last message is user → premium request.
        """
        msgs = [
            {"role": "user", "content": "Plan my trip"},
            {"role": "assistant", "content": "Searching flights..."},
            {"role": "tool", "content": "Found 3 flights", "tool_call_id": "tc1"},
            {"role": "assistant", "content": "Here are 3 options..."},
            {"role": "user", "content": "Book option 2"},
        ]
        assert self.config._determine_initiator(msgs) == "user"

    def test_complex_agentic_loop_ending_with_tool(self):
        """
        Agentic follow-up: last message is tool result → free.
        """
        msgs = [
            {"role": "user", "content": "Book option 2"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc2"}]},
            {"role": "tool", "content": "Booked!", "tool_call_id": "tc2"},
        ]
        assert self.config._determine_initiator(msgs) == "agent"

    def test_empty_messages(self):
        assert self.config._determine_initiator([]) == "user"

    def test_system_only(self):
        msgs = [{"role": "system", "content": "You are helpful"}]
        assert self.config._determine_initiator(msgs) == "user"


class TestResponsesInitiator:
    """Tests for GithubCopilotResponsesAPIConfig._get_initiator."""

    def setup_method(self):
        self.config = GithubCopilotResponsesAPIConfig()

    def test_string_input(self):
        assert self.config._get_initiator("Hello") == "user"

    def test_empty_string(self):
        assert self.config._get_initiator("") == "user"

    def test_empty_list(self):
        assert self.config._get_initiator([]) == "user"

    def test_single_user_item(self):
        items = [{"role": "user", "content": "Hello"}]
        assert self.config._get_initiator(items) == "user"

    def test_single_assistant_item(self):
        items = [{"role": "assistant", "content": "Hi"}]
        assert self.config._get_initiator(items) == "agent"

    def test_item_without_role(self):
        items = [{"content": "some content"}]
        assert self.config._get_initiator(items) == "agent"

    def test_multi_items_ending_with_user(self):
        """Last item is user → premium request."""
        items = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "What's up?"},
        ]
        assert self.config._get_initiator(items) == "user"

    def test_multi_items_ending_with_assistant(self):
        items = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Checking..."},
        ]
        assert self.config._get_initiator(items) == "agent"

    def test_multi_items_ending_with_no_role(self):
        """Item without role at end → agent."""
        items = [
            {"role": "user", "content": "Hello"},
            {"content": "tool output"},
        ]
        assert self.config._get_initiator(items) == "agent"

    def test_non_dict_items_ignored(self):
        """Non-dict items don't affect classification."""
        items = ["string_item"]
        assert self.config._get_initiator(items) == "user"
