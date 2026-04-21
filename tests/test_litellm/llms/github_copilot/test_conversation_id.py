"""
Unit tests for conversation ID store in common_utils.py.

Tests the get_or_create_conversation_id() function and determine_x_initiator()
helper added in Phase 2 (FIX-03, FIX-04).
"""

import re
import threading

import pytest

from litellm.llms.github_copilot import common_utils
from litellm.llms.github_copilot.common_utils import (
    determine_x_initiator,
    get_or_create_conversation_id,
)


class TestGetOrCreateConversationId:
    """Tests for the conversation ID store."""

    def setup_method(self):
        """Reset conversation store before each test."""
        with common_utils._conversation_store_lock:
            common_utils._conversation_store.clear()

    def test_same_key_returns_same_id(self):
        """Same conversation_key must return same UUID across calls."""
        id1 = get_or_create_conversation_id("user-123:session-abc")
        id2 = get_or_create_conversation_id("user-123:session-abc")
        assert id1 == id2

    def test_different_keys_return_different_ids(self):
        """Different conversation_keys must produce different UUIDs."""
        id1 = get_or_create_conversation_id("user-123:session-abc")
        id2 = get_or_create_conversation_id("user-456:session-xyz")
        assert id1 != id2

    def test_returned_id_is_valid_uuid_string(self):
        """Returned conversation_id must be a valid UUID string."""
        conversation_id = get_or_create_conversation_id("test-key")
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        assert uuid_pattern.match(
            conversation_id
        ), f"Expected UUID format, got: {conversation_id}"

    def test_same_key_hits_lru_cache(self):
        """Second call with same key must return same ID (exercises move_to_end)."""
        id1 = get_or_create_conversation_id("lru-key")
        id2 = get_or_create_conversation_id("lru-key")
        assert id1 == id2

    def test_eviction_when_store_exceeds_max(self):
        """Oldest entry must be evicted when store exceeds _MAX_CONVERSATION_STORE."""
        original_max = common_utils._MAX_CONVERSATION_STORE
        try:
            common_utils._MAX_CONVERSATION_STORE = 3
            get_or_create_conversation_id("key-1")
            get_or_create_conversation_id("key-2")
            get_or_create_conversation_id("key-3")
            # Adding a 4th entry should evict key-1
            get_or_create_conversation_id("key-4")
            with common_utils._conversation_store_lock:
                assert "key-1" not in common_utils._conversation_store
                assert len(common_utils._conversation_store) == 3
        finally:
            common_utils._MAX_CONVERSATION_STORE = original_max

    def test_thread_safe_concurrent_access(self):
        """Concurrent access with same key must return same ID (no race condition)."""
        results = []
        errors = []

        def get_id():
            try:
                results.append(get_or_create_conversation_id("shared-key"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_id) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent access: {errors}"
        assert (
            len(set(results)) == 1
        ), f"Expected 1 unique ID, got {len(set(results))}: {set(results)}"


class TestDetermineXInitiator:
    """Tests for the shared determine_x_initiator() helper."""

    def test_user_only_messages_return_user(self):
        messages = [{"role": "user", "content": "Hello"}]
        assert determine_x_initiator(messages) == "user"

    def test_system_only_messages_return_user(self):
        messages = [{"role": "system", "content": "You are helpful"}]
        assert determine_x_initiator(messages) == "user"

    def test_assistant_role_returns_agent(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Follow up"},
        ]
        assert determine_x_initiator(messages) == "agent"

    def test_tool_role_returns_agent(self):
        messages = [
            {"role": "user", "content": "Call a tool"},
            {"role": "tool", "content": "result", "tool_call_id": "tc_1"},
        ]
        assert determine_x_initiator(messages) == "agent"

    def test_no_role_item_returns_agent(self):
        """Items without 'role' key (Responses API types) must return 'agent'."""
        input_items = [
            {"role": "user", "content": "Call function"},
            {"type": "function_call", "name": "my_fn", "arguments": "{}"},
        ]
        assert determine_x_initiator(input_items) == "agent"

    def test_string_input_returns_user(self):
        assert determine_x_initiator("What is 2+2?") == "user"

    def test_empty_list_returns_user(self):
        assert determine_x_initiator([]) == "user"
