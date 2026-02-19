"""
Tests for the fix to AnthropicCacheControlHook._safe_insert_cache_control_in_message.

Background
----------
Previously, when a message had plain-string ``content``, the hook placed
``cache_control`` as a **top-level key** on the message dict::

    {"role": "system", "content": "...", "cache_control": {"type": "ephemeral"}}

Only Anthropic's own transformer handles this placement (it explicitly checks
``if "cache_control" in system_message_block`` for string-content messages).

Vertex AI and Bedrock Invoke API both rely on ``litellm.utils.is_cached_message``
to detect whether a message needs caching.  That utility returns ``False``
immediately when ``content`` is not a list, so caching was silently skipped
for those providers.

Fix
---
When ``content`` is a string, the hook now converts it to a single-item
content-block list and places ``cache_control`` **inside** the block::

    {"role": "system",
     "content": [{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}]}

This is the universal format understood by all providers.

Backward-compatibility proof
-----------------------------
* List ``content`` — unchanged behaviour (cache_control goes into last block).
* Anthropic transformer — already handles ``isinstance(content, list)`` branch.
* ``litellm.utils.is_cached_message`` — returns ``True`` for list content with
  ``cache_control`` inside a block.
* All other LiteLLM consumers that previously read ``message["cache_control"]``
  also check ``isinstance(content, list)`` as a fallback
  (``anthropic/common_utils.py``, ``router_utils/prompt_caching_cache.py``).
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.abspath("../../.."))

import pytest

from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook
from litellm.types.llms.openai import ChatCompletionCachedContent
from litellm.utils import is_cached_message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONTROL = ChatCompletionCachedContent(type="ephemeral")


def _make_hook() -> AnthropicCacheControlHook:
    return AnthropicCacheControlHook()


# ===========================================================================
# _safe_insert_cache_control_in_message — unit tests
# ===========================================================================


class TestSafeInsertStringContent:
    """
    When message content is a plain string the hook must convert it to a
    single-item content-block list and place cache_control *inside* the block.
    """

    def test_string_content_converted_to_list(self):
        msg = {"role": "system", "content": "You are a helpful assistant."}
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, CONTROL
        )
        assert isinstance(result["content"], list), (
            "content must be converted from str to list"
        )

    def test_string_content_cache_control_inside_block(self):
        msg = {"role": "system", "content": "You are a helpful assistant."}
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, CONTROL
        )
        block = result["content"][0]
        assert "cache_control" in block, (
            "cache_control must be inside the content block, not at message top level"
        )
        assert block["cache_control"] == {"type": "ephemeral"}

    def test_string_content_text_preserved(self):
        text = "You are a helpful assistant."
        msg = {"role": "system", "content": text}
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, CONTROL
        )
        assert result["content"][0]["text"] == text

    def test_string_content_type_is_text(self):
        msg = {"role": "system", "content": "Instructions."}
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, CONTROL
        )
        assert result["content"][0]["type"] == "text"

    def test_string_content_no_top_level_cache_control(self):
        """
        After the fix, cache_control must NOT appear at the message top level.
        The old behaviour placed it there for string content; only Anthropic's
        transformer handled it — Vertex AI and Bedrock silently ignored it.
        """
        msg = {"role": "system", "content": "Instructions."}
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, CONTROL
        )
        assert "cache_control" not in result, (
            "cache_control must NOT be a top-level key on the message dict"
        )

    def test_user_message_string_content(self):
        """Works for any role, not just system."""
        msg = {"role": "user", "content": "Long user context to cache."}
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, CONTROL
        )
        assert isinstance(result["content"], list)
        assert result["content"][-1]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in result

    def test_custom_control_type(self):
        """Custom control values are preserved inside the block."""
        custom = ChatCompletionCachedContent(type="ephemeral")
        msg = {"role": "system", "content": "Some text."}
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, custom
        )
        assert result["content"][0]["cache_control"] == custom


class TestSafeInsertListContent:
    """
    When message content is already a list the existing behaviour must be
    preserved: cache_control goes into the *last* block only.
    """

    def test_list_content_last_block_gets_cache_control(self):
        msg = {
            "role": "system",
            "content": [
                {"type": "text", "text": "Block 1"},
                {"type": "text", "text": "Block 2"},
            ],
        }
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, CONTROL
        )
        assert "cache_control" not in result["content"][0], (
            "First block must NOT get cache_control"
        )
        assert result["content"][-1]["cache_control"] == {"type": "ephemeral"}, (
            "Last block must get cache_control"
        )

    def test_list_content_no_top_level_cache_control(self):
        msg = {
            "role": "user",
            "content": [{"type": "text", "text": "Some text"}],
        }
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, CONTROL
        )
        assert "cache_control" not in result

    def test_list_content_single_block(self):
        msg = {"role": "user", "content": [{"type": "text", "text": "Only block"}]}
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, CONTROL
        )
        assert result["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_list_content_unchanged_non_last_blocks(self):
        original_first = {"type": "text", "text": "First"}
        msg = {
            "role": "user",
            "content": [dict(original_first), {"type": "text", "text": "Last"}],
        }
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, CONTROL
        )
        assert result["content"][0] == original_first


class TestSafeInsertEdgeCases:
    def test_none_content_is_not_modified(self):
        msg = {"role": "user", "content": None}
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, CONTROL
        )
        assert result["content"] is None
        assert "cache_control" not in result

    def test_empty_list_content_is_not_modified(self):
        msg = {"role": "user", "content": []}
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, CONTROL
        )
        assert result["content"] == []
        assert "cache_control" not in result

    def test_returns_same_message_object(self):
        """The method modifies in place and returns the same object."""
        msg = {"role": "system", "content": "text"}
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, CONTROL
        )
        assert result is msg


# ===========================================================================
# is_cached_message gate — critical for Vertex AI / Bedrock Invoke API
# ===========================================================================


class TestIsCachedMessageCompatibility:
    """
    ``litellm.utils.is_cached_message()`` is the gate used by the Vertex AI
    and Bedrock Invoke API transformers.  It returns ``False`` when ``content``
    is not a list.  After the fix, injecting into string-content messages must
    produce list content that passes this gate.
    """

    def test_string_content_after_injection_passes_gate(self):
        msg = {"role": "system", "content": "Long static system prompt."}
        injected = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, CONTROL
        )
        assert is_cached_message(injected), (
            "is_cached_message() must return True after injection into string content — "
            "this is the gate used by Vertex AI and Bedrock Invoke API transformers"
        )

    def test_list_content_after_injection_passes_gate(self):
        msg = {
            "role": "system",
            "content": [{"type": "text", "text": "Static prompt."}],
        }
        injected = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, CONTROL
        )
        assert is_cached_message(injected)

    def test_no_injection_fails_gate(self):
        """Sanity check: messages without cache_control must not pass the gate."""
        msg = {"role": "system", "content": "No caching here."}
        assert not is_cached_message(msg)

    def test_old_behaviour_top_level_fails_gate(self):
        """
        Documents the OLD (broken) behaviour: top-level cache_control on a
        string-content message is NOT recognised by is_cached_message() because
        it requires content to be a list.
        """
        msg = {
            "role": "system",
            "content": "Long prompt.",
            "cache_control": {"type": "ephemeral"},  # old top-level placement
        }
        # is_cached_message returns False — this is the ROOT CAUSE of the bug
        assert not is_cached_message(msg), (
            "is_cached_message() must return False for top-level cache_control "
            "on string content — this documents why the old behaviour was broken "
            "for Vertex AI and Bedrock Invoke API"
        )


# ===========================================================================
# Anthropic transformer backward-compat
# ===========================================================================


class TestAnthropicTransformerBackwardCompat:
    """
    Verify that the Anthropic transformer correctly handles messages after the
    fix (list content with cache_control inside the block).
    """

    def test_anthropic_translate_system_message_with_list_content(self):
        """
        The Anthropic transformer's translate_system_message must correctly
        extract cache_control from a list-content system message produced by
        the hook after the fix.
        """
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        config = AnthropicConfig()
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": "Hello"},
        ]
        system_msgs = config.translate_system_message(messages=messages)
        assert len(system_msgs) == 1
        assert system_msgs[0]["type"] == "text"
        assert system_msgs[0]["text"] == "You are a helpful assistant."
        assert system_msgs[0].get("cache_control") == {"type": "ephemeral"}, (
            "cache_control must be preserved by the Anthropic transformer "
            "when content is a list"
        )

    def test_anthropic_translate_system_message_with_string_content(self):
        """
        Existing Anthropic transformer behaviour for plain-string system
        messages (no cache_control) must be unchanged.
        """
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        config = AnthropicConfig()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        system_msgs = config.translate_system_message(messages=messages)
        assert len(system_msgs) == 1
        assert system_msgs[0]["text"] == "You are a helpful assistant."
        assert "cache_control" not in system_msgs[0]


# ===========================================================================
# _process_message_injection — role-targeting integration
# ===========================================================================


class TestProcessMessageInjection:
    """
    End-to-end tests of _process_message_injection using the fixed
    _safe_insert_cache_control_in_message.
    """

    def _inject(self, messages, role=None, index=None):
        point = {"location": "message"}
        if role is not None:
            point["role"] = role
        if index is not None:
            point["index"] = index
        return AnthropicCacheControlHook._process_message_injection(
            point=point, messages=messages
        )

    def test_role_injection_on_string_system_message(self):
        messages = [
            {"role": "system", "content": "Static system instructions."},
            {"role": "user", "content": "Question"},
        ]
        result = self._inject(messages, role="system")
        sys_msg = next(m for m in result if m["role"] == "system")
        assert isinstance(sys_msg["content"], list)
        assert sys_msg["content"][-1]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in sys_msg

    def test_index_injection_on_string_user_message(self):
        messages = [
            {"role": "user", "content": "Long document to cache."},
        ]
        result = self._inject(messages, index=-1)
        user_msg = result[0]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_index_injection_on_list_message_unchanged(self):
        """Existing list-content path must continue to work as before."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Block 1"},
                    {"type": "text", "text": "Block 2"},
                ],
            }
        ]
        result = self._inject(messages, index=0)
        assert "cache_control" not in result[0]["content"][0]
        assert result[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_injected_string_message_passes_is_cached_message(self):
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Hi"},
        ]
        result = self._inject(messages, role="system")
        sys_msg = next(m for m in result if m["role"] == "system")
        assert is_cached_message(sys_msg), (
            "After injection, is_cached_message() must return True — "
            "this is the gate used by Vertex AI and Bedrock Invoke API"
        )
