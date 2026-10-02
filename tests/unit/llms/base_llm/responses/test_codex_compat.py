"""Shared Codex wire-format normalization.

Both Bedrock endpoints reject the Codex *history* item types with
``400 Invalid 'input': value did not match any expected variant``. They are history
items, so they only appear from the second turn of a session onward — a first-turn
smoke test passes and hides the problem entirely.
"""

import json

import pytest

from litellm.llms.base_llm.responses.codex_compat import normalize_codex_input_items

USER = {"role": "user", "content": "hi"}


class TestAgentMessage:
    def test_becomes_an_assistant_message(self):
        item = {
            "type": "agent_message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "prior turn"}],
        }
        out, types = normalize_codex_input_items([item, USER])
        assert types == ("agent_message",)
        assert out[0] == {
            "type": "message",
            "role": "assistant",
            "content": ({"type": "output_text", "text": "prior turn"},),
        }

    def test_encrypted_content_slot_is_used_as_text(self):
        """Codex puts the plaintext payload there when the model issued no encrypted args."""
        item = {"type": "agent_message", "content": [{"encrypted_content": "plain"}]}
        out, _ = normalize_codex_input_items([item, USER])
        assert out[0]["content"] == ({"type": "output_text", "text": "plain"},)

    def test_non_list_content_yields_no_text_and_drops_the_item(self):
        out, types = normalize_codex_input_items([{"type": "agent_message", "content": "not a list"}, USER])
        assert out == [USER]
        assert types == ("agent_message",)

    def test_textless_item_is_dropped(self):
        out, types = normalize_codex_input_items([{"type": "agent_message", "content": []}, USER])
        assert out == [USER]
        assert types == ("agent_message",)


class TestContextCompaction:
    def test_becomes_compaction(self):
        out, types = normalize_codex_input_items([{"type": "context_compaction", "encrypted_content": "abc"}, USER])
        assert out[0] == {"type": "compaction", "encrypted_content": "abc"}
        assert types == ("context_compaction",)

    @pytest.mark.parametrize("bad", [{}, {"encrypted_content": ""}, {"encrypted_content": 7}])
    def test_without_usable_content_is_dropped(self, bad):
        out, _ = normalize_codex_input_items([{"type": "context_compaction", **bad}, USER])
        assert out == [USER]


class TestLocalShellCall:
    def test_becomes_the_function_call_its_output_pairs_with(self):
        out, types = normalize_codex_input_items(
            [{"type": "local_shell_call", "call_id": "c1", "action": {"command": ["ls"]}}, USER]
        )
        assert out[0] == {
            "type": "function_call",
            "call_id": "c1",
            "name": "local_shell",
            "arguments": json.dumps({"command": ["ls"]}),
        }
        assert types == ("local_shell_call",)

    def test_missing_action_yields_empty_arguments(self):
        out, _ = normalize_codex_input_items([{"type": "local_shell_call", "call_id": "c1"}, USER])
        assert out[0]["arguments"] == "{}"

    def test_without_call_id_is_dropped(self):
        out, _ = normalize_codex_input_items([{"type": "local_shell_call"}, USER])
        assert out == [USER]


class TestPassthroughAndShape:
    def test_string_input_untouched(self):
        assert normalize_codex_input_items("just a prompt") == ("just a prompt", ())

    def test_unrelated_items_untouched_and_no_types_reported(self):
        items = [USER, {"type": "message", "role": "assistant", "content": []}]
        out, types = normalize_codex_input_items(items)
        assert out == items
        assert types == ()

    def test_non_mapping_entries_pass_through_except_a_literal_none(self):
        """A literal ``None`` is indistinguishable from "drop this item" in the
        per-item return protocol, so it is dropped. Other non-mapping entries pass
        through untouched. This matches the behaviour before the normalizer moved
        out of the bedrock_mantle config."""
        out, types = normalize_codex_input_items(["a string", 42, None, USER])
        assert out == ["a string", 42, USER]
        assert types == ()

    def test_types_are_sorted_and_deduplicated(self):
        items = [
            {"type": "local_shell_call", "call_id": "c1"},
            {"type": "agent_message", "content": [{"text": "x"}]},
            {"type": "local_shell_call", "call_id": "c2"},
        ]
        _, types = normalize_codex_input_items(items)
        assert types == ("agent_message", "local_shell_call")

    def test_returns_a_list_not_a_tuple(self):
        """The input->messages conversion downstream narrows on isinstance(input, list);
        a tuple silently yields zero messages and the provider rejects the request."""
        out, _ = normalize_codex_input_items([{"type": "agent_message", "content": [{"text": "x"}]}, USER])
        assert isinstance(out, list)
