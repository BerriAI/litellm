from litellm.llms.base_llm.base_utils import (
    map_developer_role_to_system_role,
    merge_system_message_contents,
)


class TestMergeSystemMessageContents:
    def test_strings_join_with_blank_line(self):
        assert merge_system_message_contents("System 1", "System 2") == "System 1\n\nSystem 2"

    def test_empty_strings_do_not_pad(self):
        assert merge_system_message_contents("", "System 2") == "System 2"
        assert merge_system_message_contents("System 1", "") == "System 1"
        assert merge_system_message_contents("", "") == ""

    def test_lists_concatenate(self):
        first = [{"type": "text", "text": "Part 1"}]
        second = [{"type": "text", "text": "Part 2"}]
        assert merge_system_message_contents(first, second) == [
            {"type": "text", "text": "Part 1"},
            {"type": "text", "text": "Part 2"},
        ]

    def test_str_and_list_normalize_to_blocks(self):
        text = "System preamble"
        blocks = [{"type": "text", "text": "System rules"}]
        assert merge_system_message_contents(text, blocks) == [
            {"type": "text", "text": "System preamble"},
            {"type": "text", "text": "System rules"},
        ]
        assert merge_system_message_contents(blocks, text) == [
            {"type": "text", "text": "System rules"},
            {"type": "text", "text": "System preamble"},
        ]

    def test_empty_str_and_list_drop_the_empty_side(self):
        blocks = [{"type": "text", "text": "System rules"}]
        assert merge_system_message_contents("", blocks) == blocks
        assert merge_system_message_contents(blocks, "") == blocks


class TestMapDeveloperRoleToSystemRole:
    def test_single_developer_message(self):
        messages = [
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        assert map_developer_role_to_system_role(messages) == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

    def test_consecutive_developer_messages_merge(self):
        messages = [
            {"role": "developer", "content": "Rule 1: Be concise."},
            {"role": "developer", "content": "Rule 2: Respond in Markdown."},
            {"role": "user", "content": "Hello"},
        ]
        assert map_developer_role_to_system_role(messages) == [
            {"role": "system", "content": "Rule 1: Be concise.\n\nRule 2: Respond in Markdown."},
            {"role": "user", "content": "Hello"},
        ]

    def test_system_followed_by_developer_merges(self):
        messages = [
            {"role": "system", "content": "Base instructions."},
            {"role": "developer", "content": "Developer override."},
            {"role": "user", "content": "Hello"},
        ]
        assert map_developer_role_to_system_role(messages) == [
            {"role": "system", "content": "Base instructions.\n\nDeveloper override."},
            {"role": "user", "content": "Hello"},
        ]

    def test_developer_followed_by_system_merges(self):
        messages = [
            {"role": "developer", "content": "Developer instruction."},
            {"role": "system", "content": "System instruction."},
            {"role": "user", "content": "Hello"},
        ]
        assert map_developer_role_to_system_role(messages) == [
            {"role": "system", "content": "Developer instruction.\n\nSystem instruction."},
            {"role": "user", "content": "Hello"},
        ]

    def test_consecutive_native_system_messages_merge(self):
        messages = [
            {"role": "system", "content": "System part 1."},
            {"role": "system", "content": "System part 2."},
            {"role": "user", "content": "Hello"},
        ]
        assert map_developer_role_to_system_role(messages) == [
            {"role": "system", "content": "System part 1.\n\nSystem part 2."},
            {"role": "user", "content": "Hello"},
        ]

    def test_merge_keeps_later_cache_control_when_both_set(self):
        messages = [
            {"role": "system", "content": "A", "cache_control": {"type": "ephemeral", "ttl": "5m"}},
            {"role": "system", "content": "B", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        ]
        assert map_developer_role_to_system_role(messages) == [
            {"role": "system", "content": "A\n\nB", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        ]

    def test_merge_keeps_cache_control_supplied_only_by_later_message(self):
        messages = [
            {"role": "system", "content": "A"},
            {"role": "developer", "content": "B", "cache_control": {"type": "ephemeral"}},
        ]
        assert map_developer_role_to_system_role(messages) == [
            {"role": "system", "content": "A\n\nB", "cache_control": {"type": "ephemeral"}},
        ]

    def test_billing_metadata_system_message_is_never_merged(self):
        billing = {"role": "system", "content": "x-anthropic-billing-header: cc_version=1.0;"}
        advisory = {"role": "system", "content": "Guardrail advisory: request was flagged"}
        assert map_developer_role_to_system_role([billing, advisory]) == [billing, advisory]
        assert map_developer_role_to_system_role([advisory, billing]) == [advisory, billing]

    def test_merge_preserves_cache_control_of_first_message(self):
        messages = [
            {"role": "system", "content": "Cached prompt", "cache_control": {"type": "ephemeral"}},
            {"role": "developer", "content": "Additional instructions"},
            {"role": "user", "content": "Hello"},
        ]
        assert map_developer_role_to_system_role(messages) == [
            {
                "role": "system",
                "content": "Cached prompt\n\nAdditional instructions",
                "cache_control": {"type": "ephemeral"},
            },
            {"role": "user", "content": "Hello"},
        ]

    def test_developer_message_after_a_user_turn_is_hoisted_into_the_leading_system_message(self):
        messages = [
            {"role": "system", "content": "System turn 1"},
            {"role": "user", "content": "User turn 1"},
            {"role": "developer", "content": "Developer turn 2"},
            {"role": "user", "content": "User turn 2"},
        ]
        assert map_developer_role_to_system_role(messages) == [
            {"role": "system", "content": "System turn 1\n\nDeveloper turn 2"},
            {"role": "user", "content": "User turn 1"},
            {"role": "user", "content": "User turn 2"},
        ]

    def test_developer_message_in_second_position_without_a_leading_system_message_is_hoisted(self):
        messages = [
            {"role": "user", "content": "Hi there"},
            {"role": "developer", "content": "Answer with exactly one word."},
            {"role": "user", "content": "What is the capital of France?"},
        ]
        assert map_developer_role_to_system_role(messages) == [
            {"role": "system", "content": "Answer with exactly one word."},
            {"role": "user", "content": "Hi there"},
            {"role": "user", "content": "What is the capital of France?"},
        ]

    def test_every_later_developer_message_is_hoisted_in_order_across_assistant_and_tool_turns(self):
        messages = [
            {"role": "system", "content": "Base"},
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "developer", "content": "Update A"},
            {"role": "user", "content": "Turn 2"},
            {"role": "developer", "content": "Update B"},
        ]
        assert map_developer_role_to_system_role(messages) == [
            {"role": "system", "content": "Base\n\nUpdate A\n\nUpdate B"},
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "Turn 2"},
        ]

    def test_hoisted_block_content_developer_message_merges_as_blocks_after_string_instructions(self):
        messages = [
            {"role": "system", "content": "You are Codex"},
            {"role": "user", "content": [{"type": "text", "text": "# AGENTS.md"}]},
            {"role": "developer", "content": [{"type": "text", "text": "<permissions instructions>"}]},
            {"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]},
        ]
        assert map_developer_role_to_system_role(messages) == [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Codex"},
                    {"type": "text", "text": "<permissions instructions>"},
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "# AGENTS.md"}]},
            {"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]},
        ]

    def test_non_consecutive_native_system_messages_stay_where_the_client_put_them(self):
        messages = [
            {"role": "system", "content": "System turn 1"},
            {"role": "user", "content": "User turn 1"},
            {"role": "system", "content": "System turn 2"},
            {"role": "user", "content": "User turn 2"},
        ]
        assert map_developer_role_to_system_role(messages) == messages

    def test_list_content_merge_preserves_block_level_cache_control(self):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Cached", "cache_control": {"type": "ephemeral"}}]},
            {"role": "developer", "content": [{"type": "text", "text": "Extra"}]},
            {"role": "user", "content": "Hello"},
        ]
        assert map_developer_role_to_system_role(messages) == [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Cached", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "Extra"},
                ],
            },
            {"role": "user", "content": "Hello"},
        ]
