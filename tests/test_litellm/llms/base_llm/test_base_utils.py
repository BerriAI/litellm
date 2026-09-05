from litellm.llms.base_llm.base_utils import (
    hoist_developer_messages_into_leading_system_message,
    map_developer_role_to_system_role,
)


class TestMapDeveloperRoleToSystemRole:
    def test_developer_messages_become_system_messages_where_the_client_put_them(self):
        messages = [
            {"role": "developer", "content": "Rule 1"},
            {"role": "user", "content": "Hi there"},
            {"role": "developer", "content": "Rule 2", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "Hello"},
        ]
        assert list(map_developer_role_to_system_role(messages)) == [
            {"role": "system", "content": "Rule 1"},
            {"role": "user", "content": "Hi there"},
            {"role": "system", "content": "Rule 2", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "Hello"},
        ]

    def test_consecutive_system_messages_stay_separate(self):
        messages = [
            {"role": "system", "content": "Base instructions."},
            {"role": "developer", "content": "Developer override."},
            {"role": "user", "content": "Hello"},
        ]
        assert list(map_developer_role_to_system_role(messages)) == [
            {"role": "system", "content": "Base instructions."},
            {"role": "system", "content": "Developer override."},
            {"role": "user", "content": "Hello"},
        ]


class TestHoistDeveloperMessagesIntoLeadingSystemMessage:
    def test_single_developer_message(self):
        messages = [
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        assert hoist_developer_messages_into_leading_system_message(messages) == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

    def test_consecutive_developer_messages_merge(self):
        messages = [
            {"role": "developer", "content": "Rule 1: Be concise."},
            {"role": "developer", "content": "Rule 2: Respond in Markdown."},
            {"role": "user", "content": "Hello"},
        ]
        assert hoist_developer_messages_into_leading_system_message(messages) == [
            {"role": "system", "content": "Rule 1: Be concise.\n\nRule 2: Respond in Markdown."},
            {"role": "user", "content": "Hello"},
        ]

    def test_system_followed_by_developer_merges(self):
        messages = [
            {"role": "system", "content": "Base instructions."},
            {"role": "developer", "content": "Developer override."},
            {"role": "user", "content": "Hello"},
        ]
        assert hoist_developer_messages_into_leading_system_message(messages) == [
            {"role": "system", "content": "Base instructions.\n\nDeveloper override."},
            {"role": "user", "content": "Hello"},
        ]

    def test_developer_followed_by_system_merges(self):
        messages = [
            {"role": "developer", "content": "Developer instruction."},
            {"role": "system", "content": "System instruction."},
            {"role": "user", "content": "Hello"},
        ]
        assert hoist_developer_messages_into_leading_system_message(messages) == [
            {"role": "system", "content": "Developer instruction.\n\nSystem instruction."},
            {"role": "user", "content": "Hello"},
        ]

    def test_consecutive_native_system_messages_merge(self):
        messages = [
            {"role": "system", "content": "System part 1."},
            {"role": "system", "content": "System part 2."},
            {"role": "user", "content": "Hello"},
        ]
        assert hoist_developer_messages_into_leading_system_message(messages) == [
            {"role": "system", "content": "System part 1.\n\nSystem part 2."},
            {"role": "user", "content": "Hello"},
        ]

    def test_empty_string_side_does_not_pad(self):
        assert hoist_developer_messages_into_leading_system_message(
            [{"role": "system", "content": ""}, {"role": "developer", "content": "Rules"}]
        ) == [{"role": "system", "content": "Rules"}]
        assert hoist_developer_messages_into_leading_system_message(
            [{"role": "system", "content": "Rules"}, {"role": "developer", "content": ""}]
        ) == [{"role": "system", "content": "Rules"}]

    def test_two_cached_string_messages_keep_one_breakpoint_each(self):
        messages = [
            {"role": "system", "content": "A", "cache_control": {"type": "ephemeral", "ttl": "5m"}},
            {"role": "system", "content": "B", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        ]
        assert hoist_developer_messages_into_leading_system_message(messages) == [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "A", "cache_control": {"type": "ephemeral", "ttl": "5m"}},
                    {"type": "text", "text": "B", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                ],
            },
        ]

    def test_cache_control_supplied_only_by_the_later_message_marks_only_its_block(self):
        messages = [
            {"role": "system", "content": "A"},
            {"role": "developer", "content": "B", "cache_control": {"type": "ephemeral"}},
        ]
        assert hoist_developer_messages_into_leading_system_message(messages) == [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "A"},
                    {"type": "text", "text": "B", "cache_control": {"type": "ephemeral"}},
                ],
            },
        ]

    def test_cache_control_of_the_first_message_stays_on_its_own_block(self):
        messages = [
            {"role": "system", "content": "Cached prompt", "cache_control": {"type": "ephemeral"}},
            {"role": "developer", "content": "Additional instructions"},
            {"role": "user", "content": "Hello"},
        ]
        assert hoist_developer_messages_into_leading_system_message(messages) == [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Cached prompt", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "Additional instructions"},
                ],
            },
            {"role": "user", "content": "Hello"},
        ]

    def test_cached_string_message_merged_with_block_content_keeps_its_breakpoint(self):
        messages = [
            {"role": "system", "content": "Cached prompt", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "Hi there"},
            {"role": "developer", "content": [{"type": "text", "text": "Answer with exactly one word."}]},
            {"role": "user", "content": "What is the capital of France?"},
        ]
        assert hoist_developer_messages_into_leading_system_message(messages) == [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Cached prompt", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "Answer with exactly one word."},
                ],
            },
            {"role": "user", "content": "Hi there"},
            {"role": "user", "content": "What is the capital of France?"},
        ]

    def test_billing_metadata_keeps_its_own_block_when_merged(self):
        billing = {"role": "system", "content": "x-anthropic-billing-header: cc_version=1.0;"}
        advisory = {"role": "system", "content": "Guardrail advisory: request was flagged"}
        assert hoist_developer_messages_into_leading_system_message([billing, advisory]) == [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "x-anthropic-billing-header: cc_version=1.0;"},
                    {"type": "text", "text": "Guardrail advisory: request was flagged"},
                ],
            }
        ]
        assert hoist_developer_messages_into_leading_system_message([advisory, billing]) == [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Guardrail advisory: request was flagged"},
                    {"type": "text", "text": "x-anthropic-billing-header: cc_version=1.0;"},
                ],
            }
        ]

    def test_developer_message_after_a_user_turn_is_hoisted_into_the_leading_system_message(self):
        messages = [
            {"role": "system", "content": "System turn 1"},
            {"role": "user", "content": "User turn 1"},
            {"role": "developer", "content": "Developer turn 2"},
            {"role": "user", "content": "User turn 2"},
        ]
        assert hoist_developer_messages_into_leading_system_message(messages) == [
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
        assert hoist_developer_messages_into_leading_system_message(messages) == [
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
        assert hoist_developer_messages_into_leading_system_message(messages) == [
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
        assert hoist_developer_messages_into_leading_system_message(messages) == [
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
        assert hoist_developer_messages_into_leading_system_message(messages) == messages

    def test_list_content_merge_preserves_block_level_cache_control(self):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Cached", "cache_control": {"type": "ephemeral"}}]},
            {"role": "developer", "content": [{"type": "text", "text": "Extra"}]},
            {"role": "user", "content": "Hello"},
        ]
        assert hoist_developer_messages_into_leading_system_message(messages) == [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Cached", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "Extra"},
                ],
            },
            {"role": "user", "content": "Hello"},
        ]
