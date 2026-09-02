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

    def test_non_consecutive_system_messages_stay_separate(self):
        messages = [
            {"role": "system", "content": "System turn 1"},
            {"role": "user", "content": "User turn 1"},
            {"role": "developer", "content": "Developer turn 2"},
            {"role": "user", "content": "User turn 2"},
        ]
        assert map_developer_role_to_system_role(messages) == [
            {"role": "system", "content": "System turn 1"},
            {"role": "user", "content": "User turn 1"},
            {"role": "system", "content": "Developer turn 2"},
            {"role": "user", "content": "User turn 2"},
        ]

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
