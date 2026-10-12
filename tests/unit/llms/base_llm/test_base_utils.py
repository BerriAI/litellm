from typing import Final

from litellm.llms.base_llm.base_utils import map_developer_role_to_system_role


def test_developer_message_keeps_its_cache_control_when_mapped_to_system():
    cache_control: Final = {"type": "ephemeral"}
    messages: Final = map_developer_role_to_system_role(
        [
            {"role": "developer", "content": "Answer in one word.", "cache_control": cache_control},
            {"role": "user", "content": "Which color is the sky?"},
        ]
    )
    assert messages == [
        {"role": "system", "content": "Answer in one word.", "cache_control": cache_control},
        {"role": "user", "content": "Which color is the sky?"},
    ]
