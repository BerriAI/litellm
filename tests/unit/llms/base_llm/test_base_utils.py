from typing import Final

import pytest

from litellm.exceptions import BadRequestError
from litellm.llms.base_llm.base_utils import map_developer_role_to_system_role
from litellm.types.llms.openai import AllMessageValues


@pytest.mark.parametrize("index", [0, 2])
def test_a_developer_message_without_content_is_rejected_naming_its_index(index: int) -> None:
    conversation: Final[list[AllMessageValues]] = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ]
    messages: Final = [*conversation[:index], {"role": "developer"}, *conversation[index:]]

    with pytest.raises(BadRequestError) as rejected:
        map_developer_role_to_system_role(messages)  # pyright: ignore[reportArgumentType]  # the malformed message under test

    assert rejected.value.status_code == 400
    assert rejected.value.message.endswith(f"Missing required parameter: 'messages[{index}].content'.")


def test_only_developer_messages_need_content() -> None:
    tool_call_turn: Final[AllMessageValues] = {
        "role": "assistant",
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
    }
    tool_result: Final[AllMessageValues] = {"role": "tool", "tool_call_id": "call_1", "content": "Paris"}

    assert map_developer_role_to_system_role(
        [tool_call_turn, tool_result, {"role": "developer", "content": "Answer in one word."}]
    ) == [tool_call_turn, tool_result, {"role": "system", "content": "Answer in one word."}]
