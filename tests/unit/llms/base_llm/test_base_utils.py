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
        map_developer_role_to_system_role(
            messages,  # pyright: ignore[reportArgumentType]  # the malformed message under test
            model="kimi-k3p1-pro",
            llm_provider="fireworks_ai",
        )

    assert rejected.value.status_code == 400
    assert rejected.value.message.endswith(f"Missing required parameter: 'messages[{index}].content'.")
    assert rejected.value.param == f"messages[{index}].content"
    assert (rejected.value.llm_provider, rejected.value.model) == ("fireworks_ai", "kimi-k3p1-pro")


def test_only_developer_messages_need_content() -> None:
    tool_call_turn: Final[AllMessageValues] = {
        "role": "assistant",
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
    }
    tool_result: Final[AllMessageValues] = {"role": "tool", "tool_call_id": "call_1", "content": "Paris"}

    assert map_developer_role_to_system_role(
        [tool_call_turn, tool_result, {"role": "developer", "content": "Answer in one word."}],
        model="kimi-k3p1-pro",
        llm_provider="fireworks_ai",
    ) == [tool_call_turn, tool_result, {"role": "system", "content": "Answer in one word."}]


def test_a_batch_line_developer_message_with_null_content_maps_through() -> None:
    assert map_developer_role_to_system_role(
        [
            {"role": "developer", "content": None},  # pyright: ignore[reportAssignmentType]  # the null a batch line can carry
            {"role": "user", "content": "What is the capital of France?"},
        ],
        model="gemini-3.5-pro",
        llm_provider="vertex_ai",
    ) == [{"role": "system", "content": None}, {"role": "user", "content": "What is the capital of France?"}]
