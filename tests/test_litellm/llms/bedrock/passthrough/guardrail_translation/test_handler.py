from copy import deepcopy
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.llms.bedrock.passthrough.guardrail_translation.handler import (
    BedrockPassthroughGuardrailHandler,
)


@pytest.mark.asyncio
async def test_skip_assistant_preserves_converse_history_and_masks_user() -> None:
    assistant: Final = {
        "role": "assistant",
        "content": [
            {"text": "old reply"},
            {
                "toolUse": {
                    "toolUseId": "t1",
                    "name": "search",
                    "input": {"q": "old"},
                }
            },
        ],
    }
    data: Final = {
        "endpoint": "model/test/converse",
        "data": {
            "messages": [
                assistant,
                {"role": "user", "content": [{"text": "private"}]},
            ]
        },
    }
    original_assistant: Final = deepcopy(assistant)
    guardrail: Final = MagicMock()
    guardrail.apply_guardrail = AsyncMock(return_value={"texts": ["[MASKED]"]})
    guardrail.skip_system_message_in_guardrail = False
    guardrail.skip_tool_message_in_guardrail = False
    guardrail.skip_assistant_message_in_guardrail = True

    await BedrockPassthroughGuardrailHandler().process_input_messages(data, guardrail)

    assert guardrail.apply_guardrail.call_args.kwargs["inputs"]["texts"] == ["private"]
    assert data["data"]["messages"][0] == original_assistant
    assert data["data"]["messages"][1]["content"][0]["text"] == "[MASKED]"
