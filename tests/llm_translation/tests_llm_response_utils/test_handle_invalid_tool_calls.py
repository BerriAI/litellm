import json
import os
import sys
from datetime import datetime

sys.path.insert(
    0, os.path.abspath("../../")
)  # Adds the parent directory to the system path


import pytest


@pytest.mark.parametrize(
    "function_name, expect_modification",
    [
        ("multi_tool_use.parallel", True),
        ("my-fake-function", False),
    ],
)
def test_openai_hallucinated_tool_call_util(function_name, expect_modification):
    """
    Patch for this issue: https://community.openai.com/t/model-tries-to-call-unknown-function-multi-tool-use-parallel/490653

    Handle openai invalid tool calling response.

    OpenAI assistant will sometimes return an invalid tool calling response, which needs to be parsed

    -           "arguments": "{\"tool_uses\":[{\"recipient_name\":\"product_title\",\"parameters\":{\"content\":\"Story Scribe\"}},{\"recipient_name\":\"one_liner\",\"parameters\":{\"content\":\"Transform interview transcripts into actionable user stories\"}}]}",

    To extract actual tool calls:

    1. Parse arguments JSON object
    2. Iterate over tool_uses array to call functions:
        - get function name from recipient_name value
        - parameters will be JSON object for function arguments
    """
    from litellm.litellm_core_utils.llm_response_utils import (
        _handle_invalid_parallel_tool_calls,
    )
    from litellm.types.utils import ChatCompletionMessageToolCall

    response = _handle_invalid_parallel_tool_calls(
        tool_calls=[
            ChatCompletionMessageToolCall(
                **{
                    "function": {
                        "arguments": '{"tool_uses":[{"recipient_name":"product_title","parameters":{"content":"Story Scribe"}},{"recipient_name":"one_liner","parameters":{"content":"Transform interview transcripts into actionable user stories"}}]}',
                        "name": function_name,
                    },
                    "id": "call_IzGXwVa5OfBd9XcCJOkt2q0s",
                    "type": "function",
                }
            )
        ]
    )

    print(f"response: {response}")

    if expect_modification:
        for idx, tc in enumerate(response):
            if idx == 0:
                assert tc.model_dump() == {
                    "function": {
                        "arguments": '{"content": "Story Scribe"}',
                        "name": "product_title",
                    },
                    "id": "call_IzGXwVa5OfBd9XcCJOkt2q0s_0",
                    "type": "function",
                }
            elif idx == 1:
                assert tc.model_dump() == {
                    "function": {
                        "arguments": '{"content": "Transform interview transcripts into actionable user stories"}',
                        "name": "one_liner",
                    },
                    "id": "call_IzGXwVa5OfBd9XcCJOkt2q0s_1",
                    "type": "function",
                }
    else:
        assert len(response) == 1
        assert response[0].function.name == function_name
