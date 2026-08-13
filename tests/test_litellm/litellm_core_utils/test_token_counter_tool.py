#### What this tests ####
#    This tests litellm.token_counter() function
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

# Use the same token_counter as the main test.
from tests.test_litellm.litellm_core_utils.test_token_counter import token_counter
from tests.test_litellm.litellm_core_utils.test_token_counter_tool_data import *


@pytest.mark.parametrize(
    "messages",
    [
        SHORT,
        CONTENT_AND_TOOL_CALL,
        SYSTEM_LONG,
        TOOL_CALL_CONTENT_ARRAY,
    ],
    ids=[
        "SHORT",
        "CONTENT_AND_TOOL_CALL",
        "SYSTEM_LONG",
        "TOOL_CALL_CONTENT_ARRAY",
    ],
)
def test_token_counter_tool_increases(messages):
    conversation = []
    prev_tokens = 0
    for message in messages:
        conversation.append(message)
        tokens = token_counter(model="gpt-3.5-turbo", messages=conversation, tools=TOOLS)  # type: ignore
        print(f"tokens: {tokens}")
        assert (
            tokens > prev_tokens
        ), f"Token did not increase: {tokens} <= {prev_tokens}"
        prev_tokens = tokens


# Reuse in multiple tests
@pytest.mark.parametrize("usermessage", USER_MESSAGES, ids=USER_MESSAGES_IDS)
@pytest.mark.parametrize("tool_call", TOOL_CALL_MESSAGES, ids=TOOL_CALL_MESSAGE_ids)
def test_grow(usermessage, tool_call):
    assertGrow(usermessage, tool_call, False)


@pytest.mark.parametrize("usermessage", USER_MESSAGES, ids=USER_MESSAGES_IDS)
@pytest.mark.parametrize("tool_call", TOOL_CALL_MESSAGES, ids=TOOL_CALL_MESSAGE_ids)
def test_sum(usermessage, tool_call):
    assertGrow(usermessage, tool_call, True)


def assertGrow(usermessage, tool_call, assertBiggerThanBoth=True):
    print(usermessage, tool_call)
    tokens_usermessage = token_counter(
        model="anthropic.claude-instant-v1",
        messages=[usermessage],
        tools=TOOLS,  # type: ignore
    )
    assert tokens_usermessage > 0
    tokens_tool_call = token_counter(
        model="anthropic.claude-instant-v1",
        messages=[tool_call],
        tools=TOOLS,  # type: ignore
    )
    assert tokens_tool_call > 0
    tokens_both = token_counter(
        model="anthropic.claude-instant-v1",
        messages=[usermessage, tool_call],
        tools=TOOLS,  # type: ignore
    )
    assert tokens_both > tokens_usermessage
    assert tokens_both > tokens_tool_call
    if assertBiggerThanBoth:
        assert abs(tokens_usermessage + tokens_tool_call - tokens_both) <= 61, (
            f"tokens_usermessage: {tokens_usermessage}, tokens_tool_call: {tokens_tool_call}, "
            + f"tokens_both: {tokens_both} diff: {tokens_usermessage + tokens_tool_call - tokens_both}"
        )
