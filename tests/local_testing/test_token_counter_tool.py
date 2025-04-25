#### What this tests ####
#    This tests litellm.token_counter() function
import traceback
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm import token_counter

from test_token_counter_tool_data import *

@pytest.mark.parametrize(
    "messages",
    [MESSAGE_CHAIN, MESSAGE_CHAIN_FAILS_NO_FUNCTION, MESSAGE_CHAIN_FAILS_NO_FUNCTION_ARGS, 
     MESSAGE_CHAIN_WORKS, MESSAGE_ORIGINAL_LONG, SYSTEM_SHORT_AND_TOOL,
     SYSTEM_SHORTER_AND_TOOL, SYSTEM_SHORTER_AND_TOOL_NO_CONTENT, SYSTEM_AND_TOOL_SHORT_CONTENT],
    ids=["MESSAGE_CHAIN", "MESSAGE_CHAIN_FAILS_NO_FUNCTION", "MESSAGE_CHAIN_FAILS_NO_FUNCTION_ARGS", 
         "MESSAGE_CHAIN_WORKS", "MESSAGE_ORIGINAL_LONG", "SYSTEM_SHORT_AND_TOOL",
         "SYSTEM_SHORTER_AND_TOOL", "SYSTEM_SHORTER_AND_TOOL_NO_CONTENT", "SYSTEM_AND_TOOL_SHORT_CONTENT"],
)
def test_token_counter_tool_increases(messages):
    conversation = []
    prev_tokens = 0
    for message in messages:
        conversation.append(message)
        tokens = token_counter(model="gpt-3.5-turbo", messages=conversation, tools=tools) # type: ignore
        print(f"tokens: {tokens}")
        assert (
            tokens > prev_tokens
        ), f"Token did not increase: {tokens} <= {prev_tokens}"
        prev_tokens = tokens

# Reuse in multiple tests
@pytest.mark.parametrize("usermessage", usermessage_params, ids=usermessage_ids)
@pytest.mark.parametrize("tool_call", tool_call_params, ids=tool_call_ids)
def test_grow(usermessage, tool_call):
    assertGrow(usermessage, tool_call, False)


@pytest.mark.parametrize("usermessage", usermessage_params, ids=usermessage_ids)
@pytest.mark.parametrize("tool_call", tool_call_params, ids=tool_call_ids)
def test_sum(usermessage, tool_call):
    assertGrow(usermessage, tool_call, True)


def assertGrow(usermessage, tool_call, assertBiggerThanBoth=True):
    print(usermessage, tool_call)
    tokens_usermessage = token_counter(
        model="anthropic.claude-instant-v1", messages=[usermessage], 
        tools=tools # type: ignore
    )
    assert tokens_usermessage > 0
    tokens_tool_call = token_counter(
        model="anthropic.claude-instant-v1", messages=[tool_call], 
        tools=tools # type: ignore
    )
    assert tokens_tool_call > 0
    tokens_both = token_counter(
        model="anthropic.claude-instant-v1", messages=[usermessage, tool_call], 
        tools=tools # type: ignore
    )
    assert tokens_both > tokens_usermessage
    assert tokens_both > tokens_tool_call
    if assertBiggerThanBoth:
        assert  abs(tokens_usermessage + tokens_tool_call - tokens_both) <= 10, (f'tokens_usermessage: {tokens_usermessage}, tokens_tool_call: {tokens_tool_call}, '
              + f'tokens_both: {tokens_both} diff: {tokens_usermessage + tokens_tool_call - tokens_both}')
