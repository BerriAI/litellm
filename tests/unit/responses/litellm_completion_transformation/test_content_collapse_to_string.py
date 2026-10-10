"""
Regression tests for BerriAI/litellm#45618: the responses -> chat leg of the
bridge must collapse a single plain text block back to a string.

Providers like Databricks validate json_object requests against string
message content only, so a bridged round trip that always emits list-form
content breaks them even though the client sent valid string content.
"""

from functools import partial

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)

transform = partial(
    LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content,
    collapse_single_text_block=True,
)


def test_single_text_block_stays_structured_by_default():
    # scanning consumers (guardrail translation, DLP) call without the flag and
    # must keep the structured content they scan and rewrite in place
    result = LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(
        [{"type": "text", "text": "World"}]
    )
    assert result == [{"type": "text", "text": "World"}]



def test_single_text_block_collapses_to_string():
    # the bridge turns a plain-string user message into [{"type": "text", ...}];
    # the return leg must restore the string
    result = transform([{"type": "text", "text": "Reply in JSON with key a. hi"}])
    assert result == "Reply in JSON with key a. hi"


def test_single_input_text_block_collapses_to_string():
    result = transform([{"type": "input_text", "text": "hello"}])
    assert result == "hello"


def test_single_plain_str_element_collapses_to_string():
    result = transform(["just a string"])
    assert result == "just a string"


def test_multiple_text_blocks_stay_a_list():
    blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    result = transform(blocks)
    assert isinstance(result, list) and len(result) == 2


def test_text_block_with_cache_control_stays_a_list():
    # collapsing would drop the cache_control metadata (#32511)
    result = transform([{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}])
    assert isinstance(result, list)
    assert result[0]["cache_control"] == {"type": "ephemeral"}


def test_single_image_block_stays_a_list():
    block = {"type": "input_image", "image_url": "https://example.com/x.png"}
    result = transform([block])
    assert isinstance(result, list) and len(result) == 1


def test_string_content_passthrough_unchanged():
    assert transform("already a string") == "already a string"
