from typing import Tuple
from litellm.types.utils import ModelResponse


first_openai_chunk_example = {
    "id": "chatcmpl-7zSKLBVXnX9dwgRuDYVqVVDsgh2yp",
    "object": "chat.completion.chunk",
    "created": 1694881253,
    "model": "gpt-4-0613",
    "choices": [
        {
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "finish_reason": None,  # it's null
        }
    ],
}


def validate_first_format(chunk):
    # write a test to make sure chunk follows the same format as first_openai_chunk_example
    assert isinstance(chunk, ModelResponse), "Chunk should be a dictionary."
    assert isinstance(chunk["id"], str), "'id' should be a string."
    assert isinstance(chunk["object"], str), "'object' should be a string."
    assert isinstance(chunk["created"], int), "'created' should be an integer."
    assert isinstance(chunk["model"], str), "'model' should be a string."
    assert isinstance(chunk["choices"], list), "'choices' should be a list."
    assert not hasattr(chunk, "usage"), "Chunk cannot contain usage"

    for choice in chunk["choices"]:
        assert isinstance(choice["index"], int), "'index' should be an integer."
        assert isinstance(choice["delta"]["role"], str), "'role' should be a string."
        assert "messages" not in choice
        # openai v1.0.0 returns content as None
        assert (choice["finish_reason"] is None) or isinstance(
            choice["finish_reason"], str
        ), "'finish_reason' should be None or a string."


second_openai_chunk_example = {
    "id": "chatcmpl-7zSKLBVXnX9dwgRuDYVqVVDsgh2yp",
    "object": "chat.completion.chunk",
    "created": 1694881253,
    "model": "gpt-4-0613",
    "choices": [
        {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}  # it's null
    ],
}


def validate_second_format(chunk):
    assert isinstance(chunk, ModelResponse), "Chunk should be a dictionary."
    assert isinstance(chunk["id"], str), "'id' should be a string."
    assert isinstance(chunk["object"], str), "'object' should be a string."
    assert isinstance(chunk["created"], int), "'created' should be an integer."
    assert isinstance(chunk["model"], str), "'model' should be a string."
    assert isinstance(chunk["choices"], list), "'choices' should be a list."
    assert not hasattr(chunk, "usage"), "Chunk cannot contain usage"

    for choice in chunk["choices"]:
        assert isinstance(choice["index"], int), "'index' should be an integer."
        assert hasattr(choice["delta"], "role"), "'role' should be a string."
        # openai v1.0.0 returns content as None
        assert (choice["finish_reason"] is None) or isinstance(
            choice["finish_reason"], str
        ), "'finish_reason' should be None or a string."


last_openai_chunk_example = {
    "id": "chatcmpl-7zSKLBVXnX9dwgRuDYVqVVDsgh2yp",
    "object": "chat.completion.chunk",
    "created": 1694881253,
    "model": "gpt-4-0613",
    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
}

"""
Final chunk (sdk):
chunk: ChatCompletionChunk(id='chatcmpl-96mM3oNBlxh2FDWVLKsgaFBBcULmI', 
choices=[Choice(delta=ChoiceDelta(content=None, function_call=None, role=None, 
tool_calls=None), finish_reason='stop', index=0, logprobs=None)], 
created=1711402871, model='gpt-3.5-turbo-0125', object='chat.completion.chunk', system_fingerprint='fp_3bc1b5746c')
"""


def validate_last_format(chunk):
    """
    Ensure last chunk has no remaining content or tools
    """
    assert isinstance(chunk, ModelResponse), "Chunk should be a dictionary."
    assert isinstance(chunk["id"], str), "'id' should be a string."
    assert isinstance(chunk["object"], str), "'object' should be a string."
    assert isinstance(chunk["created"], int), "'created' should be an integer."
    assert isinstance(chunk["model"], str), "'model' should be a string."
    assert isinstance(chunk["choices"], list), "'choices' should be a list."
    assert not hasattr(chunk, "usage"), "Chunk cannot contain usage"

    for choice in chunk["choices"]:
        assert isinstance(choice["index"], int), "'index' should be an integer."
        assert choice["delta"]["content"] is None
        assert choice["delta"]["function_call"] is None
        assert choice["delta"]["role"] is None
        assert choice["delta"]["tool_calls"] is None
        assert isinstance(
            choice["finish_reason"], str
        ), "'finish_reason' should be a string."


def streaming_format_tests(idx, chunk) -> Tuple[str, bool]:
    extracted_chunk = ""
    finished = False
    print(f"chunk: {chunk}")
    if idx == 0:  # ensure role assistant is set
        validate_first_format(chunk=chunk)
        role = chunk["choices"][0]["delta"]["role"]
        assert role == "assistant"
    elif idx == 1:  # second chunk
        validate_second_format(chunk=chunk)
    if idx != 0:  # ensure no role
        if "role" in chunk["choices"][0]["delta"]:
            pass  # openai v1.0.0+ passes role = None
    if chunk["choices"][0][
        "finish_reason"
    ]:  # ensure finish reason is only in last chunk
        validate_last_format(chunk=chunk)
        finished = True
    if (
        "content" in chunk["choices"][0]["delta"]
        and chunk["choices"][0]["delta"]["content"] is not None
    ):
        extracted_chunk = chunk["choices"][0]["delta"]["content"]
    print(f"extracted chunk: {extracted_chunk}")
    return extracted_chunk, finished
