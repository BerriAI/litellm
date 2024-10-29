import sys, os
import traceback
from dotenv import load_dotenv
import copy

load_dotenv()
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
# import asyncio
import litellm
import pytest

def test_gemini_function_call_with_args():
    messages=[
        {
            "role":"user",
            "content":"Hi"
        },
        {
            "role": "assistant",
            "content": "hello",
            "tool_calls":[
                {
                    "index": 0,
                    "function": {
                        "arguments": "{\"arg\": \"test\"}",
                        "name": "test"
                    },
                    "id": "call_597e00e6-11d4-4ed2-94b2-27edee250aec",
                    "type": "function"
                }
            ],
        },
        {
            "tool_call_id": "call_597e00e6-11d4-4ed2-94b2-27edee250aec",
            "role": "tool",
            "name": "test",
            "content": [
                {
                    "type": "text",
                    "text": "42"
                }
            ]
        },
    ]
    r = litellm.completion(messages=messages, model="gemini/gemini-1.5-flash-002")
    assert len(r.choices) > 0
def test_gemini_function_call_without_args():
    messages=[
        {
            "role":"user",
            "content":"Hi"
        },
        {
            "role": "assistant",
            "content": "hello",
            "tool_calls":[
                {
                    "index": 0,
                    "function": {
                        "arguments": "{}",
                        "name": "test"
                    },
                    "id": "call_597e00e6-11d4-4ed2-94b2-27edee250aec",
                    "type": "function"
                }
            ],
        },
        {
            "tool_call_id": "call_597e00e6-11d4-4ed2-94b2-27edee250aec",
            "role": "tool",
            "name": "test",
            "content": [
                {
                    "type": "text",
                    "text": "42"
                }
            ]
        },
    ]
    r = litellm.completion(messages=messages, model="gemini/gemini-1.5-flash-002")
    assert len(r.choices) > 0