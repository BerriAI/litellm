import sys, os
import traceback
from dotenv import load_dotenv
load_dotenv()
import os, io
import litellm
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest

def test_single_function_call():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "do test"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "test"
                }
            ],
            "tool_calls": [
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
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "tell me the result."
                }
            ]
        },
    ]


    r = litellm.completion(messages=messages, model="gemini/gemini-1.5-flash-002")
    assert len(r.choices)>0
def test_multiple_function_call():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "do test"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "test"
                }
            ],
            "tool_calls": [
                {
                    "index": 0,
                    "function": {
                        "arguments": "{\"arg\": \"test\"}",
                        "name": "test"
                    },
                    "id": "call_597e00e6-11d4-4ed2-94b2-27edee250aec",
                    "type": "function"
                },
                {
                    "index": 1,
                    "function": {
                        "arguments": "{\"arg\": \"test2\"}",
                        "name": "test2"
                    },
                    "id": "call_2414e8f9-283a-002b-182a-1290ab912c02",
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
        {
            "tool_call_id": "call_2414e8f9-283a-002b-182a-1290ab912c02",
            "role": "tool",
            "name": "test2",
            "content": [
                {
                    "type": "text",
                    "text": "15"
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "tell me the results."
                }
            ]
        },
    ]


    r = litellm.completion(messages=messages, model="gemini/gemini-1.5-flash-002")
    assert len(r.choices) > 0

def test_multiple_function_call_changed_text_pos():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "do test"
                }
            ]
        },
        {

            "tool_calls": [
                {
                    "index": 0,
                    "function": {
                        "arguments": "{\"arg\": \"test\"}",
                        "name": "test"
                    },
                    "id": "call_597e00e6-11d4-4ed2-94b2-27edee250aec",
                    "type": "function"
                },
                {
                    "index": 1,
                    "function": {
                        "arguments": "{\"arg\": \"test2\"}",
                        "name": "test2"
                    },
                    "id": "call_2414e8f9-283a-002b-182a-1290ab912c02",
                    "type": "function"
                }
            ],
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "test"
                }
            ],
        },
        {
            "tool_call_id": "call_2414e8f9-283a-002b-182a-1290ab912c02",
            "role": "tool",
            "name": "test2",
            "content": [
                {
                    "type": "text",
                    "text": "15"
                }
            ]
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
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "tell me the results."
                }
            ]
        },
    ]


    r = litellm.completion(messages=messages, model="gemini/gemini-1.5-flash-002")
    assert len(r.choices) > 0