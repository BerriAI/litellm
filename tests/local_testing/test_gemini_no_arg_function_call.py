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
def test_gemini_function_call_without_args_enabled_tool_calls():
    messages=[
        {
            "role":"user",
            "content":"invoke tool call"
        }
    ]
    tools=[
        {
            "type": "function",
            "function": {
                "name": "test",
                "description": "invoke test",
                "parameters": {
                    "type": "object",
                    "properties": {
                    },
                    "required": []
                }
            }
        },
    ]
    r = litellm.completion(messages=messages, model="gemini/gemini-1.5-flash-002",tools=tools,temperature=0.0,tool_choice="required")

    assert len(r.choices) > 0,"choices not found"
    assert len(r.choices[0].message.tool_calls) > 0, "tool calls not found"
    assert r.choices[0].message.tool_calls[0].function.name=="test","tool call name is not 'test'"
    assert r.choices[0].message.tool_calls[0].function.arguments == "{}" ,"arguments is not empty"
def test_gemini_function_call_with_args_enabled_tool_calls():
    messages=[
        {
            "role":"user",
            "content":"invoke tool call with arg1(42)"
        }
    ]
    tools=[
        {
            "type": "function",
            "function": {
                "name": "test2",
                "description": "invoke test",
                "parameters": {
                    "type": "object",

                    "properties": {
                        "arg1": {
                            "type": "string",
                            "description": "arg1"
                        },
                    },

                    "required": ["arg1"]
                }
            }
        },
    ]
    r = litellm.completion(messages=messages, model="gemini/gemini-1.5-flash-002",tools=tools,temperature=0.0,tool_choice="required")

    assert len(r.choices) > 0,"choices not found"
    assert len(r.choices[0].message.tool_calls) > 0, "tool calls not found"
    assert r.choices[0].message.tool_calls[0].function.name=="test2","tool call name is not 'test'"
    assert r.choices[0].message.tool_calls[0].function.arguments == '{"arg1": "42"}' ,"arguments is not match"
