import os
import sys
import pytest
import asyncio
from typing import Optional
from unittest.mock import patch, AsyncMock

sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.integrations.custom_logger import CustomLogger
import json
from litellm.types.utils import StandardLoggingPayload
from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponseTextConfig,
    ResponseAPIUsage,
    IncompleteDetails,
)
import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from base_responses_api import BaseResponsesAPITest
from openai.types.responses.function_tool import FunctionTool


class TestAnthropicResponsesAPITest(BaseResponsesAPITest):
    def get_base_completion_call_args(self):
        #litellm._turn_on_debug()
        return {
            "model": "anthropic/claude-3-5-sonnet-latest",
        }
    
    async def test_basic_openai_responses_delete_endpoint(self, sync_mode=False):
        pass
    
    async def test_basic_openai_responses_streaming_delete_endpoint(self, sync_mode=False):
        pass


def test_multiturn_tool_calls():
    # Test streaming response with tools for Anthropic
    litellm._turn_on_debug()
    shell_tool = dict(FunctionTool(
        type="function",
        name="shell",
        description="Runs a shell command, and returns its output.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "array", "items": {"type": "string"}},
                "workdir": {"type": "string", "description": "The working directory for the command."}
            },
            "required": ["command"]
        },
        strict=True
    ))
    

    
    # Step 1: Initial request with the tool
    response = litellm.responses(
        input=[{
            'role': 'user', 
            'content': [
                {'type': 'input_text', 'text': 'make a hello world html file'}
            ], 
            'type': 'message'
        }],
        model='anthropic/claude-3-7-sonnet-latest',
        instructions='You are a helpful coding assistant.',
        tools=[shell_tool]
    )
    
    print("response=", response)
    
    # Step 2: Send the results of the tool call back to the model
    # Get the response ID and tool call ID from the response

    response_id = response.id
    tool_call_id = ""
    for item in response.output:
        if 'type' in item and item['type'] == 'function_call':
            tool_call_id = item['call_id']
            break

    # Use await with asyncio.run for the async function
    follow_up_response = litellm.responses(
        model='anthropic/claude-3-7-sonnet-latest',
        previous_response_id=response_id,
        input=[{
            'type': 'function_call_output',
            'call_id': tool_call_id,
            'output': '{"output":"<html>\\n<head>\\n  <title>Hello Page</title>\\n</head>\\n<body>\\n  <h1>Hi</h1>\\n  <p>Welcome to this simple webpage!</p>\\n</body>\\n</html> > index.html\\n","metadata":{"exit_code":0,"duration_seconds":0}}'
        }],
        tools=[shell_tool]
    )
    
    print("follow_up_response=", follow_up_response)
        

    