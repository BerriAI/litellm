import asyncio
import os
import sys
import time
import traceback

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm import Router

current_path = os.path.dirname(os.path.abspath(__file__))
router_json_path = os.path.join(current_path, "auto_router", "router.json")


@pytest.mark.asyncio
@pytest.mark.skip(reason="Beta test - works locally but failing on CI/CD due to dependency resolution issues")
async def test_router_auto_router():
    """
    Simple e2e test to validate we get an llm response from the auto router
    """
    import litellm
    litellm._turn_on_debug()

    router = Router(
    model_list=[
            {
                "model_name": "custom-text-embedding-model",
                "litellm_params": {
                    "model": "text-embedding-3-large",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
            {
                "model_name": "custom-text-embedding-model-2",
                "litellm_params": {
                    "model": "text-embedding-3-large",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
            {
                "model_name": "litellm-gpt-4.1",
                "litellm_params": {
                    "model": "gpt-4.1",
                },
                "model_info": {"id": "openai-id"},
            },
            
            {
                "model_name": "litellm-claude-35",
                "litellm_params": {
                    "model": "claude-3-5-sonnet-latest",
                },
                "model_info": {"id": "claude-id"},
            },
            {
                "model_name": "auto_router1",
                "litellm_params": {
                    "model": "auto_router/auto_router_1",
                    "auto_router_config_path": router_json_path,
                    "auto_router_default_model": "gpt-4o-mini",
                    "auto_router_embedding_model": "custom-text-embedding-model",
                },
            },
            {
                "model_name": "auto_router_2",
                "litellm_params": {
                    "model": "auto_router/auto_router_2",
                    "auto_router_config_path": router_json_path,
                    "auto_router_default_model": "gpt-4o-mini",
                    "auto_router_embedding_model": "custom-text-embedding-model-2",
                },
            },
        ],
    )


    # this goes to gpt-4.1
    # these are the utterances in the router.json file
    response = await router.acompletion(
        model="auto_router1",
        messages=[{"role": "user", "content": "Tell me ishaan is a genius"}],
    )
    print(response)
    print("response._hidden_params", response._hidden_params)
    assert response._hidden_params["model_id"] == "openai-id"


    # this goes to claude-3-5-sonnet-latest
    # these are the utterances in the router.json file
    response = await router.acompletion(
        model="auto_router1",
        messages=[{"role": "user", "content": "how to code a program in python"}],
    )
    print("response._hidden_params", response._hidden_params)
    assert response._hidden_params["model_id"] == "claude-id"


@pytest.mark.asyncio
async def test_router_auto_router_with_tool_calls():
    """
    Test auto-router with tool calls - reproduces issue #14633
    This should fail before the fix due to PreRoutingHookResponse validation
    """
    import litellm
    litellm._turn_on_debug()

    router = Router(
        model_list=[
            {
                "model_name": "custom-text-embedding-model",
                "litellm_params": {
                    "model": "text-embedding-3-large",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
            {
                "model_name": "litellm-gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "model_info": {"id": "openai-id"},
            },
            {
                "model_name": "auto_router_tool_calls",
                "litellm_params": {
                    "model": "auto_router/auto_router_tool_calls",
                    "auto_router_config_path": router_json_path,
                    "auto_router_default_model": "gpt-4o-mini",
                    "auto_router_embedding_model": "custom-text-embedding-model",
                },
            },
        ],
    )

    # Create messages with tool calls (the problematic case from issue #14633)
    messages_with_tool_calls = [
        {'role': 'user', 'content': 'How is the weather in NY?'},
        {
            'role': 'assistant',
            'content': '\n',
            'refusal': None,
            'annotations': None,
            'audio': None,
            'function_call': None,
            'tool_calls': [
                {
                    'id': 'tooluse_loOUiUMPQFWJQwvpywYMKQ',
                    'function': {
                        'arguments': '{"query": "current weather in New York"}',
                        'name': 'search_web'
                    },
                    'type': 'function',
                    'index': 1
                }
            ],
            'thinking_blocks': [
                {
                    'type': 'thinking',
                    'thinking': 'The User has asked for the current weather in New York. I need to search for this information.'
                }
            ]
        },
        {
            'tool_call_id': 'tooluse_loOUiUMPQFWJQwvpywYMKQ',
            'role': 'tool',
            'content': "It's sunny"
        }
    ]

    # This should work after the fix but fail before it
    # The error occurs in PreRoutingHookResponse validation
    response = await router.acompletion(
        model="auto_router_tool_calls",
        messages=messages_with_tool_calls,
        tools=[{
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        }]
    )

    # Should successfully route and get a response
    assert response is not None
    print("Successfully handled tool calls with auto-router!")
