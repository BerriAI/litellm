import logging
import os
import pytest
from mcp.types import Tool as MCPTool
from typing import Any, cast

import litellm
from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler


class MockUserAPIKeyAuth:
    """Mock UserAPIKeyAuth for testing"""

    def __init__(self):
        self.api_key = "test_key"
        self.user_id = "test_user"
        self.team_id = "test_team"
        self.user_email = "test@example.com"
        self.max_budget = 100.0
        self.spend = 0.0
        self.models = []
        self.aliases = {}
        self.config = {}
        self.permissions = {}
        self.metadata = {}
        self.object_permission_id = "test_permission_id"

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        pytest.param("gpt-4o-mini", id="openai"),
        pytest.param("claude-haiku-4-5", id="anthropic"),
    ],
)
async def test_streaming_responses_api_with_mcp_tools(
    model: str, caplog: pytest.LogCaptureFixture
):
    """
    Test the streaming responses API with MCP tools when using server_url="litellm_proxy"

    Under the hood the follow occurs

    - MCP: responses called litellm MCP manager.list_tools (MOCKED)
    - Request 1: Made to model under test with fetched tools (REAL LLM CALL)
    - MCP: Execute tool call from request 1 and returns result (MOCKED)
    - Request 2: Made to model under test with fetched tools and tool results (REAL LLM CALL)

    Return the user the result of request 2
    """
    if ("claude" in model.lower() or "anthropic" in model.lower()) and not os.getenv(
        "ANTHROPIC_API_KEY"
    ):
        pytest.skip("ANTHROPIC_API_KEY not set, skipping anthropic model test")
    if ("gpt" in model.lower() or "openai" in model.lower()) and not os.getenv(
        "OPENAI_API_KEY"
    ):
        pytest.skip("OPENAI_API_KEY not set, skipping openai model test")

    from unittest.mock import AsyncMock, patch

    print("🧪 Testing basic streaming with MCP tools...")

    mock_mcp_tools = [
        MCPTool.model_validate({
                "name": "search_repo",
                "description": "Search BerriAI/litellm repository for information",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            }, by_name=False)
    ]

    with caplog.at_level(logging.ERROR):
        with (
            patch.object(
                LiteLLM_Proxy_MCP_Handler,
                "_get_mcp_tools_from_manager",
                new_callable=AsyncMock,
            ) as mock_get_tools,
            patch.object(
                LiteLLM_Proxy_MCP_Handler,
                "_execute_tool_calls",
                new_callable=AsyncMock,
            ) as mock_execute_tools,
        ):
            mock_get_tools.return_value = (mock_mcp_tools, ["litellm_proxy"])

            def mock_execute_tool_calls_side_effect(
                tool_calls, user_api_key_auth, **kwargs
            ):
                """Mock function that returns results matching the actual tool call IDs from the LLM"""
                results = []
                for tool_call in tool_calls:
                    call_id = None
                    if isinstance(tool_call, dict):
                        call_id = tool_call.get("call_id") or tool_call.get("id")
                    elif hasattr(tool_call, "call_id"):
                        call_id = tool_call.call_id
                    elif hasattr(tool_call, "id"):
                        call_id = tool_call.id

                    if call_id:
                        results.append(
                            {
                                "tool_call_id": call_id,
                                "result": "LiteLLM is a unified interface for 100+ LLMs that translates inputs to provider-specific completion endpoints and provides consistent OpenAI-format output.",
                            }
                        )
                return results

            mock_execute_tools.side_effect = mock_execute_tool_calls_side_effect

            mcp_tool_config = cast(
                Any,
                {
                    "type": "mcp",
                    "server_url": "litellm_proxy",
                    "require_approval": "never",
                },
            )
            response = await litellm.aresponses(
                model=model,
                tools=[mcp_tool_config],
                tool_choice="required",
                input=[
                    {
                        "role": "user",
                        "type": "message",
                        "content": "give me a TLDR of what BerriAI/litellm is about",
                    }
                ],
                stream=True,
            )

            print(f"📋 Response type: {type(response)}")
            assert hasattr(
                response, "__aiter__"
            ), "Response should be an async streaming response"

            chunks = []
            async for chunk in response:
                chunks.append(chunk)
                print(f"📦 Chunk type: {getattr(chunk, 'type', 'unknown')}")

            print(f"📊 Total chunks received: {len(chunks)}")

            assert (
                mock_get_tools.call_count >= 1
            ), f"Expected MCP tools to be fetched at least once, got {mock_get_tools.call_count}"
            print(f"MCP tools fetched: {len(mock_mcp_tools)}")

            assert response is not None
            assert len(chunks) > 0, "Should have received streaming chunks"

            print("Basic streaming responses API with MCP tools test passed!")

    lite_errors = [
        record
        for record in caplog.records
        if record.levelno >= logging.ERROR
        and ("LiteLLM" in record.name or "LiteLLM" in record.getMessage())
    ]
    assert not lite_errors, "Unexpected LiteLLM errors: " + ", ".join(
        record.getMessage() for record in lite_errors
    )



@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-4o-mini"])
async def test_streaming_mcp_event_order_and_response_id_consistency(
    model: str, caplog: pytest.LogCaptureFixture
):
    """
    Test that:
    1. Streaming events are emitted in correct order (response.created, response.in_progress, response.output_item.added before MCP events)
    2. All response lifecycle events share the same response ID within a cycle
    """
    if ("gpt" in model.lower() or "openai" in model.lower()) and not os.getenv(
        "OPENAI_API_KEY"
    ):
        pytest.skip("OPENAI_API_KEY not set, skipping openai model test")

    from unittest.mock import AsyncMock, patch

    mock_mcp_tools = [
        MCPTool.model_validate({
                "name": "get_weather",
                "description": "Get weather for a city",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"],
                },
            }, by_name=False)
    ]

    with caplog.at_level(logging.ERROR):
        with (
            patch.object(
                LiteLLM_Proxy_MCP_Handler,
                "_get_mcp_tools_from_manager",
                new_callable=AsyncMock,
            ) as mock_get_tools,
            patch.object(
                LiteLLM_Proxy_MCP_Handler,
                "_execute_tool_calls",
                new_callable=AsyncMock,
            ) as mock_execute_tools,
        ):
            mock_get_tools.return_value = (mock_mcp_tools, ["litellm_proxy"])

            def mock_execute_side_effect(tool_calls, user_api_key_auth, **kwargs):
                results = []
                for tool_call in tool_calls:
                    call_id = None
                    if isinstance(tool_call, dict):
                        call_id = tool_call.get("call_id") or tool_call.get("id")
                    elif hasattr(tool_call, "call_id"):
                        call_id = tool_call.call_id
                    elif hasattr(tool_call, "id"):
                        call_id = tool_call.id
                    if call_id:
                        results.append(
                            {
                                "tool_call_id": call_id,
                                "result": "Sunny, 72°F",
                            }
                        )
                return results

            mock_execute_tools.side_effect = mock_execute_side_effect

            mcp_tool_config = cast(
                Any,
                {
                    "type": "mcp",
                    "server_url": "litellm_proxy",
                    "require_approval": "never",
                },
            )

            response = await litellm.aresponses(
                model=model,
                tools=[mcp_tool_config],
                input=[
                    {
                        "role": "user",
                        "type": "message",
                        "content": "What's the weather in San Francisco?",
                    }
                ],
                stream=True,
            )

            events = []
            async for chunk in response:
                events.append(chunk)

            assert len(events) > 0, "Should receive streaming events"

            created_idx = next(
                (
                    i
                    for i, e in enumerate(events)
                    if getattr(e, "type", None) == "response.created"
                ),
                None,
            )
            in_progress_idx = next(
                (
                    i
                    for i, e in enumerate(events)
                    if getattr(e, "type", None) == "response.in_progress"
                ),
                None,
            )
            output_item_added_idx = next(
                (
                    i
                    for i, e in enumerate(events)
                    if getattr(e, "type", None) == "response.output_item.added"
                ),
                None,
            )
            mcp_in_progress_idx = next(
                (
                    i
                    for i, e in enumerate(events)
                    if "mcp_list_tools.in_progress" in str(getattr(e, "type", ""))
                ),
                None,
            )
            completed_idx = next(
                (
                    i
                    for i, e in enumerate(events)
                    if getattr(e, "type", None) == "response.completed"
                ),
                None,
            )

            assert created_idx is not None, "response.created event should be present"
            assert (
                in_progress_idx is not None
            ), "response.in_progress event should be present"
            assert (
                output_item_added_idx is not None
            ), "response.output_item.added event should be present"

            assert (
                created_idx < in_progress_idx
            ), "response.created should come before response.in_progress"
            assert (
                in_progress_idx < output_item_added_idx
            ), "response.in_progress should come before response.output_item.added"

            if mcp_in_progress_idx is not None:
                assert (
                    output_item_added_idx < mcp_in_progress_idx
                ), "response.output_item.added should come before response.mcp_list_tools.in_progress"

            response_ids = []
            for i, event in enumerate(events):
                event_type = getattr(event, "type", None)
                if hasattr(event, "response"):
                    response_obj = getattr(event, "response", None)
                    if response_obj and hasattr(response_obj, "id"):
                        event_type_value = (
                            event_type.value
                            if hasattr(event_type, "value")
                            else str(event_type)
                        )
                        if any(
                            x in event_type_value
                            for x in [
                                "response.created",
                                "response.in_progress",
                                "response.completed",
                            ]
                        ):
                            response_ids.append((i, event_type_value, response_obj.id))

            assert (
                len(response_ids) >= 2
            ), f"Should have at least 2 response lifecycle events. Found {len(response_ids)}"

            cycles = []
            current_cycle = []
            current_id = None

            for idx, event_type, resp_id in response_ids:
                if current_id is None or resp_id == current_id:
                    current_cycle.append((idx, event_type, resp_id))
                    current_id = resp_id
                else:
                    if current_cycle:
                        cycles.append(current_cycle)
                    current_cycle = [(idx, event_type, resp_id)]
                    current_id = resp_id
            if current_cycle:
                cycles.append(current_cycle)

            for cycle_num, cycle in enumerate(cycles):
                cycle_ids = set(resp_id for _, _, resp_id in cycle)
                assert (
                    len(cycle_ids) == 1
                ), f"Cycle {cycle_num + 1} should have consistent response ID. Found {len(cycle_ids)} unique IDs"

            assert (
                completed_idx is not None
            ), "response.completed event should be present"

    lite_errors = [
        record
        for record in caplog.records
        if record.levelno >= logging.ERROR
        and ("LiteLLM" in record.name or "LiteLLM" in record.getMessage())
    ]
    assert not lite_errors, "Unexpected LiteLLM errors: " + ", ".join(
        record.getMessage() for record in lite_errors
    )
