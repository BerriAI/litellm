from unittest.mock import AsyncMock, patch

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy._types import LiteLLM_SkillsTable, UserAPIKeyAuth
from litellm.proxy.hooks.litellm_skills.main import SkillsInjectionHook

SKILL_TOOL_NAME = "litellm_skill_e2b8dca8_031a_4481_b034_b9ec7d4eb7bf"


def _db_skill() -> LiteLLM_SkillsTable:
    return LiteLLM_SkillsTable(
        skill_id=SKILL_TOOL_NAME,
        display_title="Persona Skill",
        description="Speak politely",
        instructions="Always address the user respectfully",
        source="custom",
    )


def _skill_request_data(model: str = "gpt-4") -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "container": {
            "skills": [{"type": "custom", "skill_id": SKILL_TOOL_NAME}],
        },
    }


def _request_data():
    return {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "run the skill"}],
        "litellm_metadata": {
            "_litellm_code_execution_enabled": True,
            "_skill_files": {SKILL_TOOL_NAME: {"main.py": b"print('hi')"}},
        },
    }


def _tool_use_response(tool_name):
    return {
        "stop_reason": "tool_use",
        "content": [
            {"type": "tool_use", "id": "toolu_1", "name": tool_name, "input": {}}
        ],
    }

@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ["completion", "acompletion"])
async def test_pre_call_chat_completions_emits_openai_tool_schema(call_type):
    """
    Regression: /v1/chat/completions must not emit Anthropic tool objects.

    OpenAI rejects tools missing ``type`` with:
    Missing required parameter: 'tools[0].type'
    """
    hook = SkillsInjectionHook()
    data = _skill_request_data(model="gpt-4")

    with patch.object(
        hook, "_fetch_skill_from_db", new=AsyncMock(return_value=_db_skill())
    ):
        result = await hook.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="test"),
            cache=DualCache(),
            data=data,
            call_type=call_type,
        )

    assert isinstance(result, dict)
    assert "container" not in result
    tools = result["tools"]
    assert len(tools) >= 1
    assert tools[0]["type"] == "function"
    assert "function" in tools[0]
    assert "name" in tools[0]["function"]
    assert "input_schema" not in tools[0]

    system_messages = [
        m for m in result["messages"] if isinstance(m, dict) and m.get("role") == "system"
    ]
    assert system_messages
    assert "Always address the user respectfully" in system_messages[0]["content"]


@pytest.mark.asyncio
async def test_pre_call_anthropic_messages_keeps_anthropic_tool_schema():
    """Messages API must keep Anthropic tool shape (name + input_schema)."""
    hook = SkillsInjectionHook()
    data = _skill_request_data(model="claude-sonnet-4-5")

    with patch.object(
        hook, "_fetch_skill_from_db", new=AsyncMock(return_value=_db_skill())
    ):
        result = await hook.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="test"),
            cache=DualCache(),
            data=data,
            call_type="anthropic_messages",
        )

    assert isinstance(result, dict)
    tools = result["tools"]
    assert len(tools) >= 1
    assert "type" not in tools[0] or tools[0].get("type") != "function"
    assert "name" in tools[0]
    assert "input_schema" in tools[0]
    assert "function" not in tools[0]


@pytest.mark.asyncio
async def test_post_call_success_hook_executes_litellm_skill_tool():
    """DB skill tool names carry the litellm_skill_ prefix and must trigger the execution loop."""
    hook = SkillsInjectionHook()
    response = _tool_use_response(SKILL_TOOL_NAME)

    with patch.object(
        hook, "_execute_code_loop_messages_api", new=AsyncMock(return_value=response)
    ) as mock_loop:
        result = await hook.async_post_call_success_deployment_hook(
            request_data=_request_data(), response=response, call_type=None
        )

    mock_loop.assert_awaited_once()
    assert result is response


@pytest.mark.asyncio
async def test_execute_code_loop_dispatches_litellm_skill_tool():
    """The agentic loop must route litellm_skill_ tool calls to _execute_skill_tool."""
    hook = SkillsInjectionHook()
    final_response = {"stop_reason": "end_turn", "content": []}

    with (
        patch.object(
            hook, "_execute_skill_tool", new=AsyncMock(return_value="skill ran")
        ) as mock_exec,
        patch("litellm.anthropic.acreate", new=AsyncMock(return_value=final_response)),
    ):
        result = await hook._execute_code_loop_messages_api(
            data=_request_data(),
            response=_tool_use_response(SKILL_TOOL_NAME),
            skill_files={"main.py": b"print('hi')"},
        )

    mock_exec.assert_awaited_once()
    assert mock_exec.await_args.args[0] == SKILL_TOOL_NAME
    assert result is final_response
