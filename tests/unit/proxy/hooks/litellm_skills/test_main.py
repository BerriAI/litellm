from unittest.mock import AsyncMock, patch

import pytest

from litellm.proxy.hooks.litellm_skills.main import SkillsInjectionHook

SKILL_TOOL_NAME = "litellm_skill_e2b8dca8_031a_4481_b034_b9ec7d4eb7bf"


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
