import pytest

from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig


def _function_tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object"}},
    }


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    ("model", "thinking", "tool_choice", "expected"),
    [
        ("deepseek-v4-pro", {"type": "enabled"}, "required", "auto"),
        (
            "deepseek-v4-pro",
            {"type": "enabled"},
            {"type": "function", "function": {"name": "shell"}},
            "auto",
        ),
        ("deepseek-v4-pro", {"type": "enabled"}, "none", "none"),
        ("deepseek-v4-pro", {"type": "enabled"}, "auto", "auto"),
        ("deepseek-v4-pro", {"type": "enabled"}, None, None),
        ("deepseek-v4-pro", {"type": "disabled"}, "required", "required"),
        ("deepseek-chat", {"type": "enabled"}, "required", "required"),
    ],
)
async def test_transform_request_normalizes_tool_choice_for_thinking(
    is_async: bool,
    model: str,
    thinking: dict[str, str],
    tool_choice: object | None,
    expected: object | None,
):
    config = DeepSeekChatConfig()
    tool_choice_param = {"tool_choice": tool_choice} if tool_choice is not None else {}
    request = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "optional_params": {
            "thinking": thinking,
            "tools": [_function_tool("shell")],
            **tool_choice_param,
        },
        "litellm_params": {},
        "headers": {},
    }

    body = (
        await config.async_transform_request(**request)
        if is_async
        else config.transform_request(**request)
    )

    assert body.get("tool_choice") == expected
