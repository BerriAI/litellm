import pytest

import litellm
from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig


def _function_tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object"}},
    }


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    ("model", "thinking", "tool_choice", "drop_params", "expected"),
    [
        ("deepseek-v4-pro", {"type": "enabled"}, "required", True, "auto"),
        (
            "deepseek-v4-pro",
            {"type": "enabled"},
            {"type": "function", "function": {"name": "shell"}},
            True,
            "auto",
        ),
        ("deepseek-v4-pro", {"type": "enabled"}, "none", False, "none"),
        ("deepseek-v4-pro", {"type": "enabled"}, "auto", False, "auto"),
        ("deepseek-v4-pro", {"type": "enabled"}, None, False, None),
        ("deepseek-v4-pro", {"type": "disabled"}, "required", False, "required"),
        ("deepseek-chat", {"type": "enabled"}, "required", False, "required"),
    ],
)
async def test_transform_request_normalizes_tool_choice_for_thinking(
    is_async: bool,
    model: str,
    thinking: dict[str, str],
    tool_choice: object | None,
    drop_params: bool,
    expected: object | None,
):
    config = DeepSeekChatConfig()
    tool_choice_param = {"tool_choice": tool_choice} if tool_choice is not None else {}
    request = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "optional_params": {
            "thinking": thinking,
            "tools": [_function_tool("shell"), _function_tool("read_file")],
            **tool_choice_param,
        },
        "litellm_params": {"drop_params": drop_params},
        "headers": {},
    }

    body = (
        await config.async_transform_request(**request)
        if is_async
        else config.transform_request(**request)
    )

    assert body.get("tool_choice") == expected


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "tool_choice",
    [
        "required",
        {"type": "function", "function": {"name": "shell"}},
    ],
)
async def test_transform_request_rejects_forced_tool_choice_without_drop_params(
    is_async: bool,
    tool_choice: object,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "drop_params", False)
    config = DeepSeekChatConfig()
    request = {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "optional_params": {
            "thinking": {"type": "enabled"},
            "tools": [_function_tool("shell"), _function_tool("read_file")],
            "tool_choice": tool_choice,
        },
        "litellm_params": {},
        "headers": {},
    }

    with pytest.raises(litellm.UnsupportedParamsError, match="drop_params=True"):
        if is_async:
            await config.async_transform_request(**request)
        else:
            config.transform_request(**request)
