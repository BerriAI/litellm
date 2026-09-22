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
    (
        "model",
        "thinking",
        "tool_choice",
        "request_drop_params",
        "global_drop_params",
        "expected",
    ),
    [
        (
            "deepseek-v4-pro",
            {"type": "enabled"},
            "required",
            True,
            False,
            "auto",
        ),
        (
            "deepseek-v4-pro",
            {"type": "enabled"},
            "required",
            False,
            True,
            "auto",
        ),
        (
            "deepseek-v4-pro",
            {"type": "enabled"},
            {"type": "function", "function": {"name": "shell"}},
            True,
            False,
            "auto",
        ),
        ("deepseek-v4-pro", {"type": "enabled"}, "none", False, False, "none"),
        ("deepseek-v4-pro", {"type": "enabled"}, "auto", False, False, "auto"),
        ("deepseek-v4-pro", {"type": "enabled"}, None, False, False, None),
        (
            "deepseek-v4-pro",
            {"type": "disabled"},
            "required",
            False,
            False,
            "required",
        ),
        (
            "deepseek-chat",
            {"type": "enabled"},
            "required",
            False,
            False,
            "required",
        ),
    ],
)
async def test_transform_request_normalizes_tool_choice_for_thinking(
    is_async: bool,
    model: str,
    thinking: dict[str, str],
    tool_choice: object | None,
    request_drop_params: bool,
    global_drop_params: bool,
    expected: object | None,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "drop_params", global_drop_params)
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
        "litellm_params": {"drop_params": request_drop_params},
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

    if is_async:
        with pytest.raises(litellm.UnsupportedParamsError, match="drop_params=True"):
            await config.async_transform_request(**request)
    else:
        with pytest.raises(litellm.UnsupportedParamsError, match="drop_params=True"):
            config.transform_request(**request)
