from typing import Final

import pytest

from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig


@pytest.mark.parametrize(
    "model", ("deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash-vision-exp", "deepseek/deepseek-v4-flash")
)
@pytest.mark.parametrize("effort", ("minimal", "low", "medium", "high", "xhigh", "max", "ultra"))
@pytest.mark.parametrize("thinking", (None, "enabled"))
def test_v4_keeps_requested_effort_with_enabled_thinking(model: str, effort: str, thinking: str | None) -> None:
    params: Final = {"reasoning_effort": effort, **({"thinking": {"type": thinking}} if thinking else {})}
    result: Final = DeepSeekChatConfig().map_openai_params(params, {}, model, False)
    assert result == {"thinking": {"type": "enabled"}, "reasoning_effort": effort}


@pytest.mark.parametrize(
    "thinking,effort,expected",
    (
        (None, None, {}),
        (None, "none", {"thinking": {"type": "disabled"}}),
        ("disabled", "low", {"thinking": {"type": "disabled"}}),
        ("disabled", "max", {"thinking": {"type": "disabled"}}),
        ("enabled", "none", {"thinking": {"type": "enabled"}}),
        ("enabled", None, {"thinking": {"type": "enabled"}}),
    ),
)
def test_v4_retains_explicit_thinking_precedence(
    thinking: str | None, effort: str | None, expected: dict[str, dict[str, str]]
) -> None:
    params: Final = {"reasoning_effort": effort, **({"thinking": {"type": thinking}} if thinking else {})}
    result: Final = DeepSeekChatConfig().map_openai_params(params, {}, "deepseek-flash", False)
    assert result == expected


@pytest.mark.parametrize("model", ("deepseek-chat", "deepseek-reasoner", "deepseek/deepseek-reasoner"))
def test_legacy_models_retain_toggle_only_mapping(model: str) -> None:
    result: Final = DeepSeekChatConfig().map_openai_params({"reasoning_effort": "low"}, {}, model, False)
    assert result == {"thinking": {"type": "enabled"}}
