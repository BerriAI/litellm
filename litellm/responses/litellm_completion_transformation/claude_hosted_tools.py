import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal

from openai.types.responses import ResponseComputerToolCall
from openai.types.responses.response_computer_tool_call import (
    Action,
    ActionClick,
    ActionDoubleClick,
    ActionDrag,
    ActionDragPath,
    ActionKeypress,
    ActionMove,
    ActionScreenshot,
    ActionScroll,
    ActionType,
    ActionWait,
)
from pydantic import ConfigDict, TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import message_field, parts_of
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.openai import (
    ChatCompletionResponseMessage,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
)

CLAUDE_COMPUTER_TOOLSET_TYPE: Final = "computer_toolset_20260801"
CLAUDE_CODE_EXECUTION_TOOL_TYPE: Final = "code_execution_20260521"
COMPUTER_TOOLSET_NAME: Final = "computer"
COMPUTER_MEMBERS_WITHOUT_AN_OPENAI_ACTION: Final = (
    "zoom",
    "triple_click",
    "left_mouse_down",
    "left_mouse_up",
    "cursor_position",
    "hold_key",
)
SCROLL_PIXELS_PER_WHEEL_CLICK: Final = 100
REPLAYED_WAIT_SECONDS: Final = 1

_COMPUTER_CALL_ITEM_ID_PREFIX: Final = "cu_"
_DROP_PARAMS_HINT: Final = "set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n"
_CLICK_BUTTON_BY_MEMBER: Final[Mapping[str, Literal["left", "right", "wheel"]]] = MappingProxyType(
    {"left_click": "left", "right_click": "right", "middle_click": "wheel"}
)
_CLICK_MEMBER_BY_BUTTON: Final = MappingProxyType(
    {button: member for member, button in _CLICK_BUTTON_BY_MEMBER.items()}
)
_SCROLL_VECTOR_BY_DIRECTION: Final = MappingProxyType({"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)})
_OPENAI_KEY_BY_XDOTOOL_KEY: Final = MappingProxyType(
    {
        "Return": "ENTER",
        "Escape": "ESC",
        "BackSpace": "BACKSPACE",
        "Tab": "TAB",
        "space": "SPACE",
        "Delete": "DELETE",
        "Home": "HOME",
        "End": "END",
        "Page_Up": "PAGEUP",
        "Page_Down": "PAGEDOWN",
        "Up": "ARROWUP",
        "Down": "ARROWDOWN",
        "Left": "ARROWLEFT",
        "Right": "ARROWRIGHT",
        "ctrl": "CTRL",
        "alt": "ALT",
        "shift": "SHIFT",
        "super": "CMD",
    }
)
_XDOTOOL_KEY_BY_OPENAI_KEY: Final = MappingProxyType(
    {
        **{openai_key: xdotool_key for xdotool_key, openai_key in _OPENAI_KEY_BY_XDOTOOL_KEY.items()},
        "ENTER": "Return",
        "ESCAPE": "Escape",
        "CONTROL": "ctrl",
        "OPTION": "alt",
        "META": "super",
        "SUPER": "super",
        "WIN": "super",
    }
)
_ACTION_ADAPTER: Final[TypeAdapter[Action]] = TypeAdapter(Action)


class _ClaudeMemberInput(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    coordinate: tuple[int, int] | None = None
    start_coordinate: tuple[int, int] | None = None
    text: str | None = None
    scroll_direction: Literal["up", "down", "left", "right"] | None = None
    scroll_amount: int | None = None


def claude_tools_for_openai_hosted_tools(
    tools: Sequence[Mapping[str, object]], drop_params: bool
) -> tuple[Mapping[str, object], ...]:
    return tuple(_claude_tool_for(tool, drop_params) for tool in tools)


def _claude_tool_for(tool: Mapping[str, object], drop_params: bool) -> Mapping[str, object]:
    match tool.get("type"):
        case "computer_use_preview":
            return {
                "type": CLAUDE_COMPUTER_TOOLSET_TYPE,
                "configs": {member: {"enabled": False} for member in COMPUTER_MEMBERS_WITHOUT_AN_OPENAI_ACTION},
            }
        case "code_interpreter":
            return _claude_code_execution_tool(_container_settings_claude_lacks(tool.get("container")), drop_params)
        case _:
            return tool


def _claude_code_execution_tool(settings_claude_lacks: tuple[str, ...], drop_params: bool) -> Mapping[str, object]:
    named_settings: Final = ", ".join(f"`{setting}`" for setting in settings_claude_lacks)
    if settings_claude_lacks and not drop_params:
        raise UnsupportedParamsError(
            message=(
                f"The `code_interpreter` tool's {named_settings} has no equivalent in Claude's code execution tool. "
                f"To drop it and keep code execution, {_DROP_PARAMS_HINT}"
            ),
            status_code=400,
        )
    if settings_claude_lacks:
        verbose_logger.warning(
            "Dropping the `code_interpreter` tool's %s (drop_params=True) because Claude's code execution tool "
            "has no equivalent of it.",
            named_settings,
        )
    return {"type": CLAUDE_CODE_EXECUTION_TOOL_TYPE, "name": "code_execution"}


def _container_settings_claude_lacks(container: object) -> tuple[str, ...]:
    if container is None:
        return ()
    settings: Final = as_str_mapping(container)
    if settings is None or settings.get("type") != "auto":
        return ("container",)
    return tuple(
        f"container.{key}"
        for key, value in settings.items()
        if key != "type" and not _claude_matches_container_setting(key, value)
    )


def _claude_matches_container_setting(key: str, value: object) -> bool:
    match key, value:
        case ("file_ids", [] | None) | ("memory_limit", None) | ("network_policy", {"type": "disabled"} | None):
            return True
        case _:
            return False


def requests_openai_computer_use(tools: object) -> bool:
    return any(message_field(tool, "type") == "computer_use_preview" for tool in parts_of(tools))


def is_computer_member_call(tool_call: object) -> bool:
    match message_field(tool_call, "provider_specific_fields"):
        case {"toolset_name": "computer"}:
            return True
        case _:
            return False


def openai_computer_call(
    call_id: str, member: str, arguments: str, status: Literal["in_progress", "completed"]
) -> ResponseComputerToolCall | None:
    try:
        member_input: Final = _ClaudeMemberInput.model_validate_json(arguments or "{}")
    except ValidationError:
        return None
    action: Final = openai_computer_action(member, member_input)
    if action is None:
        return None
    return ResponseComputerToolCall(
        id=f"{_COMPUTER_CALL_ITEM_ID_PREFIX}{call_id}",
        call_id=call_id,
        action=action,
        pending_safety_checks=[],
        status=status,
        type="computer_call",
    )


def openai_computer_action(member: str, member_input: _ClaudeMemberInput) -> Action | None:
    held_keys: Final = _held_openai_keys(member_input)
    match member, member_input.coordinate, member_input.text:
        case "screenshot", _, _:
            return ActionScreenshot(type="screenshot")
        case "wait", _, _:
            return ActionWait(type="wait")
        case "type", _, str() as text:
            return ActionType(type="type", text=text)
        case "key", _, str() as chord:
            return ActionKeypress(type="keypress", keys=list(_openai_keys(chord)))
        case ("left_click" | "right_click" | "middle_click") as click, (x, y), _:
            return ActionClick(
                type="click", button=_CLICK_BUTTON_BY_MEMBER[click], x=x, y=y, keys=list(held_keys) or None
            )
        case "double_click", (x, y), _:
            return ActionDoubleClick(type="double_click", x=x, y=y, keys=list(held_keys) or None)
        case "mouse_move", (x, y), _:
            return ActionMove(type="move", x=x, y=y)
        case "left_click_drag", (x, y), _:
            return _openai_drag(member_input, x, y)
        case "scroll", (x, y), _:
            return _openai_scroll(x, y, member_input)
        case _:
            return None


def _held_openai_keys(member_input: _ClaudeMemberInput) -> tuple[str, ...]:
    return _openai_keys(member_input.text) if member_input.text else ()


def _openai_keys(chord: str) -> tuple[str, ...]:
    return tuple(_OPENAI_KEY_BY_XDOTOOL_KEY.get(key, key) for key in chord.split("+"))


def _openai_drag(member_input: _ClaudeMemberInput, x: int, y: int) -> ActionDrag | None:
    start: Final = member_input.start_coordinate
    if start is None:
        return None
    return ActionDrag(
        type="drag",
        path=[ActionDragPath(x=start[0], y=start[1]), ActionDragPath(x=x, y=y)],
        keys=list(_held_openai_keys(member_input)) or None,
    )


def _openai_scroll(x: int, y: int, member_input: _ClaudeMemberInput) -> ActionScroll | None:
    if member_input.scroll_direction is None or member_input.scroll_amount is None:
        return None
    step_x, step_y = _SCROLL_VECTOR_BY_DIRECTION[member_input.scroll_direction]
    distance: Final = member_input.scroll_amount * SCROLL_PIXELS_PER_WHEEL_CLICK
    return ActionScroll(
        type="scroll",
        x=x,
        y=y,
        scroll_x=step_x * distance,
        scroll_y=step_y * distance,
        keys=list(_held_openai_keys(member_input)) or None,
    )


def claude_computer_member_call_message(computer_call: Mapping[str, object]) -> ChatCompletionResponseMessage:
    call_id: Final = computer_call.get("call_id") or computer_call.get("id")
    member, member_input = claude_computer_member_call(computer_call.get("action"))
    return ChatCompletionResponseMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ChatCompletionToolCallChunk(
                id=call_id if isinstance(call_id, str) else "",
                type="function",
                function=ChatCompletionToolCallFunctionChunk(name=member, arguments=json.dumps(member_input)),
                index=0,
                provider_specific_fields={"toolset_name": COMPUTER_TOOLSET_NAME},
            )
        ],
    )


def claude_computer_member_call(action: object) -> tuple[str, Mapping[str, object]]:
    try:
        parsed: Final = _ACTION_ADAPTER.validate_python(action)
    except ValidationError as error:
        raise UnsupportedParamsError(
            message=f"A `computer_call` input item has no valid `action`: {error}", status_code=400
        ) from error
    member_call: Final = _claude_member_call_for(parsed)
    if member_call is None:
        raise UnsupportedParamsError(
            message=f"The `computer_call` action {parsed.model_dump_json()} has no equivalent in Claude's computer use.",
            status_code=400,
        )
    return member_call


def _claude_member_call_for(action: Action) -> tuple[str, Mapping[str, object]] | None:
    match action:
        case ActionScreenshot():
            return "screenshot", {}
        case ActionWait():
            return "wait", {"duration": REPLAYED_WAIT_SECONDS}
        case ActionType(text=text):
            return "type", {"text": text}
        case ActionKeypress(keys=keys):
            return "key", {"text": _xdotool_chord(keys)}
        case ActionClick(button=button, x=x, y=y, keys=keys):
            return _claude_click(_CLICK_MEMBER_BY_BUTTON.get(button), x, y, keys)
        case ActionDoubleClick(x=x, y=y, keys=keys):
            return "double_click", {"coordinate": [x, y], **_claude_held_keys(keys)}
        case ActionMove(x=x, y=y):
            return "mouse_move", {"coordinate": [x, y]}
        case ActionScroll(x=x, y=y, scroll_x=scroll_x, scroll_y=scroll_y, keys=keys):
            return "scroll", {"coordinate": [x, y], **_claude_scroll(scroll_x, scroll_y), **_claude_held_keys(keys)}
        case ActionDrag(path=[start, *_, end], keys=keys):
            return "left_click_drag", {
                "start_coordinate": [start.x, start.y],
                "coordinate": [end.x, end.y],
                **_claude_held_keys(keys),
            }
        case _:
            return None


def _claude_click(
    member: str | None, x: int, y: int, keys: Sequence[str] | None
) -> tuple[str, Mapping[str, object]] | None:
    if member is None:
        return None
    return member, {"coordinate": [x, y], **_claude_held_keys(keys)}


def _claude_held_keys(keys: Sequence[str] | None) -> Mapping[str, object]:
    return {"text": _xdotool_chord(keys)} if keys else {}


def _xdotool_chord(keys: Sequence[str]) -> str:
    return "+".join(_XDOTOOL_KEY_BY_OPENAI_KEY.get(key.upper(), key) for key in keys)


def _claude_scroll(scroll_x: int, scroll_y: int) -> Mapping[str, object]:
    vertical: Final = abs(scroll_y) >= abs(scroll_x)
    distance: Final = scroll_y if vertical else scroll_x
    forward_direction: Final = "down" if vertical else "right"
    backward_direction: Final = "up" if vertical else "left"
    return {
        "scroll_direction": forward_direction if distance >= 0 else backward_direction,
        "scroll_amount": max(1, round(abs(distance) / SCROLL_PIXELS_PER_WHEEL_CLICK)),
    }
