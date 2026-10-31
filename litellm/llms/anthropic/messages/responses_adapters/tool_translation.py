import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final, TypeAlias

from pydantic import ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_logger
from litellm.exceptions import UnsupportedParamsError
from litellm.integrations.anthropic_cache_control_hook import targets_openai_api
from litellm.llms.anthropic.messages.utils import json_copy_or_original
from litellm.types.llms.anthropic import ANTHROPIC_TOOL_SEARCH_TOOL_TYPES
from litellm.types.utils import LlmProviders

_VERSIONED_TOOL_TYPE: Final = re.compile(r"(?P<family>[a-z0-9_]+?)_(?P<version>\d{8})")
_UNVERSIONED_SERVER_TOOL_TYPES: Final = frozenset({"mcp_toolset", "computer_toolset"})
_DEFERRAL_DROP_PATH: Final = re.compile(r"tools\[[^\]]*\]\.defer_loading")
_DropPaths: TypeAlias = list[object] | tuple[object, ...]
_DROP_PATHS: Final[TypeAdapter[_DropPaths]] = TypeAdapter(_DropPaths, config=ConfigDict(strict=True))
_TEXT_EDITOR_INSERT_TEXT_SINCE: Final = "20250429"
_COMPUTER_MODIFIERS_SINCE: Final = "20250124"
_COMPUTER_ZOOM_SINCE: Final = "20251124"
_COMPUTER_ACTIONS_20241022: Final = (
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
)
_COMPUTER_ACTIONS_20250124: Final = (
    *_COMPUTER_ACTIONS_20241022,
    "left_mouse_down",
    "left_mouse_up",
    "scroll",
    "hold_key",
    "wait",
    "triple_click",
)
_COMPUTER_ACTIONS_20251124: Final = (*_COMPUTER_ACTIONS_20250124, "zoom")


@dataclass(frozen=True, slots=True)
class ResponsesTool:
    name: str
    payload: Mapping[str, object]
    choice: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class UntranslatableTool:
    name: str
    type: str


TranslatedTool: TypeAlias = ResponsesTool | UntranslatableTool


def _family_and_version(tool_type: str) -> tuple[str, str]:
    versioned: Final = _VERSIONED_TOOL_TYPE.fullmatch(tool_type)
    if versioned is None:
        return tool_type, ""
    return versioned["family"], versioned["version"]


def _string(description: str) -> Mapping[str, object]:
    return {"type": "string", "description": description}


def _integer(description: str) -> Mapping[str, object]:
    return {"type": "integer", "description": description}


def _integers(count: int, description: str) -> Mapping[str, object]:
    return {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": count,
        "maxItems": count,
        "description": description,
    }


def _enum(values: Sequence[str]) -> Mapping[str, object]:
    return {"type": "string", "enum": list(values)}


def _function(name: str, description: str, properties: Mapping[str, object], required: Sequence[str]) -> ResponsesTool:
    return ResponsesTool(
        name=name,
        payload={
            "type": "function",
            "name": name,
            "strict": False,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": list(required)},
        },
        choice={"type": "function", "name": name},
    )


def _hosted(name: str, payload: Mapping[str, object]) -> ResponsesTool:
    return ResponsesTool(name=name, payload=payload, choice={"type": payload["type"]})


def _bash_tool(name: str) -> ResponsesTool:
    return _function(
        name,
        "Run a command in a persistent bash session. The working directory, environment, and files carry over "
        "between calls.",
        {
            "command": _string("The bash command to run. Required unless restart is true."),
            "restart": {"type": "boolean", "description": "Set to true to restart the bash session."},
        },
        (),
    )


def _text_editor_tool(name: str, version: str) -> ResponsesTool:
    inserts_insert_text: Final = version >= _TEXT_EDITOR_INSERT_TEXT_SINCE
    edit_properties: Final[Mapping[str, object]] = {
        "command": _enum(
            ("view", "create", "str_replace", "insert")
            if inserts_insert_text
            else ("view", "create", "str_replace", "insert", "undo_edit")
        ),
        "path": _string("Absolute path to the file or directory."),
        "view_range": _integers(2, "Optional [start, end] line numbers for view. An end of -1 reads to the end."),
        "file_text": _string("Content of the new file, for create."),
        "old_str": _string("Exact text to replace, for str_replace. It must match exactly one place in the file."),
        "new_str": _string(
            "Replacement text, for str_replace."
            if inserts_insert_text
            else "Replacement text for str_replace, or the text to add for insert."
        ),
        "insert_line": _integer("Line number to insert after, for insert. 0 inserts at the start of the file."),
    }
    return _function(
        name,
        "View, create, and edit text files. view lists a directory or shows a file with line numbers.",
        (
            {**edit_properties, "insert_text": _string("Text to add, for insert.")}
            if inserts_insert_text
            else edit_properties
        ),
        ("command", "path"),
    )


def _memory_tool(name: str) -> ResponsesTool:
    return _function(
        name,
        "Read and write files in the /memories directory, which persists between conversations. View it before "
        "starting a task, and record progress and learnings as you work.",
        {
            "command": _enum(("view", "create", "str_replace", "insert", "delete", "rename")),
            "path": _string("Path under /memories, for every command but rename."),
            "view_range": _integers(2, "Optional [start, end] line numbers for view."),
            "file_text": _string("Content of the new file, for create."),
            "old_str": _string("Exact text to replace, for str_replace."),
            "new_str": _string("Replacement text, for str_replace."),
            "insert_line": _integer("Line number to insert after, for insert."),
            "insert_text": _string("Text to add, for insert."),
            "old_path": _string("Current path, for rename."),
            "new_path": _string("New path, for rename."),
        },
        ("command",),
    )


def _computer_properties(version: str, zoom: bool) -> Mapping[str, object]:
    pointer: Final[Mapping[str, object]] = {
        "text": _string("Text to type for type, or keys to press for key and hold_key, such as ctrl+s."),
        "coordinate": _integers(2, "[x, y] pixel position for a click, mouse_move, or the end of left_click_drag."),
        "start_coordinate": _integers(2, "[x, y] pixel position where left_click_drag starts."),
    }
    if version < _COMPUTER_MODIFIERS_SINCE:
        return {"action": _enum(_COMPUTER_ACTIONS_20241022), **pointer}
    modifiers: Final[Mapping[str, object]] = {
        **pointer,
        "scroll_direction": _enum(("up", "down", "left", "right")),
        "scroll_amount": _integer("Number of scroll steps, for scroll."),
        "text": _string(
            "Text to type for type, keys to press for key and hold_key, such as ctrl+s, or a modifier key to hold "
            "during scroll, such as shift."
        ),
        "duration": {"type": "number", "description": "Seconds to wait for wait, or to hold for hold_key."},
        "key": _string("Modifier key to hold during a click, such as shift."),
    }
    if not zoom:
        return {"action": _enum(_COMPUTER_ACTIONS_20250124), **modifiers}
    return {
        "action": _enum(_COMPUTER_ACTIONS_20251124),
        **modifiers,
        "region": _integers(4, "[x0, y0, x1, y1] screen region to show at full resolution, for zoom."),
    }


def _computer_tool(name: str, version: str, tool: Mapping[str, object]) -> ResponsesTool:
    width: Final = tool.get("display_width_px")
    height: Final = tool.get("display_height_px")
    display: Final = (
        f"a {width}x{height} pixel display" if isinstance(width, int) and isinstance(height, int) else "a display"
    )
    return _function(
        name,
        f"Control {display} with the mouse and keyboard, and take screenshots to see it. Coordinates are pixels "
        "from the top left.",
        _computer_properties(version, zoom=version >= _COMPUTER_ZOOM_SINCE and tool.get("enable_zoom") is True),
        ("action",),
    )


def _tool_search(name: str, tool: Mapping[str, object]) -> ResponsesTool:
    return ResponsesTool(name=name, payload=dict(tool), choice=None)


def _custom_function(name: str, tool: Mapping[str, object]) -> ResponsesTool:
    description: Final = tool.get("description")
    input_schema: Final = tool.get("input_schema")
    return ResponsesTool(
        name=name,
        payload={
            "type": "function",
            "name": name,
            "strict": bool(tool.get("strict")),
            **({"description": description} if "description" in tool else {}),
            **({"parameters": json_copy_or_original(input_schema)} if "input_schema" in tool else {}),
        },
        choice={"type": "function", "name": name},
    )


def is_tool_search(tool: Mapping[str, object]) -> bool:
    tool_type: Final = tool.get("type")
    return isinstance(tool_type, str) and tool_type in ANTHROPIC_TOOL_SEARCH_TOOL_TYPES


def forced_tool_name(tool_choice: object) -> str | None:
    match tool_choice:
        case {"type": "tool", "name": str() as name}:
            return name
        case _:
            return None


def _defer_loading(tool: Mapping[str, object], forced: str | None) -> object:
    if forced is not None and tool.get("name") == forced:
        return None
    if "defer_loading" in tool:
        return tool["defer_loading"]
    match tool.get("custom"):
        case {"defer_loading": bool() as wrapped}:
            return wrapped
        case _:
            return None


def _defers_a_function(tool: Mapping[str, object], forced: str | None) -> bool:
    translated: Final = _translated_tool(tool, False) if _defer_loading(tool, forced) is True else None
    return isinstance(translated, ResponsesTool) and translated.payload.get("type") == "function"


def keeps_tool_search(
    tools: Sequence[Mapping[str, object]], *, maps_tool_search: bool, forced: str | None = None
) -> bool:
    return (
        maps_tool_search
        and any(is_tool_search(tool) for tool in tools)
        and any(_defers_a_function(tool, forced) for tool in tools)
    )


def _with_deferral(translated: TranslatedTool, defer_loading: object) -> TranslatedTool:
    match translated:
        case ResponsesTool(payload=payload) if payload.get("type") == "function" and isinstance(defer_loading, bool):
            return replace(translated, payload={**payload, "defer_loading": defer_loading})
        case _:
            return translated


def responses_tool(
    tool: Mapping[str, object], *, runs_tool_search: bool = False, forced: str | None = None
) -> TranslatedTool:
    translated: Final = _translated_tool(tool, runs_tool_search)
    return _with_deferral(translated, _defer_loading(tool, forced)) if runs_tool_search else translated


def _translated_tool(tool: Mapping[str, object], runs_tool_search: bool) -> TranslatedTool:
    tool_type: Final = tool.get("type")
    tool_name: Final = tool.get("name")
    name: Final = tool_name if isinstance(tool_name, str) else ""
    type_name: Final = tool_type if isinstance(tool_type, str) else ""
    family, version = _family_and_version(type_name)
    if type_name.startswith("web_search") or name == "web_search":
        return _hosted(name, {"type": "web_search_preview"})
    if runs_tool_search and is_tool_search(tool):
        return _tool_search(name, tool)
    if not version and family not in _UNVERSIONED_SERVER_TOOL_TYPES:
        return _custom_function(name, tool)
    match family:
        case "bash":
            return _bash_tool(name)
        case "text_editor":
            return _text_editor_tool(name, version)
        case "memory":
            return _memory_tool(name)
        case "computer":
            return _computer_tool(name, version, tool)
        case _:
            return UntranslatableTool(name=name, type=type_name)


_DROP_PARAMS_HOW_TO: Final = (
    "set `drop_params: true` in the model's `litellm_params` or under `litellm_settings` on the proxy, or "
    "`litellm.drop_params=True` in the SDK"
)


def _tool_types(tools: Sequence[UntranslatableTool]) -> str:
    return ", ".join(dict.fromkeys(tool.type for tool in tools))


def untranslatable_tools_error(tools: Sequence[UntranslatableTool]) -> UnsupportedParamsError:
    return UnsupportedParamsError(
        message=(
            f"LiteLLM does not translate these Anthropic tools to the Responses API: {_tool_types(tools)}. "
            f"To drop them, {_DROP_PARAMS_HOW_TO}"
        )
    )


def log_dropped_tools(tools: Sequence[UntranslatableTool]) -> None:
    verbose_logger.warning(
        "Dropping Anthropic tools the Responses API cannot run under drop_params: %s", _tool_types(tools)
    )


def unforceable_tool_choice_error(name: str) -> UnsupportedParamsError:
    return UnsupportedParamsError(
        message=f"OpenAI cannot force the tool search tool {name} through `tool_choice`. To drop it, {_DROP_PARAMS_HOW_TO}"
    )


def _drops_deferral(additional_drop_params: object) -> bool:
    try:
        paths: Final = _DROP_PATHS.validate_python(additional_drop_params)
    except ValidationError:
        return False
    return any(isinstance(path, str) and _DEFERRAL_DROP_PATH.fullmatch(path) is not None for path in paths)


def maps_openai_tool_search(model: str, api_base: object, additional_drop_params: object) -> bool:
    return (
        targets_openai_api(api_base)
        and not _drops_deferral(additional_drop_params)
        and litellm.supports_tool_search(model=model, custom_llm_provider=LlmProviders.OPENAI.value)
    )
