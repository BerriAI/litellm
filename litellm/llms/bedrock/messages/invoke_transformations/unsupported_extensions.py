from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from types import MappingProxyType
from typing import Annotated, Final, NoReturn, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

import litellm
from litellm.constants import ANTHROPIC_MID_CONVERSATION_TOOL_CHANGE_BLOCK_TYPES
from litellm.litellm_core_utils.prompt_templates.factory import DEFAULT_USER_CONTINUE_MESSAGE
from litellm.types.llms.anthropic import ANTHROPIC_BETA_HEADER_VALUES


class OptInKnob(Enum):
    DROP_PARAMS = "drop_params"
    MODIFY_PARAMS = "modify_params"


class Extension(Enum):
    MESSAGE_OUTPUT_CONFIG = ANTHROPIC_BETA_HEADER_VALUES.MID_CONVERSATION_OUTPUT_CONFIG_2026_07_01.value
    THINKING_DISPLAY_UPDATES = ANTHROPIC_BETA_HEADER_VALUES.THINKING_DISPLAY_UPDATES_2026_08_18.value
    TOOL_CHANGES = ANTHROPIC_BETA_HEADER_VALUES.MID_CONVERSATION_TOOL_CHANGES_2026_07_01.value

    @property
    def beta(self) -> str:
        return self.value

    @property
    def knob(self) -> OptInKnob:
        return OptInKnob.MODIFY_PARAMS if self is Extension.TOOL_CHANGES else OptInKnob.DROP_PARAMS


@dataclass(frozen=True, slots=True)
class OptIns:
    drop_params: bool
    modify_params: bool

    def allows(self, knob: OptInKnob) -> bool:
        return self.drop_params if knob is OptInKnob.DROP_PARAMS else self.modify_params


@dataclass(frozen=True, slots=True)
class Offender:
    path: str
    extension: Extension


@dataclass(frozen=True, slots=True)
class Unchanged:
    pass


@dataclass(frozen=True, slots=True)
class Sanitized:
    replacements: Mapping[str, JsonValue]
    removed: tuple[Offender, ...]


@dataclass(frozen=True, slots=True)
class Refused:
    offenders: tuple[Offender, ...]


SanitizeOutcome: TypeAlias = Unchanged | Sanitized | Refused


class _Block(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    type: str

    @property
    def is_tool_change(self) -> bool:
        return self.type in ANTHROPIC_MID_CONVERSATION_TOOL_CHANGE_BLOCK_TYPES


class _Message(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    role: str | None = None
    content: str | tuple[Annotated[_Block | JsonValue, Field(union_mode="left_to_right")], ...] | None = None
    output_config: JsonValue = None


class _Thinking(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    display: JsonValue = None


class _Request(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    messages: tuple[Annotated[_Message | JsonValue, Field(union_mode="left_to_right")], ...] | None = None
    thinking: Annotated[_Thinking | JsonValue, Field(union_mode="left_to_right")] = None


_PLACEHOLDER_BLOCK: Final = _Block.model_validate(
    MappingProxyType({"type": "text", "text": DEFAULT_USER_CONTINUE_MESSAGE["content"]})
)
_WITHOUT_OUTPUT_CONFIG: Final = MappingProxyType({"output_config": True})
_WITHOUT_DISPLAY: Final = MappingProxyType({"display": True})
_SANITIZED_FIELDS: Final = MappingProxyType({"messages": True, "thinking": True})


def _asks_for_display_updates(thinking: _Thinking | JsonValue) -> bool:
    return isinstance(thinking, _Thinking) and thinking.display == "updates"


def _blocks(message: _Message) -> tuple[_Block | JsonValue, ...]:
    return message.content if isinstance(message.content, tuple) else ()


def _is_tool_change(block: _Block | JsonValue) -> bool:
    return isinstance(block, _Block) and block.is_tool_change


def _thinking_offenders(thinking: _Thinking | JsonValue) -> tuple[Offender, ...]:
    if not _asks_for_display_updates(thinking):
        return ()
    return (Offender(path="thinking.display (value 'updates')", extension=Extension.THINKING_DISPLAY_UPDATES),)


def _message_offenders(index: int, message: _Message | JsonValue) -> tuple[Offender, ...]:
    if not isinstance(message, _Message):
        return ()
    output_config: Final = (
        (Offender(path=f"messages[{index}].output_config", extension=Extension.MESSAGE_OUTPUT_CONFIG),)
        if "output_config" in message.model_fields_set
        else ()
    )
    blocks: Final = tuple(
        Offender(
            path=f"messages[{index}].content[{block_index}] (type '{block.type}')", extension=Extension.TOOL_CHANGES
        )
        for block_index, block in enumerate(_blocks(message))
        if isinstance(block, _Block) and block.is_tool_change
    )
    return output_config + blocks


def _offenders(request: _Request) -> tuple[Offender, ...]:
    per_message: Final = chain.from_iterable(
        _message_offenders(index, message) for index, message in enumerate(request.messages or ())
    )
    return _thinking_offenders(request.thinking) + tuple(per_message)


def _kept_blocks(message: _Message) -> tuple[_Block | JsonValue, ...]:
    blocks: Final = _blocks(message)
    kept: Final = tuple(block for block in blocks if not _is_tool_change(block))
    return kept if kept or message.role == "system" else (_PLACEHOLDER_BLOCK,)


def _sanitized_message(message: _Message | JsonValue, strip: frozenset[Extension]) -> tuple[JsonValue, ...]:
    if not isinstance(message, _Message):
        return (message,)
    drops_output_config: Final = (
        Extension.MESSAGE_OUTPUT_CONFIG in strip and "output_config" in message.model_fields_set
    )
    drops_blocks: Final = Extension.TOOL_CHANGES in strip and any(_is_tool_change(b) for b in _blocks(message))
    content: Final = _kept_blocks(message) if drops_blocks else message.content
    if message.role == "system" and (drops_output_config or drops_blocks) and not content:
        return ()
    kept: Final = message.model_copy(update=MappingProxyType({"content": content})) if drops_blocks else message
    return (
        kept.model_dump(
            mode="json", exclude_unset=True, exclude=_WITHOUT_OUTPUT_CONFIG if drops_output_config else None
        ),
    )


def _sanitized_thinking(thinking: _Thinking | JsonValue, strip: frozenset[Extension]) -> _Thinking | JsonValue:
    if Extension.THINKING_DISPLAY_UPDATES not in strip or not isinstance(thinking, _Thinking):
        return thinking
    if not _asks_for_display_updates(thinking):
        return thinking
    return thinking.model_dump(mode="json", exclude_unset=True, exclude=_WITHOUT_DISPLAY)


def _replacements(request: _Request, strip: frozenset[Extension]) -> Mapping[str, JsonValue]:
    messages: Final = tuple(
        chain.from_iterable(_sanitized_message(message, strip) for message in request.messages or ())
    )
    sanitized: Final = (("messages", messages), ("thinking", _sanitized_thinking(request.thinking, strip)))
    update: Final = MappingProxyType({field: value for field, value in sanitized if field in request.model_fields_set})
    return MappingProxyType(
        request.model_copy(update=update).model_dump(mode="json", exclude_unset=True, include=_SANITIZED_FIELDS)
    )


def _view(request: Mapping[str, object]) -> _Request | None:
    try:
        return _Request.model_validate(request)
    except ValidationError:
        return None


def sanitize_for_bedrock_invoke(
    request: Mapping[str, object], opt_ins: OptIns, supported: frozenset[Extension]
) -> SanitizeOutcome:
    view: Final = _view(request)
    if view is None:
        return Unchanged()
    unsupported: Final = tuple(offender for offender in _offenders(view) if offender.extension not in supported)
    blocked: Final = tuple(offender for offender in unsupported if not opt_ins.allows(offender.extension.knob))
    if blocked:
        return Refused(offenders=blocked)
    if not unsupported:
        return Unchanged()
    strip: Final = frozenset(offender.extension for offender in unsupported)
    return Sanitized(replacements=_replacements(view, strip), removed=unsupported)


def extension_betas(request: Mapping[str, object]) -> frozenset[str]:
    view: Final = _view(request)
    return frozenset(offender.extension.beta for offender in _offenders(view)) if view is not None else frozenset()


def _paths(refused: Refused, knob: OptInKnob) -> str:
    return ", ".join(offender.path for offender in refused.offenders if offender.extension.knob is knob)


def _drop_params_hint(paths: str, model: str) -> str:
    if not paths:
        return ""
    return (
        f"Bedrock Invoke does not accept {paths} on {model}. Set `drop_params: true` in this deployment's "
        "`litellm_params` (or `litellm_settings.drop_params: true` on the proxy, `litellm.drop_params = True` in the "
        "SDK) to have LiteLLM drop them (thinking.display falls back to the model default, and a system message "
        "left empty is removed), or remove them from the request."
    )


def _modify_params_hint(paths: str, model: str) -> str:
    if not paths:
        return ""
    return (
        f"Bedrock Invoke does not accept {paths} on {model}. Set `litellm_settings.modify_params: true` on the "
        "proxy or `litellm.modify_params = True` in the SDK to have LiteLLM remove those blocks (a system message "
        "left empty is removed), or remove them from the request."
    )


def raise_refusal(refused: Refused, model: str) -> NoReturn:
    param_paths: Final = _paths(refused, OptInKnob.DROP_PARAMS)
    block_paths: Final = _paths(refused, OptInKnob.MODIFY_PARAMS)
    hints: Final = tuple(
        hint for hint in (_drop_params_hint(param_paths, model), _modify_params_hint(block_paths, model)) if hint
    )
    error_class: Final = litellm.UnsupportedParamsError if param_paths else litellm.BadRequestError
    raise error_class(message=" ".join(hints), model=model, llm_provider="bedrock")
