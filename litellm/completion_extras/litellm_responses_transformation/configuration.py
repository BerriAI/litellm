import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from litellm.integrations.anthropic_cache_control_hook import CostMap, routed_cost_map, routed_model_info
from litellm.llms.openai.chat.gpt_reasoning_transformation import OpenAIGPT5Config
from litellm.responses.additional_tools import tool_identity
from litellm.responses.configuration_update import (
    ReasoningUpdate,
    TextUpdate,
    additional_tools_item,
    configuration_update_item,
)

CONFIGURATION_FIELD: Final = "openai_responses_configuration"
SUPPORTS_CONFIGURATION_UPDATE: Final = "supports_configuration_update_input_item"
_INLINE_TOOL_TYPES: Final = frozenset({"function", "custom"})
_JSON_OBJECT: Final = "json_object"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class ToolRef(_Frozen):
    identity: str
    digest: str
    inline: bool


class Settings(_Frozen):
    reasoning: Mapping[str, object] | None = None
    text: Mapping[str, object] | None = None
    tools: tuple[ToolRef, ...] | None = None


class ConfigurationNote(_Frozen):
    version: Literal[1]
    base: Settings
    effective: Settings | None = None
    starts: bool = False

    def in_effect(self) -> Settings:
        return self.effective or self.base

    def as_fields(self) -> Mapping[str, object]:
        return MappingProxyType({CONFIGURATION_FIELD: self.model_dump(mode="json", exclude_defaults=True)})


def _starting_note(settings: Settings) -> ConfigurationNote:
    return ConfigurationNote(version=1, base=settings, starts=True)


def _continuing_note(base: Settings, effective: Settings) -> ConfigurationNote:
    return ConfigurationNote(version=1, base=base, effective=None if effective == base else effective)


class _Requested(_Frozen):
    reasoning: Mapping[str, object] | None = None
    text: Mapping[str, object] | None = None
    tools: tuple[Mapping[str, object], ...] | None = None
    temperature: object = None
    top_p: object = None

    def samples(self) -> bool:
        return self.temperature is not None or self.top_p is not None


class _NoteFields(_Frozen):
    note: object = Field(default=None, alias=CONFIGURATION_FIELD)


class _Message(_Frozen):
    role: object = None
    provider_specific_fields: object = None


@dataclass(frozen=True, slots=True)
class ConfigurationPlan:
    top_level: Mapping[str, object]
    inserts: Mapping[int, tuple[Mapping[str, object], ...]]
    note: ConfigurationNote


@dataclass(frozen=True, slots=True)
class _Unstamped:
    pass


@dataclass(frozen=True, slots=True)
class _Stamped:
    index: int
    note: ConfigurationNote


_Assistant: TypeAlias = _Unstamped | _Stamped


@dataclass(frozen=True, slots=True)
class _Unexpressible:
    pass


@dataclass(frozen=True, slots=True)
class _Same:
    pass


_UNEXPRESSIBLE: Final = _Unexpressible()
_SAME: Final = _Same()


@dataclass(frozen=True, slots=True)
class _Transition:
    reasoning: ReasoningUpdate | None
    text: TextUpdate | None
    added: tuple[str, ...]


def _cost_map_entry(cost_map: CostMap, name: str, provider: str | None) -> Mapping[str, object] | None:
    keys: Final = (*((f"{provider}/{name}",) if provider else ()), name, name.rsplit("/", 1)[-1])
    return next((cost_map[key] for key in keys if key in cost_map), None)


def updates_configuration_inline(model: str, provider: str | None, litellm_params: Mapping[str, object]) -> bool:
    model_info: Final = routed_model_info(litellm_params.get("model_info")) or {}
    declared: Final = model_info.get(SUPPORTS_CONFIGURATION_UPDATE)
    if isinstance(declared, bool):
        return declared
    candidates: Final = (litellm_params.get("base_model"), model_info.get("base_model"), model)
    names: Final = tuple(dict.fromkeys(name for name in candidates if isinstance(name, str) and name))
    cost_map: Final = routed_cost_map()
    entries: Final = (_cost_map_entry(cost_map, name, provider) for name in names)
    listed: Final = next((entry for entry in entries if entry is not None), None)
    if listed is not None:
        return listed.get(SUPPORTS_CONFIGURATION_UPDATE) is True
    return any(OpenAIGPT5Config.is_model_gpt_6_plus_model(name) for name in names)


def _digest(tool: Mapping[str, object]) -> str:
    encoded: Final = json.dumps(tool, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8", "surrogatepass")).hexdigest()


def _is_inline_addable(tool: Mapping[str, object]) -> bool:
    return tool.get("type") in _INLINE_TOOL_TYPES and "defer_loading" not in tool


def _tool_ref(tool: Mapping[str, object]) -> ToolRef:
    return ToolRef(identity=tool_identity(tool), digest=_digest(tool), inline=_is_inline_addable(tool))


def _json_normalized(settings: Settings) -> Settings:
    return Settings.model_validate(settings.model_dump(mode="json"))


def _settings(requested: _Requested) -> Settings:
    return _json_normalized(
        Settings(
            reasoning=requested.reasoning,
            text=requested.text,
            tools=None if requested.tools is None else tuple(map(_tool_ref, requested.tools)),
        )
    )


def _as_json(value: object) -> object:
    return value.model_dump() if isinstance(value, BaseModel) else value


def _message(message: object) -> _Message:
    try:
        return _Message.model_validate(_as_json(message))
    except ValidationError:
        return _Message()


def _assistant(index: int, message: _Message) -> _Assistant:
    try:
        raw_note: Final = _NoteFields.model_validate(_as_json(message.provider_specific_fields)).note
    except ValidationError:
        return _Unstamped()
    if raw_note is None:
        return _Unstamped()
    try:
        return _Stamped(index, ConfigurationNote.model_validate(raw_note))
    except ValidationError:
        return _Unstamped()


def _assistants_newest_first(messages: Sequence[_Message]) -> Iterator[_Assistant]:
    for index in reversed(range(len(messages))):
        if messages[index].role == "assistant":
            yield _assistant(index, messages[index])


def _stretch_newest_first(assistants: Iterator[_Assistant]) -> Iterator[_Assistant]:
    for assistant in assistants:
        yield assistant
        if isinstance(assistant, _Stamped) and assistant.note.starts:
            return


def _segment(messages: Sequence[_Message]) -> tuple[_Stamped, ...] | None:
    stretch: Final = tuple(_stretch_newest_first(_assistants_newest_first(messages)))[::-1]
    stamped: Final = tuple(assistant for assistant in stretch if isinstance(assistant, _Stamped))
    if not stamped or len(stamped) != len(stretch):
        return None
    start: Final = stamped[0].note
    if not start.starts or start.effective is not None or any(stamp.note.base != start.base for stamp in stamped):
        return None
    return stamped


def _without(settings: Mapping[str, object] | None, key: str) -> Mapping[str, object]:
    return MappingProxyType({name: value for name, value in (settings or {}).items() if name != key})


def _reasoning_change(
    previous: Mapping[str, object] | None, following: Mapping[str, object] | None
) -> ReasoningUpdate | _Same | _Unexpressible:
    if _without(previous, "effort") != _without(following, "effort"):
        return _UNEXPRESSIBLE
    previous_effort: Final = (previous or {}).get("effort")
    following_effort: Final = (following or {}).get("effort")
    if previous_effort == following_effort:
        return _SAME
    if not isinstance(following_effort, str):
        return _UNEXPRESSIBLE
    return ReasoningUpdate(effort=following_effort)


def _default_text_format() -> Mapping[str, object]:
    return {"type": "text"}


class _TextFormat(_Frozen):
    format: Mapping[str, object] = Field(default_factory=_default_text_format)


def _text_format(text: Mapping[str, object] | None) -> Mapping[str, object] | None:
    try:
        return _TextFormat.model_validate(text or {}).format
    except ValidationError:
        return None


def _text_change(
    previous: Mapping[str, object] | None, following: Mapping[str, object] | None
) -> TextUpdate | _Same | _Unexpressible:
    if _without(previous, "format") != _without(following, "format"):
        return _UNEXPRESSIBLE
    previous_format: Final = _text_format(previous)
    following_format: Final = _text_format(following)
    if previous_format is None or following_format is None:
        return _UNEXPRESSIBLE
    if previous_format == following_format:
        return _SAME
    if _JSON_OBJECT in (previous_format.get("type"), following_format.get("type")):
        return _UNEXPRESSIBLE
    return TextUpdate(format=following_format)


def _refs_by_identity(tools: tuple[ToolRef, ...] | None) -> Mapping[str, ToolRef] | None:
    refs: Final = tools or ()
    by_identity: Final = MappingProxyType({ref.identity: ref for ref in refs})
    return by_identity if len(by_identity) == len(refs) else None


def _added_tools(
    previous: tuple[ToolRef, ...] | None, following: tuple[ToolRef, ...] | None
) -> tuple[str, ...] | _Unexpressible:
    previous_refs: Final = _refs_by_identity(previous)
    following_refs: Final = _refs_by_identity(following)
    if previous_refs is None or following_refs is None:
        return _UNEXPRESSIBLE
    if any(following_refs.get(identity) != ref for identity, ref in previous_refs.items()):
        return _UNEXPRESSIBLE
    added: Final = tuple(ref for ref in following_refs.values() if ref.identity not in previous_refs)
    if not all(ref.inline for ref in added):
        return _UNEXPRESSIBLE
    return tuple(ref.identity for ref in added)


def _transition(previous: Settings, following: Settings) -> _Transition | None:
    match (
        _reasoning_change(previous.reasoning, following.reasoning),
        _text_change(previous.text, following.text),
        _added_tools(previous.tools, following.tools),
    ):
        case (_Unexpressible(), _, _) | (_, _Unexpressible(), _) | (_, _, _Unexpressible()):
            return None
        case (reasoning, text, tuple() as added):
            return _Transition(
                reasoning=None if isinstance(reasoning, _Same) else reasoning,
                text=None if isinstance(text, _Same) else text,
                added=added,
            )


def _transitions(effectives: Sequence[Settings]) -> tuple[_Transition, ...] | None:
    transitions: Final = tuple(map(_transition, effectives, effectives[1:]))
    expressible: Final = tuple(transition for transition in transitions if transition is not None)
    return expressible if len(expressible) == len(transitions) else None


def _transition_items(
    transition: _Transition, definitions: Mapping[str, Mapping[str, object]]
) -> tuple[Mapping[str, object], ...]:
    update: Final = configuration_update_item(reasoning=transition.reasoning, text=transition.text)
    added: Final = (
        (additional_tools_item(tuple(definitions[identity] for identity in transition.added)),)
        if transition.added
        else ()
    )
    return (*((update,) if update is not None else ()), *added)


def _transition_end(messages: Sequence[_Message], index: int) -> int:
    following: Final = range(index + 1, len(messages))
    return next((position - 1 for position in following if messages[position].role != "tool"), len(messages) - 1)


def _top_level(base: Settings, definitions: Mapping[str, Mapping[str, object]]) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "reasoning": base.reasoning,
            "text": base.text,
            "tools": None if base.tools is None else [definitions[ref.identity] for ref in base.tools],
        }
    )


def _effort(settings: Settings) -> object:
    return (settings.reasoning or {}).get("effort")


def _trusted(stored: Settings, requested: Settings) -> bool:
    effort: Final = _effort(stored)
    stored_format: Final = _text_format(stored.text)
    return (
        (effort is None or isinstance(effort, str))
        and stored_format is not None
        and stored_format in (_default_text_format(), _text_format(requested.text))
    )


def _inline(
    messages: Sequence[_Message],
    requested: Settings,
    definitions: Mapping[str, Mapping[str, object]],
    *,
    samples: bool,
) -> ConfigurationPlan | None:
    segment: Final = _segment(messages)
    if segment is None:
        return None
    if samples and _effort(segment[0].note.base) != _effort(requested):
        return None
    if not all(_trusted(stamped.note.in_effect(), requested) for stamped in segment):
        return None
    transitions: Final = _transitions((*(stamped.note.in_effect() for stamped in segment), requested))
    if transitions is None:
        return None
    placed: Final = (
        (_transition_end(messages, stamped.index), _transition_items(transition, definitions))
        for stamped, transition in zip(segment, transitions, strict=True)
    )
    base: Final = segment[0].note.base
    return ConfigurationPlan(
        top_level=_top_level(base, definitions),
        inserts=MappingProxyType({index: items for index, items in placed if items}),
        note=_continuing_note(base, requested),
    )


def plan_configuration(
    messages: Sequence[object], request: Mapping[str, object], *, rewrites_history: bool
) -> ConfigurationPlan | None:
    try:
        parsed: Final = _Requested.model_validate(request)
    except ValidationError:
        return None
    requested: Final = _settings(parsed)
    rewrite: Final = ConfigurationPlan(
        top_level=MappingProxyType({}), inserts=MappingProxyType({}), note=_starting_note(requested)
    )
    if rewrites_history:
        return rewrite
    definitions: Final = MappingProxyType({tool_identity(tool): tool for tool in parsed.tools or ()})
    return _inline(tuple(map(_message, messages)), requested, definitions, samples=parsed.samples()) or rewrite
