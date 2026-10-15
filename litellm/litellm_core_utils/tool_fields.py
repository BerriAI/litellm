from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import reduce
from typing import Annotated, Final, TypeAlias

from pydantic import BeforeValidator, ConfigDict, Json, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm.litellm_core_utils.core_helpers import normalize_drop_params

ROUTES_WITH_TOOLS: Final = frozenset(
    {
        "acompletion",
        "aresponses",
        "acompact_responses",
        "anthropic_messages",
        "agenerate_content",
        "agenerate_content_stream",
        "acreate_interaction",
    }
)
_TOOL_FIELDS: Final = frozenset({"tools", "tool_choice", "parallel_tool_calls", "toolConfig", "tool_config"})
_TOOLS: Final = TypeAdapter(list[object])
_FIELD_LIST: Final = TypeAdapter(list[object], config=ConfigDict(strict=True))
_JSON_OBJECT: Final = TypeAdapter(Json[dict[str, object]])


def _schema_as_read(value: object) -> object:
    if not value:
        return None
    if not isinstance(value, str):
        return value
    try:
        return _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return value


_Schema: TypeAlias = Annotated[dict[str, object] | None, BeforeValidator(_schema_as_read)]


class _ToolFunction(TypedDict, total=False):
    parameters: ReadOnly[_Schema]


class _ToolEntry(TypedDict, total=False):
    function: ReadOnly[_ToolFunction | None]
    parameters: ReadOnly[_Schema]
    input_schema: ReadOnly[_Schema]
    vector_store_ids: ReadOnly[list[str] | None]


_TOOL_ENTRY: Final = TypeAdapter(_ToolEntry)
_WEB_SEARCH_OPTIONS: Final = TypeAdapter(dict[str, object] | None, config=ConfigDict(strict=True))


@dataclass(frozen=True, slots=True)
class DropPolicy:
    drop_params: bool
    additional_drop_params: frozenset[str]

    def drops(self, field: str) -> bool:
        return self.drop_params or field in self.additional_drop_params


@dataclass(frozen=True, slots=True)
class MalformedField:
    field: str
    message: str
    dropped: Callable[[Mapping[str, object]], Mapping[str, object]]


def _field_names(value: object) -> frozenset[str]:
    try:
        return frozenset(name for name in _FIELD_LIST.validate_python(value) if isinstance(name, str))
    except ValidationError:
        return frozenset()


def drop_policy(params: Mapping[str, object]) -> DropPolicy:
    return DropPolicy(
        drop_params=litellm.drop_params is True or normalize_drop_params(params.get("drop_params")) is True,
        additional_drop_params=_field_names(params.get("additional_drop_params")),
    )


def first_validation_error(e: ValidationError, prefix: tuple[str | int, ...] = ()) -> str:
    error: Final = e.errors(include_url=False)[0]
    return f"{'.'.join(str(part) for part in (*prefix, *error['loc']))}: {error['msg']}"


def _tool_entry_error(index: int, entry: object) -> str | None:
    try:
        _TOOL_ENTRY.validate_python(entry, strict=True)
    except ValidationError as e:
        return first_validation_error(e, prefix=("tools", index))
    return None


def _without_tool_fields(data: Mapping[str, object]) -> Mapping[str, object]:
    return {name: value for name, value in data.items() if name not in _TOOL_FIELDS}


def _malformed_tools(data: Mapping[str, object]) -> MalformedField | None:
    tools: Final = data.get("tools")
    if tools is None:
        return None
    try:
        entries: Final = _TOOLS.validate_python(tools)
    except ValidationError as e:
        return MalformedField("tools", first_validation_error(e, prefix=("tools",)), _without_tool_fields)
    entry_errors: Final = tuple(_tool_entry_error(index, entry) for index, entry in enumerate(entries))
    first_error: Final = next((error for error in entry_errors if error is not None), None)
    if first_error is None:
        return None
    kept: Final = [entry for entry, error in zip(entries, entry_errors) if error is None]
    return MalformedField(
        "tools", first_error, lambda fields: {**fields, "tools": kept} if kept else _without_tool_fields(fields)
    )


def _without_web_search_options(data: Mapping[str, object]) -> Mapping[str, object]:
    return {name: value for name, value in data.items() if name != "web_search_options"}


def _malformed_web_search_options(data: Mapping[str, object]) -> MalformedField | None:
    try:
        _WEB_SEARCH_OPTIONS.validate_python(data.get("web_search_options"))
    except ValidationError as e:
        return MalformedField(
            "web_search_options",
            first_validation_error(e, prefix=("web_search_options",)),
            _without_web_search_options,
        )
    return None


def resolve_tool_fields(
    data: Mapping[str, object], candidates: Callable[[], tuple[DropPolicy, ...]]
) -> Mapping[str, object] | MalformedField:
    checked: Final = (_malformed_tools(data), _malformed_web_search_options(data))
    malformed: Final = tuple(field for field in checked if field is not None)
    if not malformed:
        return data
    policies: Final = candidates()
    rejected: Final = next(
        (field for field in malformed if not any(policy.drops(field.field) for policy in policies)), None
    )
    if rejected is not None:
        return rejected
    return reduce(
        lambda resolved, field: field.dropped(resolved),
        (field for field in malformed if all(policy.drops(field.field) for policy in policies)),
        data,
    )


def resolve_deployment_tool_fields(
    kwargs: Mapping[str, object], route: object
) -> Mapping[str, object] | MalformedField:
    if "proxy_server_request" not in kwargs or route not in ROUTES_WITH_TOOLS:
        return kwargs
    return resolve_tool_fields(kwargs, lambda: (drop_policy(kwargs),))
