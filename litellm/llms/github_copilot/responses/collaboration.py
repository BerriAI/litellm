from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

_CLIENT_NAMESPACE: Final = "collaboration"
_UPSTREAM_NAMESPACE: Final = "codex_collaboration_plaintext"
_MESSAGE_ACTIONS: Final = frozenset({"spawn_agent", "send_message", "followup_task"})
_INVENTORY_ITEMS: Final = ("additional_tools", "tool_search_output")
_CALL_ITEMS: Final = ("function_call", "custom_tool_call")
_IDENTITY_KEYS: Final = ("type", "namespace", "name", "call_id")
_RECORD: Final = TypeAdapter(Mapping[str, object])
_SEQUENCE: Final = TypeAdapter(Sequence[object])


def _sequence(value: object) -> Sequence[object]:
    return _SEQUENCE.validate_python(value, strict=True) if isinstance(value, (list, tuple)) else ()


def _record(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return _RECORD.validate_python(value, strict=True)
    except ValidationError:
        return None


def _mapping(value: object) -> Mapping[str, object]:
    return _record(value) or MappingProxyType({})


def _inventory(request: Mapping[str, object]) -> Iterator[Mapping[str, object]]:
    for tool in _sequence(request.get("tools")):
        if (tool_record := _record(tool)) is not None:
            yield tool_record
    for item in _sequence(request.get("input")):
        if (item_record := _record(item)) is not None and item_record.get("type") in _INVENTORY_ITEMS:
            for tool in _sequence(item_record.get("tools")):
                if (nested_tool := _record(tool)) is not None:
                    yield nested_tool


def _encrypted_action(tool: object) -> str | None:
    record: Final = _record(tool)
    if record is None or record.get("type") != "function":
        return None
    name: Final = record.get("name")
    parameters: Final = _mapping(record.get("parameters"))
    properties: Final = _mapping(parameters.get("properties"))
    message: Final = _mapping(properties.get("message"))
    if isinstance(name, str) and name in _MESSAGE_ACTIONS:
        return name if message.get("type") == "string" and message.get("encrypted") is True else None
    return None


def _actions(request: Mapping[str, object]) -> Iterator[str]:
    for tool in _inventory(request):
        if tool.get("type") == "namespace" and tool.get("name") == _CLIENT_NAMESPACE:
            for child in _sequence(tool.get("tools")):
                if (action := _encrypted_action(child)) is not None:
                    yield action


def _identity(value: Mapping[str, object], source: str, target: str) -> Mapping[str, object]:
    namespace: Final = value.get("namespace")
    name: Final = value.get("name")
    if isinstance(name, str):
        for separator in (".", "__"):
            if name.startswith(source + separator):
                if namespace is not None and namespace != source:
                    raise ValueError("Conflicting Copilot collaboration namespace and function name")
                return {**value, "namespace": target, "name": name[len(source) + len(separator) :]}
    return {**value, "namespace": target} if namespace == source else value


def _plaintext_tool(tool: object) -> object:
    record: Final = _record(tool)
    if record is None or _encrypted_action(record) is None:
        return tool
    parameters: Final = _mapping(record.get("parameters"))
    properties: Final = _mapping(parameters.get("properties"))
    message: Final = _mapping(properties.get("message"))
    return {
        **record,
        "parameters": {
            **parameters,
            "properties": {
                **properties,
                "message": {key: value for key, value in message.items() if key != "encrypted"},
            },
        },
    }


def _tool(tool: object, source: str, target: str) -> object:
    record: Final = _record(tool)
    if record is None or record.get("type") != "namespace" or record.get("name") != source:
        return tool
    return {
        **record,
        "name": target,
        **(
            {
                "tools": [
                    _plaintext_tool(child) if source == _CLIENT_NAMESPACE else child
                    for child in _sequence(record.get("tools"))
                ]
            }
            if isinstance(record.get("tools"), (list, tuple))
            else {}
        ),
    }


def _tools(value: object, source: str, target: str) -> object:
    sequence: Final = _sequence(value)
    return [_tool(tool, source, target) for tool in sequence] if sequence else value


def _choice(value: object, source: str, target: str) -> object:
    record: Final = _record(value)
    if record is None:
        return value
    if record.get("type") == "namespace" and record.get("name") == source:
        return {**record, "name": target}
    if record.get("type") == "allowed_tools":
        return {**record, "tools": [_choice(tool, source, target) for tool in _sequence(record.get("tools"))]}
    rewritten: Final = _identity(record, source, target)
    name: Final = record.get("name")
    if rewritten is not record and "namespace" not in record and isinstance(name, str):
        separator: Final = "__" if name.startswith(source + "__") else "."
        return {**record, "name": target + separator + str(rewritten["name"])}
    return rewritten


def _project_item(value: object) -> object:
    record: Final = _record(value)
    if record is None:
        return value
    if record.get("type") in _INVENTORY_ITEMS and "tools" in record:
        return {**record, "tools": _tools(record.get("tools"), _CLIENT_NAMESPACE, _UPSTREAM_NAMESPACE)}
    return _identity(record, _CLIENT_NAMESPACE, _UPSTREAM_NAMESPACE) if record.get("type") in _CALL_ITEMS else value


def _project_input(value: object) -> object:
    sequence: Final = _sequence(value)
    return [_project_item(item) for item in sequence] if sequence else value


class CopilotPlaintextCollaboration:
    def __init__(self) -> None:
        self._actions: frozenset[str] = frozenset()
        self._identities: Mapping[int, Mapping[str, object]] = MappingProxyType({})

    @property
    def active(self) -> bool:
        return bool(self._actions)

    def prepare_request(self, request: Mapping[str, object]) -> Mapping[str, object]:
        self._actions = frozenset(_actions(request))
        self._identities = MappingProxyType({})
        if not self.active:
            return request
        if (
            request.get("store") is True
            or request.get("previous_response_id") is not None
            or request.get("conversation") is not None
        ):
            raise ValueError("Copilot plaintext collaboration requires stateless Responses requests")
        if any(tool.get("name") == _UPSTREAM_NAMESPACE for tool in _inventory(request)):
            raise ValueError(f"The namespace {_UPSTREAM_NAMESPACE!r} is reserved by the Copilot adapter")
        projected: Final = {
            key: (
                _tools(value, _CLIENT_NAMESPACE, _UPSTREAM_NAMESPACE)
                if key == "tools"
                else _project_input(value)
                if key == "input"
                else _choice(value, _CLIENT_NAMESPACE, _UPSTREAM_NAMESPACE)
                if key == "tool_choice"
                else value
            )
            for key, value in request.items()
        }
        return {**projected, "store": False}

    def _remember(self, value: Mapping[str, object], index: int | None, *, item: bool) -> Mapping[str, object]:
        canonical: Final = _identity(value, _UPSTREAM_NAMESPACE, _UPSTREAM_NAMESPACE)
        if index is None:
            return canonical
        known: Final = self._identities.get(index, MappingProxyType({}))
        keys: Final = _IDENTITY_KEYS if item else _IDENTITY_KEYS[1:]
        identity: Final = {
            key: canonical[key] for key in keys if isinstance(canonical.get(key), str) and canonical.get(key) != ""
        }
        if any(key in known and known[key] != field for key, field in identity.items()):
            raise ValueError("Conflicting Copilot collaboration identity for one output_index")
        merged: Final = {**known, **identity}
        self._identities = MappingProxyType({**self._identities, index: merged})
        return {**canonical, **{key: merged[key] for key in keys if key in merged}}

    def _restore_identity(self, value: Mapping[str, object]) -> Mapping[str, object]:
        restored: Final = _identity(value, _UPSTREAM_NAMESPACE, _CLIENT_NAMESPACE)
        name: Final = restored.get("name")
        if restored is value or not isinstance(name, str) or name not in self._actions:
            return restored
        if "encrypted_function_args" in value and value["encrypted_function_args"] != []:
            raise ValueError("Copilot returned encrypted arguments for a plaintext collaboration tool")
        return {**restored, "encrypted_function_args": []}

    def _restore_item(self, value: object, index: int | None) -> object:
        record: Final = _record(value)
        if record is None:
            return value
        if record.get("type") not in _CALL_ITEMS and index not in self._identities:
            return value
        remembered: Final = self._remember(record, index, item=True)
        return (
            self._restore_identity(remembered)
            if remembered.get("type") == "function_call"
            else _identity(remembered, _UPSTREAM_NAMESPACE, _CLIENT_NAMESPACE)
        )

    def restore_response(self, response: Mapping[str, object]) -> Mapping[str, object]:
        if not self.active:
            return response
        return {
            key: (
                self._restore_output(value)
                if key == "output"
                else _tools(value, _UPSTREAM_NAMESPACE, _CLIENT_NAMESPACE)
                if key == "tools"
                else _choice(value, _UPSTREAM_NAMESPACE, _CLIENT_NAMESPACE)
                if key == "tool_choice"
                else value
            )
            for key, value in response.items()
        }

    def _restore_output(self, value: object) -> object:
        sequence: Final = _sequence(value)
        return [self._restore_item(item, index) for index, item in enumerate(sequence)] if sequence else value

    def restore_event(self, event: Mapping[str, object]) -> Mapping[str, object]:
        if not self.active:
            return event
        output_index: Final = event.get("output_index")
        index: Final = output_index if isinstance(output_index, int) and not isinstance(output_index, bool) else None
        restored: Final = {
            key: (
                self._restore_item(value, index)
                if key == "item"
                else self.restore_response(response)
                if key == "response" and (response := _record(value)) is not None
                else value
            )
            for key, value in event.items()
        }
        if event.get("type") == "response.function_call_arguments.done":
            return self._restore_identity(self._remember(restored, index, item=False))
        if event.get("type") == "response.function_call_arguments.delta":
            return _identity(restored, _UPSTREAM_NAMESPACE, _CLIENT_NAMESPACE)
        return restored
