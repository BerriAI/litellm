import copy
import functools
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import (
    Final,
    TypeAlias,
)

from litellm.proxy.guardrails._content_utils import distinct_keys

MutableRequest: TypeAlias = dict[str, object]

JsonBody: TypeAlias = dict[str, object]

MAX_JSON_DEPTH: Final = 64

Slot: TypeAlias = tuple[str, Callable[[str], None]]

StreamStep: TypeAlias = Callable[[str, str, bool], Awaitable[tuple[str, str]]]

Rehydrate: TypeAlias = Callable[[Sequence[str]], Awaitable[Sequence[str]]]

SlotSink: TypeAlias = list[Slot]

MutableSeq: TypeAlias = list[object]


def as_object(value: object) -> MutableRequest | None:
    """`value` as a JSON object, or None.

    `isinstance(value, dict)` alone leaves the keys and values unknown to the type
    checker. A JSON object's keys are strings, so the type is stated once, here.
    """
    return value if isinstance(value, dict) else None


def as_array(value: object) -> MutableSeq | None:
    """`value` as a JSON array, or None. See `as_object`."""
    return value if isinstance(value, list) else None


def detached(value: object) -> object:
    """A deep copy of a reply or chunk, for restoring without touching LiteLLM's own object.

    LiteLLM keeps the object it handed the hooks to fill its response cache and its
    logs, so writing restored plaintext into that object would put it there too.
    """
    return copy.deepcopy(value)


def is_container(value: object) -> bool:
    """Whether `value` is a JSON object or array, without narrowing it to unknown types."""
    return isinstance(value, (dict, list))


def collect(container: MutableRequest, key: str, slots: SlotSink) -> None:
    """Records the string at `key`, along with the write that replaces it."""
    value: Final = container.get(key)
    if isinstance(value, str) and value:
        slots.append((value, functools.partial(container.__setitem__, key)))


def collect_entry(entries: MutableSeq, index: int, slots: SlotSink) -> None:
    """Records a string held directly in a list, rather than under a key."""
    value: Final = entries[index]
    if isinstance(value, str) and value:
        slots.append((value, functools.partial(entries.__setitem__, index)))


class RequestTooDeep(Exception):
    """A request nests text past a walk's bound.

    Skipping the rest would forward it unredacted while the guardrail reports as
    enabled, so the pre-call hook refuses the request instead.
    """


def collect_text_parts(container: MutableRequest, key: str, slots: SlotSink) -> None:
    """Collects the `text` of every part in the list held at `key`."""
    for entry in as_array(container.get(key)) or ():
        part = as_object(entry)
        if part is not None:
            collect(part, "text", slots)


def choice_index(choice: object) -> int:
    """Streaming choices are matched across chunks by their index."""
    index: Final = getattr(choice, "index", 0)
    return index if isinstance(index, int) else 0


def read_field(holder: object, name: str) -> object:
    """Reads one field from a dict or from an object.

    LiteLLM's replies arrive as Pydantic models on some paths and as plain dicts on
    others, depending how far they have been deserialised, so every response walk here
    has to handle both shapes.
    """
    fields: Final = as_object(holder)
    if fields is not None:
        return fields.get(name)
    return getattr(holder, name, None)


def read_list(holder: object, name: str) -> Sequence[object]:
    """Reads a list field from a dict or an object; anything else reads as empty.

    The entries are the reply's own objects, so writing through them edits the reply.
    """
    value: Final = read_field(holder, name)
    if isinstance(value, tuple):
        return value
    return tuple(as_array(value) or ())


def write_field(holder: object, name: str, value: object) -> None:
    """Writes one string field back into a dict or an object. Pairs with read_field."""
    if isinstance(holder, dict):
        holder[name] = value
    else:
        setattr(holder, name, value)


def collect_json_leaves(node: object, slots: SlotSink, *, strict: bool = False) -> None:
    """Collects every string leaf of a JSON-ish structure, with a write-back per leaf.

    An Anthropic `tool_use` block carries `input`, an arbitrary JSON object rather than a
    string, so a value worth restoring can sit at any depth. Bounded by `MAX_JSON_DEPTH`:
    the shape is caller or model controlled, and the bound is what stops a crafted one from
    becoming an unbounded descent. Walked with an explicit stack rather than recursively,
    so a deeply nested value cannot spend stack frames proportional to attacker-chosen
    depth.

    `strict` is for the request side, where a leaf left behind would reach the provider
    unredacted: past the bound it raises `RequestTooDeep`. On the reply side a leaf past
    the bound just keeps its placeholder, which leaks nothing, so it is skipped.
    """
    pending: Final[list[tuple[object, int]]] = [(node, 0)]  # mutable-ok: local walk stack.
    while pending:
        current, current_depth = pending.pop()
        if current_depth > MAX_JSON_DEPTH:
            if strict and is_container(current) and current:
                raise RequestTooDeep("json")
            continue
        current_object = as_object(current)
        if current_object is not None:
            for key in tuple(current_object):
                value = current_object[key]
                if isinstance(value, str) and value:
                    slots.append((value, functools.partial(current_object.__setitem__, key)))
                else:
                    pending.append((value, current_depth + 1))
            continue
        entries = as_array(current)
        if entries is not None:
            for index, value in enumerate(entries):
                if isinstance(value, str) and value:
                    slots.append((value, functools.partial(entries.__setitem__, index)))
                else:
                    pending.append((value, current_depth + 1))


def collect_json_input(node: object, slots: SlotSink) -> None:
    """Collects a decisions `input_json` payload: its strings, then its numbers, then its keys.

    The payload is the caller's own data rather than a tool's arguments, so a number or a key
    can be the personal detail itself. A number is written back only when the shield changed
    it, so the rest keep their JSON type. Key renames come last because every value write
    addresses its object by the original key.
    """
    collect_json_leaves(node, slots, strict=True)
    containers: Final = tuple(_json_containers(node))
    for container in containers:
        slots.extend(_number_slots(container))
    for container in containers:
        target = as_object(container)
        if target is not None:
            slots.extend(_KeyRenames(target, tuple(target), list(target)).slots())


def _json_containers(node: object) -> Iterator[MutableRequest | MutableSeq]:
    pending: Final[list[tuple[object, int]]] = [(node, 0)]  # mutable-ok: local walk stack.
    while pending:
        current, depth = pending.pop()
        target = as_object(current)
        entries = as_array(current)
        if depth > MAX_JSON_DEPTH:
            continue
        if target is not None:
            yield target
            pending.extend((value, depth + 1) for value in target.values())
        elif entries is not None:
            yield entries
            pending.extend((value, depth + 1) for value in entries)


def _number_text(value: object) -> str | None:
    return str(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _number_slots(container: MutableRequest | MutableSeq) -> Iterator[Slot]:
    if isinstance(container, dict):
        for key, value in tuple(container.items()):
            text = _number_text(value)
            if text is not None:
                yield text, functools.partial(_write_if_changed, text, functools.partial(container.__setitem__, key))
        return
    for index, value in enumerate(tuple(container)):
        text = _number_text(value)
        if text is not None:
            yield text, functools.partial(_write_if_changed, text, functools.partial(container.__setitem__, index))


def _write_if_changed(original: str, write: Callable[[str], None], replacement: str) -> None:
    if replacement != original:
        write(replacement)


@dataclass(frozen=True, slots=True)
class _KeyRenames:
    """Renames one object's keys in place once the shield has answered for all of them.

    Waiting for the last answer rebuilds the object once, in its original order, and
    `distinct_keys` keeps two keys that redact to the same text from overwriting each other.
    """

    target: MutableRequest
    originals: tuple[str, ...]
    names: list[str]  # mutable-ok: the shield's answers, filled in slot order

    def slots(self) -> tuple[Slot, ...]:
        indexes: Final = tuple(index for index, name in enumerate(self.originals) if name)
        return tuple(
            (self.originals[index], functools.partial(self._write, index, index == indexes[-1])) for index in indexes
        )

    def _write(self, index: int, last: bool, replacement: str) -> None:
        self.names[index] = replacement
        if not last:
            return
        values: Final = tuple(self.target.values())
        self.target.clear()
        self.target.update(zip(distinct_keys(tuple(zip(self.originals, self.names))), values))


def collect_response_item(item: object, slots: SlotSink) -> None:
    """Restorable spans in one Responses API output item, dict or object.

    Mirrors `collect_responses_fields` on the request side -- a function_call or
    mcp_call item holds `arguments`, their outputs `output`, a reasoning item `summary`
    parts -- so the two directions stay symmetric. A custom tool call carries `input` and
    a code interpreter call `code`, both model-written.
    """
    for block in read_list(item, "content"):
        for field in ("text", "refusal"):
            text = read_field(block, field)
            if isinstance(text, str) and text:
                slots.append((text, lambda new, b=block, f=field: write_field(b, f, new)))
    for part in read_list(item, "summary"):
        text = read_field(part, "text")
        if isinstance(text, str) and text:
            slots.append((text, lambda new, p=part: write_field(p, "text", new)))
    for field in ("arguments", "output", "input", "code"):
        value = read_field(item, field)
        if isinstance(value, str) and value:
            slots.append((value, lambda new, i=item, f=field: write_field(i, f, new)))


async def rehydrate_slots(slots: Sequence[Slot], rehydrate: Rehydrate) -> None:
    """Restores every span in `slots` in one batch and writes each result back."""
    if not slots:
        return
    restored: Final = await rehydrate(tuple(text for text, _ in slots))
    for (_, write), replacement in zip(slots, restored):
        write(replacement)
