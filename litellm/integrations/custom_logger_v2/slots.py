from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeAlias

from litellm.integrations.custom_logger_v2.enums import DERIVED_ONLY, Field
from litellm.integrations.custom_logger_v2.outcomes import InvalidPatchError, MergeField, Patch, PatchOp
from litellm.integrations.custom_logger_v2.patches import LiveFields, apply_patch
from litellm.types.integrations.custom_logger_v2 import Outcome


@dataclass(frozen=True, slots=True)
class HookArgs:
    auth: object = None
    request: object = None
    result: object = None
    start_time: object = None
    end_time: object = None
    outcome: Outcome | None = None
    logging_obj: object = None
    call_type: str | None = None


Derive: TypeAlias = Callable[[HookArgs], object]
Write: TypeAlias = Callable[[HookArgs, object], None]


@dataclass(frozen=True, slots=True)
class Slot:
    read: Derive
    write: Write | None = None

    @property
    def derived(self) -> bool:
        return self.write is None


def derived(read: Derive) -> Slot:
    return Slot(read=read)


def mapped(read: Derive, write: Write) -> Slot:
    return Slot(read=read, write=write)


class _Cache:
    __slots__ = ("values",)

    def __init__(self) -> None:
        self.values: Final[dict[Field, object]] = {}  # mutable-ok: one hook invocation's derived values


def slot_table_problem(slots: Mapping[Field, Slot]) -> str | None:
    writable_derived: Final = sorted(
        field.value for field, slot in slots.items() if field in DERIVED_ONLY and not slot.derived
    )
    if writable_derived:
        return f"{', '.join(writable_derived)} are derived-only, so their slots cannot write"
    return None


class SlotFields(LiveFields):
    __slots__ = ("_args", "_cache", "_inner", "_slots")

    def __init__(
        self,
        inner: Mapping[Field, object],
        slots: Mapping[Field, Slot],
        args: HookArgs,
        cache: _Cache | None = None,
    ) -> None:
        problem: Final = slot_table_problem(slots)
        if problem is not None:
            raise ValueError(problem)
        self._inner: Final = inner
        self._slots: Final = slots
        self._args: Final = args
        self._cache: Final = cache or _Cache()

    @property
    def inner(self) -> Mapping[Field, object]:
        return self._inner

    def _value(self, field: Field, slot: Slot) -> object:
        if not slot.derived:
            return slot.read(self._args)
        cached: Final = self._cache.values
        if field not in cached:
            cached[field] = slot.read(self._args)
        return cached[field]

    def __getitem__(self, field: Field) -> object:
        slot: Final = self._slots.get(field)
        if slot is None:
            return self._inner[field]
        value: Final = self._value(field, slot)
        if value is None:
            raise KeyError(field)
        return value

    def __iter__(self) -> Iterator[Field]:
        known: Final = self._cache.values
        return iter(
            (
                *(field for field in self._inner if field not in self._slots),
                *(field for field in self._slots if known.get(field) is not None),
            )
        )

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def apply(self, patch: Patch) -> Mapping[Field, object]:
        for op in patch.ops:
            if op.field in self._slots:
                self._write(op)
        rest: Final = Patch(tuple(op for op in patch.ops if op.field not in self._slots))
        if not rest.ops:
            return self
        inner: Final = apply_patch(self._inner, rest)
        return self if inner is self._inner else SlotFields(inner, self._slots, self._args, self._cache)

    def _write(self, op: PatchOp) -> None:
        write: Final = self._slots[op.field].write
        if write is None:
            raise InvalidPatchError(f"{op.field.value} is derived from the hook's arguments, so it cannot be written")
        if isinstance(op, MergeField):
            raise InvalidPatchError(f"{op.field.value} is replaced whole, never merged")
        write(self._args, op.value)


NO_SLOTS: Final[Mapping[Field, Slot]] = MappingProxyType({})


def with_slots(inner: Mapping[Field, object], slots: Mapping[Field, Slot], args: HookArgs) -> Mapping[Field, object]:
    return inner if not slots else SlotFields(inner, slots, args)


def settle_reads(fields: Mapping[Field, object], reads: Iterable[Field]) -> None:
    if not isinstance(fields, SlotFields):
        return
    for field in reads:
        _ = field in fields


__all__ = [
    "NO_SLOTS",
    "Derive",
    "HookArgs",
    "Slot",
    "SlotFields",
    "Write",
    "derived",
    "mapped",
    "settle_reads",
    "slot_table_problem",
    "with_slots",
]
