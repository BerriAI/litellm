from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Final

from litellm.integrations.custom_logger_v2.enums import Activation, Field, field_mask
from litellm.integrations.custom_logger_v2.hooks import Subscription
from litellm.integrations.custom_logger_v2.manifest import CallbackManifest


@dataclass(frozen=True, slots=True, eq=False)
class Entry:
    position: int
    manifest: CallbackManifest
    subscription: Subscription

    @property
    def activation(self) -> Activation:
        return self.subscription.activation or self.manifest.selection.activation


@dataclass(frozen=True, slots=True)
class HookPlan:
    stages: tuple[tuple[Entry, ...], ...] = ()
    inline: tuple[Entry, ...] = ()
    awaited: tuple[Entry, ...] = ()
    off_path: tuple[Entry, ...] = ()
    reads: frozenset[Field] = frozenset()


EMPTY_PLAN: Final = HookPlan()


def _blocks(subscription: Subscription) -> bool:
    return bool(subscription.writes) or subscription.may_abort


def _level(placed: tuple[tuple[Subscription, int], ...], subscription: Subscription) -> int:
    if subscription.may_abort:
        return max((level for _, level in placed), default=-1) + 1
    touched: Final = field_mask(subscription.reads | subscription.writes)
    writes: Final = field_mask(subscription.writes)
    after_writers: Final = tuple(level + 1 for prior, level in placed if field_mask(prior.writes) & touched)
    not_before_readers: Final = tuple(level for prior, level in placed if field_mask(prior.reads) & writes)
    return max((*after_writers, *not_before_readers), default=0)


def _place(
    placed: tuple[tuple[Subscription, int], ...], subscription: Subscription
) -> tuple[tuple[Subscription, int], ...]:
    return (*placed, (subscription, _level(placed, subscription)))


def _levels(entries: Sequence[Entry]) -> tuple[tuple[Entry, ...], ...]:
    placed: Final = reduce(_place, (entry.subscription for entry in entries), ())
    levels: Final = tuple(level for _, level in placed)
    return tuple(_at_level(entries, levels, level) for level in range(max(levels, default=-1) + 1))


def _at_level(entries: Sequence[Entry], levels: tuple[int, ...], level: int) -> tuple[Entry, ...]:
    return tuple(entry for entry, placed_at in zip(entries, levels, strict=True) if placed_at == level)


def plan_hook(entries: Sequence[Entry]) -> HookPlan:
    if not entries:
        return EMPTY_PLAN
    blocking: Final = tuple(entry for entry in entries if _blocks(entry.subscription))
    serial: Final = tuple(entry for entry in blocking if not entry.subscription.parallel)
    parallel: Final = tuple(entry for entry in blocking if entry.subscription.parallel)
    passive: Final = tuple(entry for entry in entries if not _blocks(entry.subscription))
    return HookPlan(
        stages=(*_levels(serial), *((parallel,) if parallel else ())),
        inline=tuple(entry for entry in passive if entry.subscription.legacy),
        awaited=tuple(entry for entry in passive if entry.subscription.awaited and not entry.subscription.legacy),
        off_path=tuple(entry for entry in passive if not (entry.subscription.awaited or entry.subscription.legacy)),
        reads=frozenset[Field]().union(*(entry.subscription.reads for entry in entries)),
    )
