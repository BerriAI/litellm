"""Which telemetry groups an operator turned on, and the sink that strips every field outside them"""

import dataclasses
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeAlias

from litellm.telemetry.records import (
    ALLOWED_HEADER_KEYS,
    OTHER_HEADER_KEY,
    AttemptRecord,
    InstanceInfo,
    RequestRecord,
    TelemetryGroup,
    TokenCounts,
    UIEvent,
)
from litellm.telemetry.sink import TelemetrySink

REQUIRES: Final[Mapping[TelemetryGroup, TelemetryGroup | None]] = MappingProxyType(
    {
        TelemetryGroup.HEARTBEAT: None,
        TelemetryGroup.REQUEST_SUCCESS: TelemetryGroup.HEARTBEAT,
        TelemetryGroup.TOKEN_INFO: TelemetryGroup.REQUEST_SUCCESS,
        TelemetryGroup.REQUEST_TAXONOMY: TelemetryGroup.REQUEST_SUCCESS,
        TelemetryGroup.EVENT_DETAILS: TelemetryGroup.REQUEST_TAXONOMY,
        TelemetryGroup.INSTANCE_CONFIGURATION: TelemetryGroup.HEARTBEAT,
        TelemetryGroup.PAGE_NAVIGATION: TelemetryGroup.HEARTBEAT,
    }
)


def _lineage(group: TelemetryGroup) -> Iterator[TelemetryGroup]:
    link: TelemetryGroup | None = group  # rebind-ok: iterative walk up the parent chain avoids recursion
    while link is not None:
        yield link
        link = REQUIRES[link]


@dataclass(frozen=True, slots=True)
class TelemetryConsent:
    groups: frozenset[TelemetryGroup] = frozenset()

    def allows(self, group: TelemetryGroup) -> bool:
        return all(link in self.groups for link in _lineage(group))

    @property
    def effective_groups(self) -> frozenset[TelemetryGroup]:
        return frozenset(group for group in self.groups if self.allows(group))


OFF: Final = TelemetryConsent()


@dataclass(frozen=True, slots=True)
class MissingRequirement:
    group: TelemetryGroup
    requires: TelemetryGroup

    def message(self) -> str:
        return f"telemetry group {self.group.value!r} needs {self.requires.value!r} turned on too"


@dataclass(frozen=True, slots=True)
class UnknownGroup:
    name: str

    def message(self) -> str:
        return f"unknown telemetry group {self.name!r}, expected one of {', '.join(g.value for g in TelemetryGroup)}"


ConsentError: TypeAlias = MissingRequirement | UnknownGroup


def consent_of(groups: Iterable[TelemetryGroup]) -> TelemetryConsent | MissingRequirement:
    chosen: Final = frozenset(groups)
    missing: Final = tuple(
        MissingRequirement(group, required)
        for group in sorted(chosen, key=tuple(TelemetryGroup).index)
        if (required := REQUIRES[group]) is not None and required not in chosen
    )
    return missing[0] if missing else TelemetryConsent(chosen)


def parse_consent(names: Iterable[str]) -> TelemetryConsent | ConsentError:
    """Comma-split names from ``LITELLM_TELEMETRY_GROUPS`` or the settings API, blanks ignored"""
    cleaned: Final = tuple(name.strip().lower() for name in names if name.strip())
    by_value: Final = MappingProxyType({group.value: group for group in TelemetryGroup})
    unknown: Final = tuple(name for name in cleaned if name not in by_value)
    if unknown:
        return UnknownGroup(unknown[0])
    return consent_of(by_value[name] for name in cleaned)


_NO_KEYS: Final[frozenset[str]] = frozenset()
_OTHER_KEYS: Final[frozenset[str]] = frozenset({OTHER_HEADER_KEY})


def _allowlisted_header_keys(header_keys: frozenset[str]) -> frozenset[str]:
    lowered: Final = frozenset(key.lower() for key in header_keys)
    unknown: Final = lowered - ALLOWED_HEADER_KEYS
    return (lowered & ALLOWED_HEADER_KEYS) | (_OTHER_KEYS if unknown else _NO_KEYS)


def _gate_request(record: RequestRecord, consent: TelemetryConsent) -> RequestRecord:
    tokens: Final = consent.allows(TelemetryGroup.TOKEN_INFO)
    taxonomy: Final = consent.allows(TelemetryGroup.REQUEST_TAXONOMY)
    details: Final = consent.allows(TelemetryGroup.EVENT_DETAILS)
    return dataclasses.replace(
        record,
        tokens=record.tokens if tokens else TokenCounts(),
        provider_cache_hit=record.provider_cache_hit and tokens,
        provider=record.provider if taxonomy else None,
        deployment_hash=record.deployment_hash if taxonomy else None,
        blocks=record.blocks if details else None,
        header_keys=_allowlisted_header_keys(record.header_keys) if details else frozenset(),
    )


class ConsentGatedSink:
    """Drops every field the operator's groups do not allow before it reaches the wrapped sink"""

    def __init__(self, inner: TelemetrySink, consent: TelemetryConsent) -> None:
        self._inner: Final = inner
        self.consent: Final = consent

    def set_instance(self, info: InstanceInfo) -> None:
        if not self.consent.allows(TelemetryGroup.HEARTBEAT):
            return
        configuration: Final = self.consent.allows(TelemetryGroup.INSTANCE_CONFIGURATION)
        self._inner.set_instance(
            dataclasses.replace(
                info,
                groups=self.consent.effective_groups,
                config_keys=info.config_keys if configuration else frozenset(),
            )
        )

    def record_request(self, record: RequestRecord) -> None:
        if not self.consent.allows(TelemetryGroup.REQUEST_SUCCESS):
            return
        self._inner.record_request(_gate_request(record, self.consent))

    def record_attempt(self, record: AttemptRecord) -> None:
        if not self.consent.allows(TelemetryGroup.REQUEST_TAXONOMY):
            return
        self._inner.record_attempt(record)

    def record_ui_event(self, event: UIEvent) -> None:
        if not self.consent.allows(TelemetryGroup.PAGE_NAVIGATION):
            return
        self._inner.record_ui_event(event)

    async def flush(self) -> None:
        if not self.consent.allows(TelemetryGroup.HEARTBEAT):
            return
        await self._inner.flush()
