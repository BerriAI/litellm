import itertools
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict

from sdk_watch.changelog import Release
from sdk_watch.sources import Candidate, Sdk
from sdk_watch.triage import Finding, NeededChange


class CheckedRelease(BaseModel):
    model_config = ConfigDict(frozen=True)

    sdk: str
    version: str
    released: date


class CarriedFinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    sdk: str
    version: str
    released: date
    change: NeededChange


class RunHistory(BaseModel):
    model_config = ConfigDict(frozen=True)

    checked: tuple[CheckedRelease, ...] = ()
    carried: tuple[CarriedFinding, ...] = ()

    @property
    def checked_keys(self) -> frozenset[tuple[str, str]]:
        return frozenset((release.sdk, release.version) for release in self.checked)


def load_history(path: Path) -> RunHistory:
    return RunHistory.model_validate_json(path.read_bytes()) if path.exists() else RunHistory()


def save_history(path: Path, history: RunHistory) -> None:
    path.write_text(history.model_dump_json())


def carried_findings(history: RunHistory, sdks: tuple[Sdk, ...]) -> tuple[Finding, ...]:
    by_name: Final = MappingProxyType({sdk.name: sdk for sdk in sdks})
    return tuple(_finding(item, by_name[item.sdk]) for item in history.carried if item.sdk in by_name)


def next_history(
    previous: RunHistory, since: date, candidates: tuple[Candidate, ...], deferred: tuple[Finding, ...]
) -> RunHistory:
    still_in_window: Final = (release for release in previous.checked if release.released >= since)
    checked_now: Final = map(_checked, candidates)
    return RunHistory(
        checked=tuple(dict.fromkeys(itertools.chain(still_in_window, checked_now))),
        carried=tuple(map(_carried, deferred)),
    )


def _checked(candidate: Candidate) -> CheckedRelease:
    return CheckedRelease(
        sdk=candidate.sdk.name, version=candidate.release.version, released=candidate.release.released
    )


def _carried(finding: Finding) -> CarriedFinding:
    release: Final = finding.candidate.release
    return CarriedFinding(
        sdk=finding.candidate.sdk.name, version=release.version, released=release.released, change=finding.change
    )


def _finding(item: CarriedFinding, sdk: Sdk) -> Finding:
    release: Final = Release(version=item.version, released=item.released, entries=())
    return Finding(candidate=Candidate(sdk=sdk, release=release), change=item.change)
