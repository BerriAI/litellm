import itertools
import re
from dataclasses import dataclass
from datetime import date
from typing import Final

_HEADING: Final = re.compile(
    r"^\[?(?P<version>\d+\.\d+\.\d+[0-9A-Za-z.\-+]*)\]?(?:\([^)]*\))?\s+\((?P<date>\d{4}-\d{2}-\d{2})\)\s*$"
)


@dataclass(frozen=True, slots=True)
class ChangelogEntry:
    section: str
    text: str


@dataclass(frozen=True, slots=True)
class Release:
    version: str
    released: date
    entries: tuple[ChangelogEntry, ...]


def parse_changelog(markdown: str) -> tuple[Release, ...]:
    releases: Final = (_parse_release(block) for block in re.split(r"(?m)^## ", markdown)[1:])
    return tuple(release for release in releases if release is not None)


def _parse_release(block: str) -> Release | None:
    heading, _, body = block.partition("\n")
    match: Final = _HEADING.match(heading.strip())
    if match is None:
        return None
    sections: Final = re.split(r"(?m)^### ", body)[1:]
    return Release(
        version=match["version"],
        released=date.fromisoformat(match["date"]),
        entries=tuple(itertools.chain.from_iterable(map(_parse_section, sections))),
    )


def _parse_section(block: str) -> tuple[ChangelogEntry, ...]:
    title, _, body = block.partition("\n")
    items: Final = re.split(r"(?m)^\* ", body)[1:]
    return tuple(ChangelogEntry(section=title.strip(), text=" ".join(item.split())) for item in items)
