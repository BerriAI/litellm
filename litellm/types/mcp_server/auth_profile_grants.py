from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

ProfileGrants: TypeAlias = Mapping[str, str]


@dataclass(frozen=True, slots=True)
class UnreadableGrants:
    pass
