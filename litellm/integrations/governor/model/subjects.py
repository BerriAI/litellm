"""Governance subjects: who a policy governs."""

from dataclasses import dataclass
from typing import Literal

SubjectKind = Literal["key", "team", "user", "end_user", "tag", "org", "model"]

#: Subject kinds whose value space is large enough that caching them per pod
#: would churn the L1 LRU. The cache facade sends these straight to L2.
HIGH_CARDINALITY_KINDS: frozenset[SubjectKind] = frozenset({"end_user", "tag"})

#: Hierarchy order from most to least specific, used to order the descriptor
#: tuple so the same request always evaluates policies in a stable sequence.
SUBJECT_PRIORITY: tuple[SubjectKind, ...] = (
    "end_user",
    "key",
    "team",
    "user",
    "tag",
    "org",
    "model",
)


@dataclass(frozen=True)
class Subject:
    kind: SubjectKind
    id: str
    display: str | None = None


@dataclass(frozen=True)
class SubjectRef:
    subjects: tuple[Subject, ...]

    def of_kind(self, kind: SubjectKind) -> Subject | None:
        for subject in self.subjects:
            if subject.kind == kind:
                return subject
        return None


def is_high_cardinality(kind: SubjectKind) -> bool:
    return kind in HIGH_CARDINALITY_KINDS
