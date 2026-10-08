from typing import Final, Generic, Literal, TypeVar

from pydantic import Field

from .models import Execution, FindingDraft, Record, TracePart

ResponseT: Final = TypeVar("ResponseT", bound=Record)


class EvidenceRequest(Record):
    action: Literal["catalog", "read", "search", "review_catalog", "read_reviews", "search_reviews", "history"]
    execution_id: str | None = None
    span_ids: tuple[str, ...] = ()
    query: str = ""
    char_start: int = Field(default=0, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    review_phase: Literal["initial", "revisited"] | None = None
    turn_start: int = Field(default=0, ge=0)
    turn_end: int | None = Field(default=None, ge=0)
    include_initial: bool = False


class PythonRequest(Record):
    action: Literal["python"]
    code: str = Field(min_length=1)
    execution_ids: tuple[str, ...] = ()
    span_ids: tuple[str, ...] = ()


class CatalogEntry(Record):
    execution: Execution
    spans: tuple[tuple[str, str, str, str, int | None, str, str], ...]
    partial: bool
    characters: int | None


class ReviewRecord(Record):
    execution_id: str
    phase: Literal["initial", "revisited"]
    content: str


class ReviewIndex(Record):
    execution_id: str
    phase: Literal["initial", "revisited"]
    characters: int


class EvidenceReply(Record):
    request: EvidenceRequest
    catalog: tuple[CatalogEntry, ...] = ()
    parts: tuple[TracePart, ...] = ()
    error: str = ""
    review_catalog: tuple[ReviewIndex, ...] = ()
    reviews: tuple[ReviewRecord, ...] = ()


class Checkpoint(Record):
    working_notes: str = Field(min_length=1)


class Candidate(Record):
    check_id: str
    kind: Literal["issue", "pattern"] = "issue"
    title: str
    hypothesis: str
    execution_ids: tuple[str, ...]
    existing_finding_id: str | None = None


class Clusters(Record):
    candidates: tuple[Candidate, ...] = ()


class Findings(Record):
    findings: tuple[FindingDraft, ...] = ()


class FindingGroup(Record):
    members: tuple[str, ...] = Field(min_length=1)
    representative: str


class FindingGroups(Record):
    groups: tuple[FindingGroup, ...]


class PythonAgentTurn(Record, Generic[ResponseT]):
    tools: tuple[EvidenceRequest | PythonRequest, ...] = ()
    checkpoint: str | None = Field(default=None, min_length=1)
    result: ResponseT | None = None
