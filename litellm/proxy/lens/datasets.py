import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial, reduce
from itertools import chain
from types import MappingProxyType
from typing import Annotated, Final, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm.constants import LENS_DATASET_MAX_CASE_CHARS, LENS_DATASET_MAX_CASES
from litellm.proxy.lens.models import (
    BuildRequest,
    BuildResult,
    BuildSource,
    CaseSource,
    DatasetCase,
    DatasetMessage,
    DatasetToolCall,
    Finding,
    FindingSource,
    Record,
    SkippedCase,
    TextSource,
    TraceSource,
)
from litellm.proxy.lens.sources import parse_execution
from litellm.rust_bridge.trace.generated.types import SpanDetail, Trace, UIContent, UIMessage

AGENT_VERSION_ATTRIBUTE: Final = "agent.version"


class DatasetReader(Protocol):
    async def trace(self, trace_id: str, trace_ref: str) -> Trace | None: ...
    async def span(self, trace_id: str, span_id: str, trace_ref: str) -> SpanDetail | None: ...
    async def findings(self, lens_id: str, ids: tuple[str, ...]) -> tuple[Finding, ...]: ...


class _Content(Record):
    messages: tuple[DatasetMessage, ...]
    reply: str
    tool_calls: tuple[DatasetToolCall, ...]


class _TextLine(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    messages: tuple[DatasetMessage, ...]
    reply: str = ""
    tool_calls: tuple[DatasetToolCall, ...] = ()
    expected: str = ""
    source: CaseSource = CaseSource()
    agent_version: str = ""


class _AssistantSummary(Record):
    content: str | None
    tool_names: tuple[str, ...]


_SUMMARIES: Final[TypeAdapter[tuple[_AssistantSummary, ...]]] = TypeAdapter(
    Annotated[tuple[_AssistantSummary, ...], Field(min_length=1)]
)
_RAW_MESSAGES: Final[TypeAdapter[tuple[DatasetMessage, ...]]] = TypeAdapter(
    Annotated[tuple[DatasetMessage, ...], Field(min_length=1)]
)


@dataclass(frozen=True, slots=True)
class _Reply:
    text: str
    tool_calls: tuple[DatasetToolCall, ...]


@dataclass(frozen=True, slots=True)
class _Admission:
    seen: frozenset[str]
    cases: tuple[DatasetCase, ...] = ()
    skipped: tuple[SkippedCase, ...] = ()


Candidate: TypeAlias = DatasetCase | SkippedCase


def case_id(messages: tuple[DatasetMessage, ...], reply: str, tool_calls: tuple[DatasetToolCall, ...]) -> str:
    content: Final = _Content(messages=messages, reply=reply, tool_calls=tool_calls)
    return hashlib.sha256(content.model_dump_json().encode()).hexdigest()


def _tool_call_chars(tool_calls: tuple[DatasetToolCall, ...]) -> int:
    return sum(len(t.name) + len(t.arguments) for t in tool_calls)


def case_chars(case: DatasetCase) -> int:
    return (
        sum(len(m.content) + len(m.name) + _tool_call_chars(m.tool_calls) for m in case.messages)
        + len(case.reply)
        + _tool_call_chars(case.tool_calls)
        + len(case.expected)
    )


def make_case(
    messages: tuple[DatasetMessage, ...],
    reply: str,
    tool_calls: tuple[DatasetToolCall, ...],
    source: CaseSource,
    expected: str = "",
    agent_version: str = "",
) -> Candidate:
    if not messages and not reply.strip() and not tool_calls:
        return SkippedCase(source=source, reason="no_content")
    case: Final = DatasetCase(
        id=case_id(messages, reply, tool_calls),
        messages=messages,
        reply=reply,
        tool_calls=tool_calls,
        expected=expected,
        source=source,
        agent_version=agent_version,
    )
    if case_chars(case) > LENS_DATASET_MAX_CASE_CHARS:
        return SkippedCase(source=source, reason="too_large")
    return case


def _message(message: UIMessage) -> DatasetMessage:
    return DatasetMessage(
        role=message["role"],
        content=message["content"],
        name=message.get("name") or "",
        tool_calls=_tool_calls((message,)),
    )


def _tool_calls(messages: Iterable[UIMessage]) -> tuple[DatasetToolCall, ...]:
    calls: Final = chain.from_iterable(m.get("tool_calls", ()) for m in messages)
    return tuple(DatasetToolCall(name=c["name"], arguments=c["arguments"]) for c in calls)


def _summary_messages(summaries: tuple[_AssistantSummary, ...]) -> tuple[DatasetMessage, ...]:
    return tuple(
        DatasetMessage(
            role="assistant",
            content=s.content or "",
            tool_calls=tuple(DatasetToolCall(name=name, arguments="") for name in s.tool_names),
        )
        for s in summaries
    )


def _text_messages(text: str) -> tuple[DatasetMessage, ...] | None:
    try:
        return _summary_messages(_SUMMARIES.validate_json(text))
    except ValidationError:
        pass
    try:
        return tuple(_RAW_MESSAGES.validate_json(text))
    except ValidationError:
        pass
    try:
        return (DatasetMessage.model_validate_json(text),)
    except ValidationError:
        return None


def _ui_text(ui: UIContent, raw: str) -> str:
    return ui["text"] if ui["kind"] == "text" else raw


def _conversation(ui: UIContent, raw: str) -> tuple[DatasetMessage, ...]:
    if ui["kind"] == "messages":
        return tuple(_message(m) for m in ui["messages"])
    if decoded := _text_messages(_ui_text(ui, raw)):
        return decoded
    return (DatasetMessage(role="user", content=raw),) if raw.strip() else ()


def _assistant_reply(messages: tuple[DatasetMessage, ...]) -> _Reply:
    replies: Final = tuple(m for m in messages if m.role == "assistant")
    return _Reply(
        "\n\n".join(m.content for m in replies if m.content),
        tuple(chain.from_iterable(m.tool_calls for m in replies)),
    )


def _reply(ui: UIContent, raw: str) -> _Reply:
    if ui["kind"] == "messages":
        return _assistant_reply(tuple(_message(m) for m in ui["messages"]))
    text: Final = _ui_text(ui, raw)
    if decoded := _text_messages(text):
        return _assistant_reply(decoded)
    return _Reply(text, ())


def case_from_span(detail: SpanDetail, source: CaseSource) -> Candidate:
    reply: Final = _reply(detail["output_ui"], detail["output"])
    return make_case(
        _conversation(detail["input_ui"], detail["input"]),
        reply.text,
        reply.tool_calls,
        source.model_copy(update=MappingProxyType({"span_id": detail["span_id"]})),
        agent_version=detail["attributes"].get(AGENT_VERSION_ATTRIBUTE, ""),
    )


async def _last_conversation(reader: DatasetReader, source: TraceSource, trace: Trace) -> SpanDetail | None:
    llm_spans: Final = sorted(
        (s for s in trace["spans"] if s["type"] == "llm"), key=lambda s: s["start_offset_ms"], reverse=True
    )
    for span in llm_spans:
        detail = await reader.span(source.trace_id, span["span_id"], source.trace_ref)
        if detail is not None and detail["input_ui"]["kind"] == "messages":
            return detail
    return None


async def _trace_cases(reader: DatasetReader, source: TraceSource) -> tuple[Candidate, ...]:
    origin: Final = CaseSource(trace_id=source.trace_id, trace_ref=source.trace_ref, span_id=source.span_id)
    if source.span_id:
        span: Final = await reader.span(source.trace_id, source.span_id, source.trace_ref)
        return (case_from_span(span, origin) if span else SkippedCase(source=origin, reason="no_content"),)
    trace: Final = await reader.trace(source.trace_id, source.trace_ref)
    detail: Final = await _last_conversation(reader, source, trace) if trace else None
    return (case_from_span(detail, origin) if detail else SkippedCase(source=origin, reason="no_content"),)


async def _evidence_case(reader: DatasetReader, origin: CaseSource, execution: str) -> Candidate:
    try:
        _, _, trace_id, trace_ref = parse_execution(execution)
    except (ValueError, ValidationError):
        return SkippedCase(source=origin, reason="no_content")
    located: Final = origin.model_copy(update=MappingProxyType({"trace_id": trace_id, "trace_ref": trace_ref}))
    span: Final = await reader.span(trace_id, origin.span_id, trace_ref)
    return case_from_span(span, located) if span else SkippedCase(source=located, reason="no_content")


def _first_by_key(keys: tuple[str, ...]) -> frozenset[int]:
    first: Final = MappingProxyType({key: index for index, key in reversed(tuple(enumerate(keys)))})
    return frozenset(first.values())


def _evidence_spans(lens_id: str, findings: tuple[Finding, ...]) -> tuple[tuple[str, CaseSource], ...]:
    pairs: Final = tuple(chain.from_iterable(((f.id, e) for e in f.evidence) for f in findings))
    kept: Final = _first_by_key(tuple(f"{e.execution_id}\0{e.span_id}" for _, e in pairs))
    return tuple(
        (e.execution_id, CaseSource(span_id=e.span_id, finding_id=fid, lens_id=lens_id))
        for index, (fid, e) in enumerate(pairs)
        if index in kept
    )


async def _finding_cases(reader: DatasetReader, source: FindingSource) -> tuple[Candidate, ...]:
    findings: Final = await reader.findings(source.lens_id, source.finding_ids)
    spans: Final = _evidence_spans(source.lens_id, findings)
    return tuple([await _evidence_case(reader, origin, execution) for execution, origin in spans])


def _looks_like_json_line(line: str) -> bool:
    return line.lstrip().startswith("{")


def _text_line_case(line: str) -> Candidate:
    try:
        parsed: Final = _TextLine.model_validate_json(line)
    except ValidationError:
        return SkippedCase(source=CaseSource(), reason="invalid")
    return make_case(
        parsed.messages,
        parsed.reply,
        parsed.tool_calls,
        parsed.source,
        expected=parsed.expected,
        agent_version=parsed.agent_version,
    )


def _text_cases(source: TextSource) -> tuple[Candidate, ...]:
    lines: Final = tuple(line for line in source.text.splitlines() if line.strip())
    if lines and all(_looks_like_json_line(line) for line in lines):
        return tuple(_text_line_case(line) for line in lines)
    return (make_case((DatasetMessage(role="user", content=source.text),), "", (), CaseSource()),)


async def _source_cases(reader: DatasetReader, source: BuildSource) -> tuple[Candidate, ...]:
    match source:
        case TraceSource():
            return await _trace_cases(reader, source)
        case FindingSource():
            return await _finding_cases(reader, source)
        case TextSource():
            return _text_cases(source)
    return assert_never(source)


def _skip(state: _Admission, skipped: SkippedCase) -> _Admission:
    return _Admission(state.seen, state.cases, (*state.skipped, skipped))


def _admit(existing_count: int, state: _Admission, candidate: Candidate) -> _Admission:
    if isinstance(candidate, SkippedCase):
        return _skip(state, candidate)
    if candidate.id in state.seen:
        return _skip(state, SkippedCase(source=candidate.source, reason="duplicate"))
    if existing_count + len(state.cases) >= LENS_DATASET_MAX_CASES:
        return _skip(state, SkippedCase(source=candidate.source, reason="over_limit"))
    return _Admission(state.seen | {candidate.id}, (*state.cases, candidate), state.skipped)


def _unread_source(source: BuildSource) -> CaseSource:
    match source:
        case TraceSource():
            return CaseSource(trace_id=source.trace_id, trace_ref=source.trace_ref, span_id=source.span_id)
        case FindingSource():
            return CaseSource(lens_id=source.lens_id, finding_id=source.finding_ids[0])
        case TextSource():
            return CaseSource()
    return assert_never(source)


async def _admit_source(
    reader: DatasetReader, existing_count: int, state: _Admission, source: BuildSource
) -> _Admission:
    if existing_count + len(state.cases) >= LENS_DATASET_MAX_CASES:
        return _skip(state, SkippedCase(source=_unread_source(source), reason="over_limit"))
    return reduce(partial(_admit, existing_count), await _source_cases(reader, source), state)


async def build_cases(request: BuildRequest, reader: DatasetReader, existing: tuple[DatasetCase, ...]) -> BuildResult:
    state = _Admission(seen=frozenset(c.id for c in existing))  # rebind-ok: sources are read in order until full
    for source in request.sources:
        state = await _admit_source(reader, len(existing), state, source)
    return BuildResult(cases=state.cases, skipped=state.skipped)


def rehashed(case: DatasetCase) -> DatasetCase:
    return case.model_copy(update=MappingProxyType({"id": case_id(case.messages, case.reply, case.tool_calls)}))


def revision_cases(cases: tuple[DatasetCase, ...]) -> tuple[DatasetCase, ...]:
    hashed: Final = tuple(rehashed(c) for c in cases)
    kept: Final = _first_by_key(tuple(c.id for c in hashed))
    return tuple(c for index, c in enumerate(hashed) if index in kept)


def revision_problem(cases: tuple[DatasetCase, ...]) -> str | None:
    if len(cases) > LENS_DATASET_MAX_CASES:
        return f"A dataset holds at most {LENS_DATASET_MAX_CASES} cases"
    if any(case_chars(c) > LENS_DATASET_MAX_CASE_CHARS for c in cases):
        return f"Each case must be at most {LENS_DATASET_MAX_CASE_CHARS} characters"
    return None


def included_cases(cases: tuple[DatasetCase, ...]) -> tuple[DatasetCase, ...]:
    return tuple(c for c in cases if c.included)


def export_jsonl(cases: tuple[DatasetCase, ...]) -> str:
    return "".join(c.model_dump_json() + "\n" for c in included_cases(cases))
