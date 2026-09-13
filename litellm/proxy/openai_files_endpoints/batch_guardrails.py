"""
Run the configured pre-call guardrails over every record of a batch input file.

Runs after ``batch_file_validation.check_batch_file_upload``, so every line here is already known
to parse as a JSON object carrying ``custom_id``, ``method``, ``url`` and ``body``.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, BinaryIO, Final, NoReturn, TypeAlias
from urllib.parse import urlsplit

from fastapi import HTTPException
from typing_extensions import assert_never

from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import is_guardrail_intervention
from litellm.litellm_core_utils.api_route_to_call_types import get_call_types_for_route
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.llms.openai import BatchGuardrailRecord, BatchGuardrailReport
from litellm.types.utils import CallTypes, CallTypesLiteral

if TYPE_CHECKING:
    from litellm.proxy.utils import ProxyLogging

EMPTY_MAPPING: Final[Mapping[str, object]] = MappingProxyType({})

_SCAN_WINDOW: Final = 32

# Past this the rewrite rolls to disk, keeping the router's per-deployment deepcopy of the handle
# as cheap as it is for the spooled upload this replaces.
_REWRITE_SPOOL_BYTES: Final = 1024 * 1024

# custom_id is caller-supplied and reaches a log line, so it is stripped of control characters
# and capped rather than rendered as given.
_CONTROL_CHARACTERS: Final = re.compile(r"[\x00-\x1f\x7f]")
_CUSTOM_ID_LOG_LIMIT: Final = 128
_SUMMARY_LIMIT: Final = 50

_SCAN_METADATA_KEY: Final = "litellm_metadata"
_SCAN_METADATA_BAGS: Final = (_SCAN_METADATA_KEY, "metadata")

# Set by pre_call_hook when a guardrail rerouted the request to a different model.
_ROUTE_APPLIED_KEY: Final = "sensitive_data_routing_applied"

# Dropped before dispatch and restored afterwards rather than diffed. Guardrail dispatch writes
# its bookkeeping into `metadata`, and a record's own metadata is not scanned content on the
# online path either. `guardrails` is dropped because guardrail selection reads it ahead of the
# proxy-injected list, so leaving it would let a record's own body opt out of the chain its key
# and team selected; online that key can only add to the list, never replace it.
_INJECTED_KEYS: Final = frozenset({_SCAN_METADATA_KEY, "metadata", "guardrails"})

# Only what guardrail dispatch reads. The parent OTel span is deliberately left out: parenting one
# guardrail span per record would put tens of thousands of spans on a single upload's trace.
_SCAN_METADATA_KEYS: Final = frozenset(
    {
        "guardrails",
        "_guardrail_pipelines",
        "_pipeline_managed_guardrails",
        "user_api_key_metadata",
        "user_api_key_team_metadata",
        "tags",
        "headers",
    }
)

_SCANNABLE_CALL_TYPES: Final = frozenset(
    {
        CallTypes.acompletion,
        CallTypes.atext_completion,
        CallTypes.aembedding,
        CallTypes.aresponses,
        CallTypes.anthropic_messages,
    }
)

# Mirrors the record classifier in litellm/llms/bedrock/files/transformation.py, so a record
# litellm already accepts without a url keeps working.
_BODY_SHAPE_CALL_TYPES: Final = (
    ("messages", CallTypes.acompletion),
    ("prompt", CallTypes.atext_completion),
    ("input", CallTypes.aembedding),
)


@dataclass(frozen=True, slots=True)
class UnparseableRecord:
    line_number: int


@dataclass(frozen=True, slots=True)
class UnscannableRecord:
    line_number: int
    custom_id: str | None
    url: str | None


@dataclass(frozen=True, slots=True)
class UnroutableRecord:
    line_number: int
    custom_id: str | None
    guardrail: str | None


BatchScanFailure: TypeAlias = UnparseableRecord | UnscannableRecord | UnroutableRecord


@dataclass(frozen=True, slots=True)
class _Redaction:
    """A rewritten record on its way to the scan spool, held only for the window it was scanned in."""

    line_number: int
    custom_id: str | None
    text: str


@dataclass(frozen=True, slots=True)
class RecordRedacted:
    line_number: int
    custom_id: str | None
    offset: int
    length: int
    """Where the re-serialized record sits in the scan spool, so a large file's rewrites stay off the heap."""


@dataclass(frozen=True, slots=True)
class RecordDropped:
    line_number: int
    custom_id: str | None
    guardrail: str | None = None


_RecordChange: TypeAlias = RecordRedacted | RecordDropped
_ScanOutcome: TypeAlias = BatchScanFailure | _Redaction | RecordDropped


@dataclass(frozen=True, slots=True)
class BatchScanResult:
    """What the scan decided, per record. Empty changes means the upload proceeds untouched."""

    changes: tuple[_RecordChange, ...]
    scanned_records: int
    redactions: BinaryIO
    """Spool holding every rewritten record, keyed by the offsets on each ``RecordRedacted``."""

    @property
    def submitted_records(self) -> int:
        return self.scanned_records - sum(1 for change in self.changes if isinstance(change, RecordDropped))

    def summary(self) -> str:
        """Compact per-record outcome for the server-side log line, capped so one upload cannot flood it."""
        shown: Final = ", ".join(
            f"line {change.line_number}{_describe(change.custom_id)} "
            f"{'redacted' if isinstance(change, RecordRedacted) else 'dropped'}"
            for change in self.changes[:_SUMMARY_LIMIT]
        )
        remaining: Final = len(self.changes) - _SUMMARY_LIMIT
        return shown if remaining <= 0 else f"{shown}, and {remaining} more"

    def report(self) -> BatchGuardrailReport:
        return BatchGuardrailReport(
            submitted_records=self.submitted_records,
            modified_records=tuple(
                BatchGuardrailRecord(
                    line=change.line_number,
                    custom_id=change.custom_id,
                    action="redacted" if isinstance(change, RecordRedacted) else "dropped",
                    guardrail=change.guardrail if isinstance(change, RecordDropped) else None,
                )
                for change in self.changes
            ),
        )


@dataclass(frozen=True, slots=True)
class _ParsedRecord:
    line_number: int
    payload: Mapping[str, object]


def _rejected(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail={"error": message})  # mutable-ok: FastAPI detail shape


def raise_public(failure: BatchScanFailure) -> NoReturn:
    """Map a scan failure onto the 400 contract the files endpoint already returns."""
    match failure:
        case UnparseableRecord(line_number=line_number):
            raise _rejected(
                f"The 'body' of batch input line {line_number} is not an object, so guardrails cannot be applied to it"
            )
        case UnscannableRecord(line_number=line_number, custom_id=custom_id, url=url):
            raise _rejected(
                f"Batch input line {line_number}{_describe(custom_id)} targets {url or 'no url'} "
                "and its body has no messages, prompt or input, so guardrails cannot read it. "
                "Give the record a chat, completion, embedding, responses or messages body"
            )
        case UnroutableRecord(line_number=line_number, custom_id=custom_id, guardrail=guardrail):
            raise _rejected(
                f"Batch input line {line_number}{_describe(custom_id)} was routed to a different model by "
                f"{guardrail or 'a guardrail'}, and every record of a batch file goes to one provider, so "
                "the file cannot be submitted. Send that record outside the batch"
            )
        case _:
            assert_never(failure)


def raise_nothing_to_submit() -> NoReturn:
    """Every record was blocked, so there is no batch left to create."""
    raise _rejected(
        "Every record in the batch input file was blocked by a guardrail, so there is nothing left to submit"
    )


def _is_content_block(exc: BaseException) -> bool:
    """
    Whether the guardrail judged the record, as opposed to failing to judge it.

    Stricter than ``is_guardrail_intervention``, which answers a different question and counts
    every ``GuardrailRaisedException`` as a block. Several integrations raise that same exception
    for an unreachable backend or an unparseable response, and only when the operator configured
    the guardrail to fail closed, so treating it as a block would turn "refuse this request" into
    "drop this record and submit the rest", which is the silent loss of enforcement this whole
    path exists to prevent. A guardrail that does not say it blocked content aborts the upload.

    Guardrails that report a technical failure as an ``HTTPException`` carrying a block status
    are caught by ``__cause__``: raising ``from`` the underlying error is a deliberate statement
    that something else caused this, which a verdict on content never is. Implicit context is
    left alone, since a block raised inside an unrelated ``except`` would read as a failure.
    """
    if isinstance(exc, GuardrailRaisedException):
        return exc.blocked_content
    if exc.__cause__ is not None:
        return False
    return is_guardrail_intervention(exc)


def _naming_guardrail(exc: BaseException) -> str | None:
    """The guardrail that raised, from whichever place it recorded its own name."""
    named: Final = getattr(exc, "guardrail_name", None)
    if isinstance(named, str):
        return named
    detail: Final = getattr(exc, "detail", None)
    enriched: Final = detail.get("guardrail_name") if isinstance(detail, dict) else None
    return enriched if isinstance(enriched, str) else None


def _describe(custom_id: str | None) -> str:
    if not custom_id:
        return ""
    safe: Final = _CONTROL_CHARACTERS.sub(" ", custom_id)[:_CUSTOM_ID_LOG_LIMIT]
    return f" (custom_id {safe})"


def _iter_lines(source: BinaryIO) -> Iterator[tuple[int, bytes]]:
    """
    Yield every non-blank line with its 1-based number, so both passes number records alike.

    Bytes, not text. The upload validation immediately before this parses each line as bytes,
    where the json module sniffs the encoding itself and accepts a leading byte order mark or a
    lone surrogate. Decoding to `str` first is stricter than that, so a file written by any of
    the editors that emit a BOM would pass validation and then fail the scan.
    """
    for line_number, raw_line in enumerate(source, start=1):
        if raw_line.strip():
            yield line_number, raw_line


def _iter_records(source: BinaryIO) -> Iterator[_ParsedRecord]:
    """Yield one record per line, relying on the upload validation that already ran."""
    for line_number, raw_line in _iter_lines(source):
        yield _ParsedRecord(line_number=line_number, payload=json.loads(raw_line))


def _call_type_from_url(url: str) -> CallTypesLiteral | None:
    """
    Resolve the route a record names, tolerating how callers actually write it.

    An absolute url has to reduce to its path or nothing matches, and a record naming
    ``/v1/responses`` in full would fall through to its body, where ``input`` reads as an
    embedding and the record gets scanned as the wrong call type rather than the right one.
    """
    try:
        path: Final = urlsplit(url).path.split("?")[0].rstrip("/")
    except ValueError:
        # urlsplit rejects a few malformed authorities outright, and the validation that ran
        # before this only checks the key is present. An unreadable url is one we do not
        # recognize, which is what falling back to the body shape already handles.
        return None
    call_types: Final = get_call_types_for_route(path)
    if call_types is None:
        return None
    scannable: Final = next((c for c in call_types if c in _SCANNABLE_CALL_TYPES), None)
    return None if scannable is None else scannable.value


def _call_type_from_body(body: Mapping[str, object]) -> CallTypesLiteral | None:
    shape: Final = next((call_type for field, call_type in _BODY_SHAPE_CALL_TYPES if field in body), None)
    return None if shape is None else shape.value


def _scannable_call_type(url: object, body: Mapping[str, object]) -> CallTypesLiteral | None:
    """
    Resolve how to scan a record: its url when we recognize one, otherwise its body shape.

    An unrecognized url falls through to the body rather than rejecting, because a record we can
    still read is a record we can still scan, and the provider transformers treat an unknown url
    as chat rather than as an error.
    """
    from_url: Final = _call_type_from_url(url) if isinstance(url, str) and url else None
    return from_url if from_url is not None else _call_type_from_body(body)


def _custom_id_of(payload: Mapping[str, object]) -> str | None:
    """
    The record's identifier, rendered as text.

    The batch spec asks for a string, but callers do send numbers, and reporting those as null
    would leave the one field a caller reconciles on empty for exactly the records it needs.
    """
    custom_id: Final = payload.get("custom_id")
    if isinstance(custom_id, str):
        # A lone surrogate parses out of the file but cannot be encoded back out, and this value
        # is echoed in the response, so rendering it would fail the whole upload with a 500.
        return custom_id.encode("utf-8", "replace").decode("utf-8")
    return str(custom_id) if isinstance(custom_id, (int, float)) and not isinstance(custom_id, bool) else None


def _fingerprint(body: Mapping[str, object], keys: frozenset[str]) -> str:
    """
    Order-insensitive projection, so a guardrail re-serializing a dict does not read as a change.

    An absent key projects to ``null`` while a key holding ``None`` projects to the string
    ``"null"``, so adding or dropping a null-valued key still reads as a change.
    """
    return json.dumps(
        tuple(
            (key, json.dumps(body[key], sort_keys=True, default=str) if key in body else None) for key in sorted(keys)
        )
    )


def build_scan_metadata(request_metadata: Mapping[str, object]) -> Mapping[str, object]:
    """
    Narrow the request metadata to the keys guardrail dispatch reads.

    Passing the whole thing through would carry values that cannot be copied, such as the parent
    OTel span, and would hand every record proxy state it has no business seeing.
    """
    return MappingProxyType(
        {key: value for key, value in request_metadata.items() if key in _SCAN_METADATA_KEYS}
    )  # mutable-ok: MappingProxyType freezes the comprehension


async def _scan_record(
    record: _ParsedRecord,
    scan_metadata: Mapping[str, object],
    user_api_key_dict: UserAPIKeyAuth,
    proxy_logging_obj: ProxyLogging,
) -> _ScanOutcome | None:
    body: Final = record.payload.get("body")
    if not isinstance(body, dict):
        return UnparseableRecord(line_number=record.line_number)

    custom_id: Final = _custom_id_of(record.payload)
    url: Final = record.payload.get("url")
    call_type: Final = _scannable_call_type(url, body)
    if call_type is None:
        return UnscannableRecord(
            line_number=record.line_number,
            custom_id=custom_id,
            url=url if isinstance(url, str) else None,
        )

    scan_input: Final[dict[str, object]] = copy.deepcopy(body)  # mutable-ok: pre_call_hook mutates the dict it is given
    own_injected: Final = MappingProxyType({key: body[key] for key in _INJECTED_KEYS if key in body})
    for injected in _INJECTED_KEYS:
        scan_input.pop(injected, None)
    # Both bags, because guardrails read whichever one their own route populates and a record
    # scanned as chat reaches ones that only ever look at `metadata`; both are injected keys, so
    # neither survives into the record that ships. Deep, and per bag per record, because `headers`
    # and `tags` are nested containers otherwise shared with the upload request and with every
    # other record in the window. The narrowing above already removed what cannot be copied.
    for injected in _SCAN_METADATA_BAGS:
        scan_input[injected] = copy.deepcopy(dict(scan_metadata))  # mutable-ok: guardrails write here

    try:
        # The chain hands back the body it produced, which may be a replacement for the dict it was
        # given rather than that same dict mutated, so this is what gets compared.
        scanned: Final[dict] = await proxy_logging_obj.pre_call_hook(  # mutable-ok: the guardrails' own dict
            user_api_key_dict=user_api_key_dict,
            data=scan_input,
            call_type=call_type,
            guardrails_only=True,
        )
    except Exception as exc:
        if _is_content_block(exc):
            return RecordDropped(line_number=record.line_number, custom_id=custom_id, guardrail=_naming_guardrail(exc))
        raise

    rerouted: Final = scanned.get("metadata")
    if isinstance(rerouted, dict) and rerouted.get(_ROUTE_APPLIED_KEY):
        return UnroutableRecord(
            line_number=record.line_number,
            custom_id=custom_id,
            guardrail=rerouted.get("sensitive_data_routing_guardrail"),
        )

    compared: Final = (frozenset(body) | frozenset(scanned)) - _INJECTED_KEYS
    if _fingerprint(scanned, compared) == _fingerprint(body, compared):
        return None
    for injected in _INJECTED_KEYS:
        scanned.pop(injected, None)
    scanned.update(own_injected)
    return _Redaction(
        line_number=record.line_number,
        custom_id=custom_id,
        text=json.dumps({**record.payload, "body": scanned}),  # mutable-ok: json.dumps needs a plain dict
    )


async def _scan_window(
    window: tuple[_ParsedRecord, ...],
    scan_metadata: Mapping[str, object],
    user_api_key_dict: UserAPIKeyAuth,
    proxy_logging_obj: ProxyLogging,
) -> tuple[tuple[int, _ScanOutcome | BaseException], ...]:
    """``return_exceptions=True`` so one record raising never leaves its siblings unobserved."""
    outcomes: Final = await asyncio.gather(
        *(_scan_record(record, scan_metadata, user_api_key_dict, proxy_logging_obj) for record in window),
        return_exceptions=True,
    )
    return tuple((record.line_number, outcome) for record, outcome in zip(window, outcomes) if outcome is not None)


def _spool(redactions: BinaryIO, redaction: _Redaction) -> RecordRedacted:
    """Park the rewritten record on disk so only its location is carried for the rest of the scan."""
    encoded: Final = redaction.text.encode("utf-8")
    redactions.seek(0, 2)
    offset: Final = redactions.tell()
    redactions.write(encoded)
    return RecordRedacted(
        line_number=redaction.line_number,
        custom_id=redaction.custom_id,
        offset=offset,
        length=len(encoded),
    )


def _worst(problems: tuple[tuple[int, BatchScanFailure | BaseException], ...]) -> BatchScanFailure | BaseException:
    """A guardrail that blocked outranks a record we merely refused; then earliest line wins."""
    raised: Final = tuple(problem for problem in problems if isinstance(problem[1], BaseException))
    return min(raised or problems, key=lambda problem: problem[0])[1]


async def scan_batch_input_file(
    *,
    file_source: BinaryIO,
    request_metadata: Mapping[str, object],
    user_api_key_dict: UserAPIKeyAuth,
    proxy_logging_obj: ProxyLogging,
) -> BatchScanFailure | BatchScanResult:
    """
    Stream a batch input file and run the pre-call guardrail chain against every record.

    A record a guardrail rewrites is kept in its rewritten form and a record it blocks is dropped,
    which is what the online path does per request. Both are returned for reporting. A guardrail
    exception that is not a block is re-raised untouched so its status code survives, since dropping
    a record that was never inspected is worse than refusing the file.
    """
    scan_metadata: Final = build_scan_metadata(request_metadata)
    problems: Final[list[tuple[int, BatchScanFailure | BaseException]]] = []  # mutable-ok: spans windows
    changes: Final[list[_RecordChange]] = []  # mutable-ok: accumulates across windows
    window: Final[list[_ParsedRecord]] = []  # mutable-ok: bounded read-ahead buffer
    scanned: Final[list[int]] = []  # mutable-ok: counts records the scan actually reached
    redactions: Final = tempfile.SpooledTemporaryFile(  # noqa: SIM115  # the rewrite reads this back
        max_size=_REWRITE_SPOOL_BYTES
    )

    async def drain() -> None:
        if window:
            scanned.append(len(window))
            for line_number, outcome in await _scan_window(
                tuple(window), scan_metadata, user_api_key_dict, proxy_logging_obj
            ):
                if isinstance(outcome, _Redaction):
                    changes.append(_spool(redactions, outcome))
                elif isinstance(outcome, RecordDropped):
                    changes.append(outcome)
                else:
                    problems.append((line_number, outcome))
            window.clear()

    try:
        for item in _iter_records(file_source):
            window.append(item)
            if len(window) >= _SCAN_WINDOW:
                await drain()
                if problems:
                    break
        if not problems:
            await drain()
    except BaseException:
        redactions.close()
        raise
    finally:
        file_source.seek(0)

    if problems:
        redactions.close()
        worst: Final = _worst(tuple(problems))
        if isinstance(worst, BaseException):
            raise worst
        return worst
    if not changes:
        redactions.close()
    return BatchScanResult(
        changes=tuple(sorted(changes, key=lambda change: change.line_number)),
        scanned_records=sum(scanned),
        redactions=redactions,
    )


def _read_spooled(redactions: BinaryIO, change: RecordRedacted) -> bytes:
    redactions.seek(change.offset)
    return redactions.read(change.length)


def rewrite_batch_input_file(file_source: BinaryIO, result: BatchScanResult) -> BinaryIO:
    """
    Re-emit the file with redacted records rewritten and dropped records left out.

    Untouched records are copied through as written rather than re-serialized, so enabling the
    feature does not reformat records no guardrail objected to. Blank lines between records are
    not carried over, since they are not records. Rewritten records are read back from the scan's
    spool rather than from memory, so a file whose records are mostly rewritten does not put a
    second copy of itself on the heap.
    """
    redacted: Final = MappingProxyType(
        {change.line_number: change for change in result.changes if isinstance(change, RecordRedacted)}
    )  # mutable-ok: MappingProxyType freezes the lookup table
    dropped: Final = frozenset(change.line_number for change in result.changes if isinstance(change, RecordDropped))

    output: Final = tempfile.SpooledTemporaryFile(  # noqa: SIM115  # the caller uploads this handle
        max_size=_REWRITE_SPOOL_BYTES
    )
    wrote_any = False  # rebind-ok: tracks whether a separator is needed
    try:
        for line_number, raw_line in _iter_lines(file_source):
            if line_number in dropped:
                continue
            change = redacted.get(line_number)
            line = raw_line.rstrip(b"\n") if change is None else _read_spooled(result.redactions, change)
            output.write(b"\n" + line if wrote_any else line)
            wrote_any = True
    except BaseException:
        output.close()
        raise
    finally:
        file_source.seek(0)
    output.seek(0)
    return output
