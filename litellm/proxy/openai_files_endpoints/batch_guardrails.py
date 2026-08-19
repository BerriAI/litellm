"""Run the configured pre-call guardrails over every record of a batch input file."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, BinaryIO, Final, NoReturn

from fastapi import HTTPException
from typing_extensions import assert_never

from litellm.litellm_core_utils.api_route_to_call_types import get_call_types_for_route
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypes, CallTypesLiteral

if TYPE_CHECKING:
    from litellm.proxy.utils import ProxyLogging

EMPTY_MAPPING: Final[Mapping[str, object]] = MappingProxyType({})

_SCAN_WINDOW: Final = 32

_SCAN_METADATA_KEY: Final = "litellm_metadata"

# `metadata` is dropped rather than diffed: guardrail dispatch writes its bookkeeping into it
# whenever the payload has one, and a record's own metadata is not scanned content on the
# online path either.
_INJECTED_KEYS: Final = frozenset({_SCAN_METADATA_KEY, "metadata"})

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
class RedactionRequired:
    line_number: int
    custom_id: str | None


BatchScanFailure = UnparseableRecord | UnscannableRecord | RedactionRequired


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
                f"Batch input line {line_number} is not a JSON object with a 'body' field, "
                "so guardrails cannot be applied to it"
            )
        case UnscannableRecord(line_number=line_number, custom_id=custom_id, url=url):
            raise _rejected(
                f"Batch input line {line_number}{_describe(custom_id)} targets {url or 'no url'} "
                "and its body has no messages, prompt or input, so guardrails cannot read it. "
                "Give the record a chat, completion, embedding, responses or messages body"
            )
        case RedactionRequired(line_number=line_number, custom_id=custom_id):
            raise _rejected(
                f"A guardrail changed batch input line {line_number}{_describe(custom_id)}. "
                "Per-record redaction is not enabled, so the file was rejected rather than modified"
            )
        case _:
            assert_never(failure)


def _describe(custom_id: str | None) -> str:
    return f" (custom_id {custom_id})" if custom_id else ""


def _iter_records(source: BinaryIO) -> Iterator[_ParsedRecord | UnparseableRecord]:
    """Yield one record per line. JSONL is one object per line, so a line that does not parse is bad."""
    for line_number, raw_line in enumerate(source, start=1):
        try:
            text = raw_line.decode("utf-8") if isinstance(raw_line, (bytes, bytearray)) else raw_line
        except UnicodeDecodeError:
            yield UnparseableRecord(line_number=line_number)
            return
        if not text.strip():
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            yield UnparseableRecord(line_number=line_number)
            return
        yield (
            _ParsedRecord(line_number=line_number, payload=payload)
            if isinstance(payload, dict)
            else UnparseableRecord(line_number=line_number)
        )


def _call_type_from_url(url: str) -> CallTypesLiteral | None:
    # Callers write the url by hand, so a query string or a trailing slash is not a different route.
    path: Final = url.split("?")[0].rstrip("/")
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
    custom_id: Final = payload.get("custom_id")
    return custom_id if isinstance(custom_id, str) else None


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
) -> BatchScanFailure | None:
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

    scan_input: Final[dict] = copy.deepcopy(body)  # mutable-ok: pre_call_hook mutates the dict it is given
    scan_input.pop("metadata", None)
    scan_input[_SCAN_METADATA_KEY] = dict(scan_metadata)  # mutable-ok: guardrails write bookkeeping here

    returned: Final = await proxy_logging_obj.pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        data=scan_input,
        call_type=call_type,
        guardrails_only=True,
    )
    # A guardrail may return a replacement dict rather than mutating the one it was given; that
    # replacement is what the request would have become, so it is what gets compared.
    scanned: Final = returned if isinstance(returned, dict) else scan_input

    compared: Final = (frozenset(body) | frozenset(scanned)) - _INJECTED_KEYS
    if _fingerprint(scanned, compared) != _fingerprint(body, compared):
        return RedactionRequired(line_number=record.line_number, custom_id=custom_id)
    return None


async def _scan_window(
    window: tuple[_ParsedRecord, ...],
    scan_metadata: Mapping[str, object],
    user_api_key_dict: UserAPIKeyAuth,
    proxy_logging_obj: ProxyLogging,
) -> tuple[tuple[int, BatchScanFailure | BaseException], ...]:
    """``return_exceptions=True`` so one record raising never leaves its siblings unobserved."""
    outcomes: Final = await asyncio.gather(
        *(_scan_record(record, scan_metadata, user_api_key_dict, proxy_logging_obj) for record in window),
        return_exceptions=True,
    )
    return tuple((record.line_number, outcome) for record, outcome in zip(window, outcomes) if outcome is not None)


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
) -> BatchScanFailure | None:
    """
    Stream a batch input file and run the pre-call guardrail chain against every record.

    Returns the record to reject, or None when every record passed. A guardrail that blocks raises
    its own exception, which is re-raised untouched so its status code survives.
    """
    scan_metadata: Final = build_scan_metadata(request_metadata)
    problems: Final[list[tuple[int, BatchScanFailure | BaseException]]] = []  # mutable-ok: spans windows
    window: Final[list[_ParsedRecord]] = []  # mutable-ok: bounded read-ahead buffer

    async def drain() -> None:
        if window:
            problems.extend(await _scan_window(tuple(window), scan_metadata, user_api_key_dict, proxy_logging_obj))
            window.clear()

    try:
        for item in _iter_records(file_source):
            if isinstance(item, UnparseableRecord):
                await drain()
                problems.append((item.line_number, item))
                break
            window.append(item)
            if len(window) >= _SCAN_WINDOW:
                await drain()
                if problems:
                    break
        if not problems:
            await drain()
    finally:
        file_source.seek(0)

    if not problems:
        return None
    worst: Final = _worst(tuple(problems))
    if isinstance(worst, BaseException):
        raise worst
    return worst
