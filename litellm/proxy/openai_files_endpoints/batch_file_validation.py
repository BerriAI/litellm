import json
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain
from typing import BinaryIO, Final, NoReturn, assert_never

from litellm.proxy._types import ProxyException

BATCH_LINE_REQUIRED_KEYS: Final = ("custom_id", "method", "url", "body")
_MB: Final = 1024 * 1024


@dataclass(frozen=True, slots=True)
class BatchFileTooLarge:
    size_bytes: int
    limit_mb: int


@dataclass(frozen=True, slots=True)
class BatchFileWrongExtension:
    filename: str


@dataclass(frozen=True, slots=True)
class BatchFileEmpty:
    pass


@dataclass(frozen=True, slots=True)
class BatchFileInvalidJsonLine:
    line_number: int


@dataclass(frozen=True, slots=True)
class BatchFileLineNotObject:
    line_number: int


@dataclass(frozen=True, slots=True)
class BatchFileMissingLineKey:
    line_number: int
    key: str


BatchFileValidationFailure = (
    BatchFileTooLarge
    | BatchFileWrongExtension
    | BatchFileEmpty
    | BatchFileInvalidJsonLine
    | BatchFileLineNotObject
    | BatchFileMissingLineKey
)


def _file_size_bytes(file_source: bytes | BinaryIO) -> int:
    if isinstance(file_source, bytes):
        return len(file_source)
    file_source.seek(0, 2)
    size: Final = file_source.tell()
    file_source.seek(0)
    return size


def _iter_lines(file_source: bytes | BinaryIO) -> Iterator[bytes]:
    if isinstance(file_source, bytes):
        return iter(file_source.splitlines())
    file_source.seek(0)
    return iter(file_source)


def _check_line(line_number: int, raw_line: bytes) -> BatchFileValidationFailure | None:
    try:
        parsed: Final = json.loads(raw_line)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return BatchFileInvalidJsonLine(line_number=line_number)
    if not isinstance(parsed, dict):
        return BatchFileLineNotObject(line_number=line_number)
    missing: Final = next((key for key in BATCH_LINE_REQUIRED_KEYS if key not in parsed), None)
    if missing is None:
        return None
    return BatchFileMissingLineKey(line_number=line_number, key=missing)


def _scan_lines(file_source: bytes | BinaryIO) -> BatchFileValidationFailure | None:
    content_lines: Final = (
        (line_number, raw_line)
        for line_number, raw_line in enumerate(_iter_lines(file_source), start=1)
        if raw_line.strip()
    )
    first_line: Final = next(content_lines, None)
    if first_line is None:
        return BatchFileEmpty()
    return next(
        (
            failure
            for line_number, raw_line in chain((first_line,), content_lines)
            for failure in (_check_line(line_number, raw_line),)
            if failure is not None
        ),
        None,
    )


def check_batch_file_upload(
    filename: str | None,
    file_source: bytes | BinaryIO,
    max_batch_file_size_mb: int | None,
) -> BatchFileValidationFailure | None:
    if filename is None or not filename.lower().endswith(".jsonl"):
        return BatchFileWrongExtension(filename=filename or "")
    if max_batch_file_size_mb is not None and max_batch_file_size_mb > 0:
        size_bytes: Final = _file_size_bytes(file_source)
        if size_bytes > max_batch_file_size_mb * _MB:
            return BatchFileTooLarge(size_bytes=size_bytes, limit_mb=max_batch_file_size_mb)
    scan_failure: Final = _scan_lines(file_source)
    if not isinstance(file_source, bytes):
        file_source.seek(0)
    return scan_failure


def raise_batch_file_validation_failure(failure: BatchFileValidationFailure) -> NoReturn:
    match failure:
        case BatchFileTooLarge(size_bytes=size_bytes, limit_mb=limit_mb):
            raise ProxyException(
                message=(
                    f"Batch input file is {size_bytes / _MB:.1f} MB, which exceeds the configured "
                    f"max_batch_file_size_mb of {limit_mb} MB. The file was not forwarded to the provider."
                ),
                type="invalid_request_error",
                param="file",
                code=413,
            )
        case BatchFileWrongExtension(filename=filename):
            raise ProxyException(
                message=(
                    f"Invalid file format for Batch API: '{filename}'. "
                    "Batch input files must be .jsonl files. The file was not forwarded to the provider."
                ),
                type="invalid_request_error",
                param="file",
                code=400,
            )
        case BatchFileEmpty():
            raise ProxyException(
                message="Batch input file has no request lines. The file was not forwarded to the provider.",
                type="invalid_request_error",
                param="file",
                code=400,
            )
        case BatchFileInvalidJsonLine(line_number=line_number):
            raise ProxyException(
                message=(
                    f"Batch input file line {line_number} is not valid JSON. "
                    "The file was not forwarded to the provider."
                ),
                type="invalid_request_error",
                param="file",
                code=400,
            )
        case BatchFileLineNotObject(line_number=line_number):
            raise ProxyException(
                message=(
                    f"Batch input file line {line_number} must be a JSON object. "
                    "The file was not forwarded to the provider."
                ),
                type="invalid_request_error",
                param="file",
                code=400,
            )
        case BatchFileMissingLineKey(line_number=line_number, key=key):
            raise ProxyException(
                message=(
                    f"Missing required parameter: '{key}' (batch input file line {line_number}). "
                    f"Each line must be a JSON object with keys {', '.join(BATCH_LINE_REQUIRED_KEYS)}. "
                    "The file was not forwarded to the provider."
                ),
                type="invalid_request_error",
                param=key,
                code=400,
            )
        case _:
            assert_never(failure)
