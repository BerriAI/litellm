"""
Upload validation applied to every purpose at POST /v1/files.

batch_file_validation.py checks the JSONL shape of purpose="batch" uploads; this
module applies the same fast-fail-before-forwarding shape (size cap, blocked
extensions, path-traversal filenames) regardless of purpose.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final, NoReturn, assert_never

from litellm.proxy._types import ProxyException
from litellm.proxy.common_utils.path_utils import safe_filename

MB: Final = 1024 * 1024


def coerce_optional_int_setting(raw: object) -> int | None:
    """A general_settings value declared as an optional integer, e.g. max_file_size_mb.

    bool is an int subclass, so an explicit isinstance(raw, bool) exclusion is needed
    or a YAML `true`/`false` would silently pass as 1/0.
    """
    if raw is None:
        return None
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    raise TypeError(f"expected an integer, got {raw!r}")


def coerce_optional_str_list_setting(raw: object) -> tuple[str, ...]:
    """A general_settings value declared as an optional list of strings, e.g. blocked_file_extensions."""
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise TypeError(f"expected a list of strings, got {raw!r}")
    return tuple(raw)


@dataclass(frozen=True, slots=True)
class UploadedFileTooLarge:
    size_bytes: int
    limit_mb: int


@dataclass(frozen=True, slots=True)
class UploadedFileBlockedExtension:
    extension: str


@dataclass(frozen=True, slots=True)
class UploadedFileUnsafeFilename:
    filename: str


UploadValidationFailure = UploadedFileTooLarge | UploadedFileBlockedExtension | UploadedFileUnsafeFilename


def _file_size_bytes(file_source: bytes | BinaryIO) -> int:
    if isinstance(file_source, bytes):
        return len(file_source)
    original_position: Final = file_source.tell()
    file_source.seek(0, 2)
    size: Final = file_source.tell()
    file_source.seek(original_position)
    return size


def check_upload_file_size(
    file_source: bytes | BinaryIO,
    max_file_size_mb: int | None,
) -> UploadedFileTooLarge | None:
    if max_file_size_mb is None or max_file_size_mb <= 0:
        return None
    size_bytes: Final = _file_size_bytes(file_source)
    if size_bytes > max_file_size_mb * MB:
        return UploadedFileTooLarge(size_bytes=size_bytes, limit_mb=max_file_size_mb)
    return None


def check_blocked_extension(
    filename: str | None,
    blocked_extensions: tuple[str, ...],
) -> UploadedFileBlockedExtension | None:
    if not blocked_extensions or not filename:
        return None
    try:
        extension: Final = Path(safe_filename(filename)).suffix.lower()
    except ValueError:
        return None
    # The uploaded name's extension is normalized above; blocked_extensions comes
    # straight from config.yaml or the DB and is normalized here too, so a
    # differently-cased entry (".EXE") still catches a lowercase upload.
    normalized_blocked: Final = frozenset(item.lower() for item in blocked_extensions)
    if extension and extension in normalized_blocked:
        return UploadedFileBlockedExtension(extension=extension)
    return None


def check_unsafe_filename(filename: str | None) -> UploadedFileUnsafeFilename | None:
    """Reject a filename before it can influence any storage path or backend call.

    Only flags a genuine traversal component ("..") or a null byte, so an ordinary
    name like "report.v2.pdf" or ".env" is never rejected.
    """
    if not filename:
        return None
    if "\x00" in filename:
        return UploadedFileUnsafeFilename(filename=filename)
    normalized: Final = filename.replace("\\", "/")
    if any(part == ".." for part in normalized.split("/")):
        return UploadedFileUnsafeFilename(filename=filename)
    return None


def raise_upload_validation_failure(failure: UploadValidationFailure) -> NoReturn:
    match failure:
        case UploadedFileTooLarge(size_bytes=size_bytes, limit_mb=limit_mb):
            raise ProxyException(
                message=(
                    f"Uploaded file exceeds the configured max_file_size_mb of {limit_mb} MB "
                    f"(read stopped at {size_bytes / MB:.1f} MB). The file was not forwarded to the provider."
                ),
                type="invalid_request_error",
                param="file",
                code=413,
            )
        case UploadedFileBlockedExtension(extension=extension):
            raise ProxyException(
                message=(
                    f"File extension '{extension}' is blocked by this proxy's blocked_file_extensions "
                    "setting. The file was not forwarded to the provider."
                ),
                type="invalid_request_error",
                param="file",
                code=400,
            )
        case UploadedFileUnsafeFilename(filename=filename):
            raise ProxyException(
                message=(
                    f"Filename '{filename}' is not allowed: directory traversal sequences are not "
                    "permitted in uploaded file names. The file was not forwarded to the provider."
                ),
                type="invalid_request_error",
                param="file",
                code=400,
            )
        case _:
            assert_never(failure)
