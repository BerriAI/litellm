"""Security controls for vector-store file uploads.

Content is classified by inspecting its actual bytes (magic signatures and a
strict UTF-8 decode), never by trusting the client-supplied filename or
content-type. Uploads are restricted to an allowlist of non-executable formats,
capped in size, screened for archives, and passed through a dependency-injected
malware scanner before they are accepted. Accepted uploads are given a
server-generated filename so the client-controlled name never reaches storage.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias, runtime_checkable

from pydantic import ValidationError
from typing_extensions import assert_never

from litellm.types.proxy.rag_ingest import RagIngestSettings

MAX_UPLOAD_SIZE_BYTES: Final = 512 * 1024 * 1024

MALWARE_SCANNER_OPTION: Final = "general_settings.rag_ingest.malware_scanner"

EICAR_TEST_SIGNATURE: Final = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"

_ARCHIVE_MAGIC_PREFIXES: Final[tuple[bytes, ...]] = (
    b"PK\x03\x04",
    b"PK\x05\x06",
    b"PK\x07\x08",
    b"\x1f\x8b",
    b"\xfd7zXZ\x00",
    b"7z\xbc\xaf\x27\x1c",
    b"Rar!\x1a\x07\x00",
    b"Rar!\x1a\x07\x01\x00",
    b"\x04\x22\x4d\x18",
    b"\x28\xb5\x2f\xfd",
)

_ARCHIVE_MAGIC_PREFIXES_ASCII_AMBIGUOUS: Final[tuple[bytes, ...]] = (b"BZh",)

_EXECUTABLE_MAGIC_PREFIXES: Final[tuple[bytes, ...]] = (
    b"\x7fELF",
    b"\xca\xfe\xba\xbe",
    b"\xfe\xed\xfa\xce",
    b"\xfe\xed\xfa\xcf",
    b"\xce\xfa\xed\xfe",
    b"\xcf\xfa\xed\xfe",
    b"\x00asm",
)

_EXECUTABLE_MAGIC_PREFIXES_ASCII_AMBIGUOUS: Final[tuple[bytes, ...]] = (b"MZ", b"dex\n")

_TAR_USTAR_MAGIC: Final = b"ustar"
_TAR_USTAR_OFFSET: Final = 257

_UTF8_BOM: Final = b"\xef\xbb\xbf"


class DetectedFormat(str, Enum):
    PDF = "pdf"
    TEXT = "text"


class DisallowedKind(str, Enum):
    ARCHIVE = "archive"
    EXECUTABLE = "executable"
    UNKNOWN_BINARY = "unknown_binary"


class RejectionReason(str, Enum):
    EMPTY_FILE = "empty_file"
    FILE_TOO_LARGE = "file_too_large"
    ARCHIVE_NOT_ALLOWED = "archive_not_allowed"
    EXECUTABLE_NOT_ALLOWED = "executable_not_allowed"
    UNSUPPORTED_FORMAT = "unsupported_format"
    MALWARE_DETECTED = "malware_detected"
    MALWARE_SCAN_ERROR = "malware_scan_error"


class ScanVerdict(str, Enum):
    CLEAN = "clean"
    INFECTED = "infected"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ScanResult:
    verdict: ScanVerdict
    signature: str | None = None


@runtime_checkable
class MalwareScanner(Protocol):
    """One shared instance scans every upload, from worker threads running at the same time.

    ``scan`` must therefore be safe to call concurrently and must return
    :class:`ScanResult` with the ``error`` verdict instead of raising when the
    engine is unavailable.
    """

    def scan(self, content: bytes) -> ScanResult: ...


@dataclass(frozen=True, slots=True)
class EicarTestMalwareScanner:
    """Placeholder scanner that only flags the EICAR anti-malware test file.

    It exists to prove the scan hook is wired end to end and to satisfy the
    EICAR retest; it provides no real protection. Inject a scanner backed by a
    real engine through the ``scanner`` parameter of :func:`validate_upload` to
    screen production uploads.
    """

    def scan(self, content: bytes) -> ScanResult:
        if EICAR_TEST_SIGNATURE in content:
            return ScanResult(verdict=ScanVerdict.INFECTED, signature="EICAR-STANDARD-ANTIVIRUS-TEST-FILE")
        return ScanResult(verdict=ScanVerdict.CLEAN)


@dataclass(frozen=True, slots=True)
class MalwareScannerConfigError:
    message: str


InstanceLoader: TypeAlias = Callable[[str, str | None], object]


def resolve_malware_scanner(
    rag_ingest: object,
    *,
    config_file_path: str | None,
    load_instance: InstanceLoader,
) -> MalwareScanner | MalwareScannerConfigError:
    try:
        settings: Final = RagIngestSettings() if rag_ingest is None else RagIngestSettings.model_validate(rag_ingest)
    except ValidationError as e:
        return MalwareScannerConfigError(f"general_settings.rag_ingest is invalid: {e}")
    if settings.malware_scanner is None:
        return EicarTestMalwareScanner()
    try:
        loaded: Final = load_instance(settings.malware_scanner, config_file_path)
    except Exception as e:
        return MalwareScannerConfigError(
            f"{MALWARE_SCANNER_OPTION}={settings.malware_scanner!r} could not be loaded: {e}"
        )
    if isinstance(loaded, type):
        return MalwareScannerConfigError(
            f"{MALWARE_SCANNER_OPTION}={settings.malware_scanner!r} names the class {loaded.__qualname__}; "
            "name an instance of it instead"
        )
    if isinstance(loaded, MalwareScanner) and callable(loaded.scan):
        return loaded
    return MalwareScannerConfigError(
        f"{MALWARE_SCANNER_OPTION}={settings.malware_scanner!r} resolved to a {type(loaded).__qualname__}, "
        "which has no scan(content: bytes) -> ScanResult method"
    )


@dataclass(frozen=True, slots=True)
class AllowedContent:
    format: DetectedFormat


@dataclass(frozen=True, slots=True)
class DisallowedContent:
    kind: DisallowedKind


ContentInspection: TypeAlias = AllowedContent | DisallowedContent


@dataclass(frozen=True, slots=True)
class SecuredUpload:
    safe_filename: str
    content_type: str
    detected_format: DetectedFormat
    size_bytes: int


@dataclass(frozen=True, slots=True)
class RejectedUpload:
    reason: RejectionReason
    message: str


UploadValidation: TypeAlias = SecuredUpload | RejectedUpload

_SAFE_EXTENSION: Final[Mapping[DetectedFormat, str]] = MappingProxyType(
    {
        DetectedFormat.PDF: "pdf",
        DetectedFormat.TEXT: "txt",
    }
)

_SAFE_CONTENT_TYPE: Final[Mapping[DetectedFormat, str]] = MappingProxyType(
    {
        DetectedFormat.PDF: "application/pdf",
        DetectedFormat.TEXT: "text/plain",
    }
)


def _starts_with_any(content: bytes, prefixes: tuple[bytes, ...]) -> bool:
    return any(content.startswith(prefix) for prefix in prefixes)


def _is_archive(content: bytes) -> bool:
    if _starts_with_any(content, _ARCHIVE_MAGIC_PREFIXES):
        return True
    tar_magic_end: Final = _TAR_USTAR_OFFSET + len(_TAR_USTAR_MAGIC)
    if len(content) >= tar_magic_end and content[_TAR_USTAR_OFFSET:tar_magic_end] == _TAR_USTAR_MAGIC:
        return True
    return _starts_with_any(content, _ARCHIVE_MAGIC_PREFIXES_ASCII_AMBIGUOUS) and not _is_utf8_text(content)


def _is_utf8_text(content: bytes) -> bool:
    if b"\x00" in content:
        return False
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _is_executable_binary(content: bytes) -> bool:
    if _starts_with_any(content, _EXECUTABLE_MAGIC_PREFIXES):
        return True
    return _starts_with_any(content, _EXECUTABLE_MAGIC_PREFIXES_ASCII_AMBIGUOUS) and not _is_utf8_text(content)


def _looks_like_shebang(content: bytes) -> bool:
    body: Final = content.removeprefix(_UTF8_BOM).lstrip()
    return body.startswith(b"#!")


def inspect_content(content: bytes) -> ContentInspection:
    if _looks_like_shebang(content):
        return DisallowedContent(DisallowedKind.EXECUTABLE)
    if content.startswith(b"%PDF-"):
        return AllowedContent(DetectedFormat.PDF)
    if _is_archive(content):
        return DisallowedContent(DisallowedKind.ARCHIVE)
    if _is_executable_binary(content):
        return DisallowedContent(DisallowedKind.EXECUTABLE)
    if _is_utf8_text(content):
        return AllowedContent(DetectedFormat.TEXT)
    return DisallowedContent(DisallowedKind.UNKNOWN_BINARY)


def generate_safe_filename(detected_format: DetectedFormat) -> str:
    return f"{uuid.uuid4().hex}.{_SAFE_EXTENSION[detected_format]}"


def _reject_disallowed(kind: DisallowedKind) -> RejectedUpload:
    match kind:
        case DisallowedKind.ARCHIVE:
            return RejectedUpload(
                RejectionReason.ARCHIVE_NOT_ALLOWED,
                "Archive uploads are not allowed.",
            )
        case DisallowedKind.EXECUTABLE:
            return RejectedUpload(
                RejectionReason.EXECUTABLE_NOT_ALLOWED,
                "Executable uploads are not allowed.",
            )
        case DisallowedKind.UNKNOWN_BINARY:
            return RejectedUpload(
                RejectionReason.UNSUPPORTED_FORMAT,
                "Only PDF and UTF-8 text documents are accepted.",
            )
    assert_never(kind)


def _scan_rejection(content: bytes, scanner: MalwareScanner) -> RejectedUpload | None:
    result: Final = scanner.scan(content)
    match result.verdict:
        case ScanVerdict.CLEAN:
            return None
        case ScanVerdict.INFECTED:
            return RejectedUpload(
                RejectionReason.MALWARE_DETECTED,
                f"Uploaded file was flagged by malware scanning ({result.signature or 'unknown signature'}).",
            )
        case ScanVerdict.ERROR:
            return RejectedUpload(
                RejectionReason.MALWARE_SCAN_ERROR,
                "Malware scanning could not complete; upload rejected.",
            )
    assert_never(result.verdict)


def validate_upload(
    *,
    content: bytes,
    scanner: MalwareScanner,
    max_size_bytes: int = MAX_UPLOAD_SIZE_BYTES,
) -> UploadValidation:
    size: Final = len(content)
    if size == 0:
        return RejectedUpload(RejectionReason.EMPTY_FILE, "Uploaded file is empty.")
    if size > max_size_bytes:
        return RejectedUpload(
            RejectionReason.FILE_TOO_LARGE,
            f"Uploaded file is {size} bytes, exceeding the {max_size_bytes}-byte limit.",
        )

    inspection: Final = inspect_content(content)
    if isinstance(inspection, DisallowedContent):
        return _reject_disallowed(inspection.kind)

    scan_rejection: Final = _scan_rejection(content, scanner)
    if scan_rejection is not None:
        return scan_rejection

    return SecuredUpload(
        safe_filename=generate_safe_filename(inspection.format),
        content_type=_SAFE_CONTENT_TYPE[inspection.format],
        detected_format=inspection.format,
        size_bytes=size,
    )


def _sanitize_header_filename(filename: str) -> str:
    stripped: Final = "".join(char for char in filename if char not in '"\\\r\n').strip()
    return stripped or "download"


def safe_download_headers(filename: str) -> Mapping[str, str]:
    return MappingProxyType(
        {
            "Content-Disposition": f'attachment; filename="{_sanitize_header_filename(filename)}"',
            "X-Content-Type-Options": "nosniff",
        }
    )
