"""Unit tests for vector-store upload security controls.

These pin the pentest M4 remediation: an allowlist enforced by real content
inspection (not extension/mime trust), a size cap, archive and executable
rejection, server-generated filenames, safe download headers, and a
dependency-injected malware scanner validated with the EICAR test file.
"""

from dataclasses import dataclass

import pytest

from litellm.proxy.rag_endpoints.upload_security import (
    EICAR_TEST_SIGNATURE,
    DetectedFormat,
    EicarTestMalwareScanner,
    RejectedUpload,
    RejectionReason,
    ScanResult,
    ScanVerdict,
    SecuredUpload,
    generate_safe_filename,
    inspect_content,
    safe_download_headers,
    validate_upload,
)


@dataclass(frozen=True)
class _StubScanner:
    result: ScanResult

    def scan(self, content: bytes) -> ScanResult:
        return self.result


_CLEAN_SCANNER = _StubScanner(ScanResult(ScanVerdict.CLEAN))
_INFECTED_SCANNER = _StubScanner(ScanResult(ScanVerdict.INFECTED, signature="Test.Sig"))
_ERROR_SCANNER = _StubScanner(ScanResult(ScanVerdict.ERROR))

_PDF_BYTES = b"%PDF-1.7\n1 0 obj<<>>endobj\n"
_TEXT_BYTES = "the quick brown fox\n".encode("utf-8")
_ELF_BYTES = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 32
_PE_BYTES = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00"
_ZIP_BYTES = b"PK\x03\x04\x14\x00\x00\x00"
_GZIP_BYTES = b"\x1f\x8b\x08\x00\x00\x00\x00\x00"
_SHEBANG_BYTES = b"#!/bin/bash\nrm -rf /\n"


def _tar_bytes() -> bytes:
    header = bytearray(512)
    header[257:262] = b"ustar"
    return bytes(header)


def _expect_rejected(content: bytes, reason: RejectionReason, *, max_size_bytes: int = 512 * 1024 * 1024) -> None:
    result = validate_upload(content=content, scanner=_CLEAN_SCANNER, max_size_bytes=max_size_bytes)
    assert isinstance(result, RejectedUpload), f"expected rejection, got {result!r}"
    assert result.reason is reason, f"expected {reason}, got {result.reason}"


def test_empty_file_rejected():
    _expect_rejected(b"", RejectionReason.EMPTY_FILE)


def test_oversized_file_rejected():
    _expect_rejected(b"%PDF-" + b"a" * 100, RejectionReason.FILE_TOO_LARGE, max_size_bytes=10)


def test_zip_archive_rejected():
    _expect_rejected(_ZIP_BYTES, RejectionReason.ARCHIVE_NOT_ALLOWED)


def test_gzip_archive_rejected():
    _expect_rejected(_GZIP_BYTES, RejectionReason.ARCHIVE_NOT_ALLOWED)


def test_tar_archive_rejected():
    _expect_rejected(_tar_bytes(), RejectionReason.ARCHIVE_NOT_ALLOWED)


def test_elf_executable_rejected():
    _expect_rejected(_ELF_BYTES, RejectionReason.EXECUTABLE_NOT_ALLOWED)


def test_windows_pe_executable_rejected():
    _expect_rejected(_PE_BYTES, RejectionReason.EXECUTABLE_NOT_ALLOWED)


def test_shebang_script_rejected():
    _expect_rejected(_SHEBANG_BYTES, RejectionReason.EXECUTABLE_NOT_ALLOWED)


def test_unknown_binary_rejected():
    _expect_rejected(b"\x89\x01\x02\x00\xff\xfe garbage", RejectionReason.UNSUPPORTED_FORMAT)


def test_pdf_accepted_with_server_filename_and_content_type():
    result = validate_upload(content=_PDF_BYTES, scanner=_CLEAN_SCANNER)
    assert isinstance(result, SecuredUpload)
    assert result.detected_format is DetectedFormat.PDF
    assert result.content_type == "application/pdf"
    assert result.safe_filename.endswith(".pdf")
    assert result.size_bytes == len(_PDF_BYTES)


def test_utf8_text_accepted():
    result = validate_upload(content=_TEXT_BYTES, scanner=_CLEAN_SCANNER)
    assert isinstance(result, SecuredUpload)
    assert result.detected_format is DetectedFormat.TEXT
    assert result.content_type == "text/plain"
    assert result.safe_filename.endswith(".txt")


def test_inspect_content_classifies_directly():
    from litellm.proxy.rag_endpoints.upload_security import AllowedContent, DisallowedContent, DisallowedKind

    assert inspect_content(_PDF_BYTES) == AllowedContent(DetectedFormat.PDF)
    assert inspect_content(_TEXT_BYTES) == AllowedContent(DetectedFormat.TEXT)
    assert inspect_content(_ZIP_BYTES) == DisallowedContent(DisallowedKind.ARCHIVE)
    assert inspect_content(_ELF_BYTES) == DisallowedContent(DisallowedKind.EXECUTABLE)


def test_server_generated_filenames_are_unique_and_ignore_client_name():
    first = generate_safe_filename(DetectedFormat.PDF)
    second = generate_safe_filename(DetectedFormat.PDF)
    assert first != second
    assert first.endswith(".pdf")
    assert "/" not in first and "\\" not in first


def test_malware_hook_blocks_infected_clean_format():
    result = validate_upload(content=_TEXT_BYTES, scanner=_INFECTED_SCANNER)
    assert isinstance(result, RejectedUpload)
    assert result.reason is RejectionReason.MALWARE_DETECTED
    assert "Test.Sig" in result.message


def test_malware_scan_error_fails_closed():
    result = validate_upload(content=_TEXT_BYTES, scanner=_ERROR_SCANNER)
    assert isinstance(result, RejectedUpload)
    assert result.reason is RejectionReason.MALWARE_SCAN_ERROR


def test_injected_clean_scanner_allows_valid_file():
    result = validate_upload(content=_TEXT_BYTES, scanner=_CLEAN_SCANNER)
    assert isinstance(result, SecuredUpload)


def test_eicar_default_scanner_flags_only_eicar():
    scanner = EicarTestMalwareScanner()
    assert scanner.scan(EICAR_TEST_SIGNATURE).verdict is ScanVerdict.INFECTED
    assert scanner.scan(b"totally benign text").verdict is ScanVerdict.CLEAN


def test_eicar_upload_passes_format_but_blocked_by_scanner():
    """EICAR is valid ASCII text, so only the malware hook can stop it."""
    format_only = validate_upload(content=EICAR_TEST_SIGNATURE, scanner=_CLEAN_SCANNER)
    assert isinstance(format_only, SecuredUpload)

    scanned = validate_upload(content=EICAR_TEST_SIGNATURE, scanner=EicarTestMalwareScanner())
    assert isinstance(scanned, RejectedUpload)
    assert scanned.reason is RejectionReason.MALWARE_DETECTED


def test_safe_download_headers_force_attachment_and_nosniff():
    headers = safe_download_headers("file_abc123")
    assert headers["Content-Disposition"] == 'attachment; filename="file_abc123"'
    assert headers["X-Content-Type-Options"] == "nosniff"


@pytest.mark.parametrize("hostile", ['a"; drop', "a\r\nSet-Cookie: x=1", "../../etc/passwd", ""])
def test_safe_download_headers_sanitize_injection(hostile):
    disposition = safe_download_headers(hostile)["Content-Disposition"]
    assert "\r" not in disposition and "\n" not in disposition
    assert disposition.count('"') == 2
