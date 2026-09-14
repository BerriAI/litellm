import io

import pytest

from litellm.proxy._types import ProxyException
from litellm.proxy.openai_files_endpoints.general_upload_validation import (
    MB,
    UploadedFileBlockedExtension,
    UploadedFileTooLarge,
    UploadedFileUnsafeFilename,
    check_blocked_extension,
    check_unsafe_filename,
    check_upload_file_size,
    raise_upload_validation_failure,
)


def test_size_under_cap_allowed():
    assert check_upload_file_size(b"x" * 100, 1) is None


def test_size_over_cap_rejected_for_bytes():
    content = b"x" * (2 * MB)
    assert check_upload_file_size(content, 1) == UploadedFileTooLarge(size_bytes=len(content), limit_mb=1)


def test_size_over_cap_rejected_for_binaryio_and_restores_caller_position():
    """The handle is caller-owned; inspecting its size must not discard where the caller had it."""
    content = b"x" * (2 * MB)
    handle = io.BytesIO(content)
    handle.seek(17)
    assert check_upload_file_size(handle, 1) == UploadedFileTooLarge(size_bytes=len(content), limit_mb=1)
    assert handle.tell() == 17


def test_size_under_cap_allowed_for_binaryio_restores_caller_position():
    handle = io.BytesIO(b"x" * 100)
    handle.seek(42)
    assert check_upload_file_size(handle, 1) is None
    assert handle.tell() == 42


def test_size_exactly_at_cap_allowed():
    content = b"x" * MB
    assert check_upload_file_size(content, 1) is None


def test_no_cap_skips_size_check():
    assert check_upload_file_size(b"x" * (10 * MB), None) is None


@pytest.mark.parametrize("cap", [0, -3])
def test_nonpositive_cap_disables_size_check(cap):
    assert check_upload_file_size(b"x" * (10 * MB), cap) is None


def test_blocked_extension_rejected():
    assert check_blocked_extension("payload.exe", (".exe", ".sh")) == UploadedFileBlockedExtension(extension=".exe")


def test_blocked_extension_match_is_case_insensitive():
    assert check_blocked_extension("payload.EXE", (".exe",)) == UploadedFileBlockedExtension(extension=".exe")


def test_blocked_extension_match_is_case_insensitive_for_configured_value():
    """A config entry like blocked_file_extensions: ['.EXE'] must still catch a lowercase upload."""
    assert check_blocked_extension("payload.exe", (".EXE",)) == UploadedFileBlockedExtension(extension=".exe")


def test_extension_not_in_blocklist_allowed():
    assert check_blocked_extension("report.pdf", (".exe", ".sh")) is None


def test_empty_blocklist_allows_everything():
    assert check_blocked_extension("payload.exe", ()) is None


def test_no_filename_skips_extension_check():
    assert check_blocked_extension(None, (".exe",)) is None


def test_path_traversal_filename_rejected():
    assert check_unsafe_filename("../../etc/passwd") == UploadedFileUnsafeFilename(filename="../../etc/passwd")


def test_windows_style_path_traversal_filename_rejected():
    assert check_unsafe_filename("..\\..\\windows\\system32\\config") == UploadedFileUnsafeFilename(
        filename="..\\..\\windows\\system32\\config"
    )


def test_traversal_embedded_after_extension_rejected():
    assert check_unsafe_filename("report.jsonl/../../etc/cron.d/evil") == UploadedFileUnsafeFilename(
        filename="report.jsonl/../../etc/cron.d/evil"
    )


def test_null_byte_filename_rejected():
    assert check_unsafe_filename("report.pdf\x00.exe") == UploadedFileUnsafeFilename(filename="report.pdf\x00.exe")


@pytest.mark.parametrize("filename", ["report.pdf", ".env", "a.b.c.jsonl", "my file (1).csv", None])
def test_ordinary_filenames_allowed(filename):
    assert check_unsafe_filename(filename) is None


@pytest.mark.parametrize(
    "failure, expected_code, expected_fragments",
    [
        (
            UploadedFileTooLarge(size_bytes=15728640, limit_mb=10),
            "413",
            ("15.0 MB", "max_file_size_mb", "10 MB", "not forwarded"),
        ),
        (
            UploadedFileBlockedExtension(extension=".exe"),
            "400",
            (".exe", "blocked_file_extensions", "not forwarded"),
        ),
        (
            UploadedFileUnsafeFilename(filename="../../etc/passwd"),
            "400",
            ("../../etc/passwd", "traversal", "not forwarded"),
        ),
    ],
)
def test_failures_map_to_openai_shaped_proxy_exceptions(failure, expected_code, expected_fragments):
    with pytest.raises(ProxyException) as exc_info:
        raise_upload_validation_failure(failure)
    assert exc_info.value.code == expected_code
    assert exc_info.value.type == "invalid_request_error"
    assert exc_info.value.param == "file"
    for fragment in expected_fragments:
        assert fragment in exc_info.value.message
