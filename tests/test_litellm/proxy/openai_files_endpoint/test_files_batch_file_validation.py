import io

import pytest

from litellm.proxy._types import ProxyException
from litellm.proxy.openai_files_endpoints.batch_file_validation import (
    BATCH_LINE_REQUIRED_KEYS,
    BatchFileEmpty,
    BatchFileInvalidJsonLine,
    BatchFileLineNotObject,
    BatchFileMissingLineKey,
    BatchFileTooLarge,
    BatchFileWrongExtension,
    check_batch_file_upload,
    raise_batch_file_validation_failure,
)

VALID_LINE = (
    b'{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions",'
    b' "body": {"model": "gpt-4.1-nano", "messages": [{"role": "user", "content": "hi"}]}}'
)


def test_valid_bytes_pass():
    assert check_batch_file_upload("batch.jsonl", VALID_LINE + b"\n" + VALID_LINE + b"\n", 10) is None


def test_valid_binaryio_passes_and_resets_position():
    handle = io.BytesIO(VALID_LINE + b"\n" + VALID_LINE + b"\n")
    handle.seek(17)
    assert check_batch_file_upload("batch.jsonl", handle, 10) is None
    assert handle.tell() == 0


def test_uppercase_extension_accepted():
    assert check_batch_file_upload("BATCH.JSONL", VALID_LINE, None) is None


@pytest.mark.parametrize("filename", ["batch.csv", "batch.json", "batch", None])
def test_wrong_extension_rejected(filename):
    assert check_batch_file_upload(filename, VALID_LINE, None) == BatchFileWrongExtension(filename=filename or "")


def test_size_over_cap_rejected_for_bytes():
    content = b"x" * (2 * 1024 * 1024)
    assert check_batch_file_upload("batch.jsonl", content, 1) == BatchFileTooLarge(
        size_bytes=len(content), limit_mb=1
    )


def test_size_over_cap_rejected_for_binaryio():
    content = b"x" * (2 * 1024 * 1024)
    assert check_batch_file_upload("batch.jsonl", io.BytesIO(content), 1) == BatchFileTooLarge(
        size_bytes=len(content), limit_mb=1
    )


def test_size_exactly_at_cap_allowed():
    line = VALID_LINE + b"\n"
    padding_key = b'{"custom_id": "pad", "method": "POST", "url": "/v1/chat/completions", "body": {"note": "'
    pad_line = padding_key + b"a" * (1024 * 1024 - len(line) - len(padding_key) - len(b'"}}\n')) + b'"}}\n'
    content = line + pad_line
    assert len(content) == 1024 * 1024
    assert check_batch_file_upload("batch.jsonl", content, 1) is None


def test_no_cap_skips_size_check():
    content = (VALID_LINE + b"\n") * 5000
    assert check_batch_file_upload("batch.jsonl", content, None) is None


@pytest.mark.parametrize("cap", [0, -3])
def test_nonpositive_cap_disables_size_check(cap):
    content = (VALID_LINE + b"\n") * 5000
    assert check_batch_file_upload("batch.jsonl", content, cap) is None


@pytest.mark.parametrize("content", [b"", b"\n\n", b"  \n\t\n"])
def test_empty_file_rejected(content):
    assert check_batch_file_upload("batch.jsonl", content, None) == BatchFileEmpty()


def test_invalid_json_line_rejected_with_line_number():
    content = VALID_LINE + b"\n" + b"not json at all\n" + VALID_LINE + b"\n"
    assert check_batch_file_upload("batch.jsonl", content, None) == BatchFileInvalidJsonLine(line_number=2)


def test_non_utf8_line_rejected_as_invalid_json():
    assert check_batch_file_upload("batch.jsonl", b"\xff\xfe\x00\x01\n", None) == BatchFileInvalidJsonLine(
        line_number=1
    )


def test_non_object_line_rejected():
    content = VALID_LINE + b"\n" + b'["custom_id", "method"]\n'
    assert check_batch_file_upload("batch.jsonl", content, None) == BatchFileLineNotObject(line_number=2)


@pytest.mark.parametrize("missing_key", BATCH_LINE_REQUIRED_KEYS)
def test_missing_required_key_rejected(missing_key):
    import json

    line_dict = {
        "custom_id": "req-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "gpt-4.1-nano"},
    }
    del line_dict[missing_key]
    content = VALID_LINE + b"\n" + json.dumps(line_dict).encode() + b"\n"
    assert check_batch_file_upload("batch.jsonl", content, None) == BatchFileMissingLineKey(
        line_number=2, key=missing_key
    )


def test_blank_lines_do_not_shift_line_numbers():
    content = b"\n" + VALID_LINE + b"\n\n" + b"broken\n"
    assert check_batch_file_upload("batch.jsonl", content, None) == BatchFileInvalidJsonLine(line_number=4)


def test_failed_scan_leaves_handle_open_and_reset():
    handle = io.BytesIO(b"not json\n" + VALID_LINE + b"\n")
    assert check_batch_file_upload("batch.jsonl", handle, None) == BatchFileInvalidJsonLine(line_number=1)
    assert not handle.closed
    assert handle.tell() == 0


def test_scan_stops_at_first_failure():
    class ExplodingLines(io.BytesIO):
        def __init__(self):
            super().__init__(b"not json\n" + VALID_LINE + b"\n")
            self.lines_read = 0

        def __next__(self):
            self.lines_read += 1
            return super().__next__()

    handle = ExplodingLines()
    assert check_batch_file_upload("batch.jsonl", handle, None) == BatchFileInvalidJsonLine(line_number=1)
    assert handle.lines_read == 1


@pytest.mark.parametrize(
    "failure, expected_code, expected_param, expected_fragments",
    [
        (
            BatchFileTooLarge(size_bytes=220200960, limit_mb=10),
            "413",
            "file",
            ("210.0 MB", "max_batch_file_size_mb", "10 MB", "not forwarded"),
        ),
        (
            BatchFileWrongExtension(filename="batch.csv"),
            "400",
            "file",
            ("batch.csv", ".jsonl", "not forwarded"),
        ),
        (BatchFileEmpty(), "400", "file", ("no request lines", "not forwarded")),
        (BatchFileInvalidJsonLine(line_number=3), "400", "file", ("line 3", "not valid JSON")),
        (BatchFileLineNotObject(line_number=2), "400", "file", ("line 2", "JSON object")),
        (
            BatchFileMissingLineKey(line_number=5, key="method"),
            "400",
            "method",
            ("'method'", "line 5", "custom_id, method, url, body"),
        ),
    ],
)
def test_failures_map_to_openai_shaped_proxy_exceptions(failure, expected_code, expected_param, expected_fragments):
    with pytest.raises(ProxyException) as exc_info:
        raise_batch_file_validation_failure(failure)
    assert exc_info.value.code == expected_code
    assert exc_info.value.type == "invalid_request_error"
    assert exc_info.value.param == expected_param
    for fragment in expected_fragments:
        assert fragment in exc_info.value.message
