"""Contract: uploads and listings the proxy refuses itself, before any provider is called.

`POST /v1/files` validates the upload locally: the `purpose` value, the filename, the `expires_after` pair,
and, for `purpose=batch`, the batch input itself (it must be a `.jsonl` file with at least one request line,
and every non-blank line must be a JSON object with `custom_id`, `method`, `url` and `body`). `GET /v1/files`
validates `limit`. Each refusal is an OpenAI-shaped error `{"error": {"message", "type", "param", "code"}}`
with a fixed status and message that client code and dashboards match on, and in every case nothing reaches
the upstream: the docstrings quote "The file was not forwarded to the provider" because the tests check it.

A team can also force an expiry on every file its keys upload through team metadata
`enforced_file_expires_after`; that is covered here because it rewrites the caller's `expires_after`.

Assumed: `dec1_resources_support.booted_rig`; error bodies follow `ProxyException.to_dict`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import httpx
import pytest
from integration.providers.dec1_resources_support import (
    Rig,
    batch_line,
    booted_rig,
    error_of,
    jsonl,
    multipart_field,
    unique,
)

_LINE: Final = jsonl(batch_line("gpt-4o-mini"))
_NOT_FORWARDED: Final = "The file was not forwarded to the provider."


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with booted_rig(tmp_path_factory.mktemp("dec1-resources-files-validation")) as booted:
        yield booted


def _refused(rig: Rig, response: httpx.Response, status: int, message: str, param: str | None) -> None:
    assert response.status_code == status, response.text
    error: Final = error_of(response)
    assert error["message"] == message
    assert error["type"] == "invalid_request_error"
    assert error["param"] == param
    assert rig.drained() == ()


def test_unknown_purpose_is_400_naming_the_purpose_parameter(rig: Rig) -> None:
    """A file is uploaded with `purpose=nonsense`.

    The proxy answers 400 with `type: invalid_request_error`, `param: purpose` and a message that starts
    `Invalid purpose: nonsense. Must be one of:` followed by the accepted values. Nothing is sent upstream.
    """
    rig.wire.drain()
    response: Final = rig.upload("purpose.jsonl", _LINE, purpose="nonsense")
    assert response.status_code == 400, response.text
    error: Final = error_of(response)
    assert str(error["message"]).startswith("Invalid purpose: nonsense. Must be one of: (")
    assert "'batch'" in str(error["message"]) and "'fine-tune'" in str(error["message"])
    assert (error["type"], error["param"]) == ("invalid_request_error", "purpose")
    assert rig.drained() == ()


def test_batch_upload_that_is_not_jsonl_is_400(rig: Rig) -> None:
    """A valid batch body is uploaded with `purpose=batch` under the name `input.txt`.

    The proxy answers 400, `param: file`, with the exact message naming the file and saying batch input files
    must be `.jsonl`. The check is on the filename extension, not the content.
    """
    rig.wire.drain()
    _refused(
        rig,
        rig.upload("input.txt", _LINE),
        400,
        f"Invalid file format for Batch API: 'input.txt'. Batch input files must be .jsonl files. {_NOT_FORWARDED}",
        "file",
    )


def test_batch_upload_extension_check_ignores_case(rig: Rig) -> None:
    """A valid batch body is uploaded as `INPUT.JSONL`.

    The extension is compared case-insensitively, so the upload is accepted with 200 and reaches the upstream
    once.
    """
    stem: Final = unique("UPPER")
    rig.wire.drain()
    response: Final = rig.upload(f"{stem}.JSONL", _LINE)
    assert response.status_code == 200, response.text
    assert len(rig.drained()) == 1


@pytest.mark.parametrize("content", [b"", b"\n\n  \n"], ids=["zero-bytes", "only-blank-lines"])
def test_batch_upload_with_no_request_lines_is_400(rig: Rig, content: bytes) -> None:
    """A `.jsonl` batch upload is empty or holds only blank lines.

    The proxy answers 400, `param: file`, message `Batch input file has no request lines. The file was not
    forwarded to the provider.`
    """
    rig.wire.drain()
    _refused(
        rig,
        rig.upload("empty.jsonl", content),
        400,
        f"Batch input file has no request lines. {_NOT_FORWARDED}",
        "file",
    )


def test_batch_upload_with_a_line_that_is_not_json_names_the_physical_line(rig: Rig) -> None:
    """A batch upload has a valid first line, a blank second line and a third line that is not JSON.

    The proxy answers 400, `param: file`, and the message names line 3: blank lines are skipped as records but
    still count toward the reported line number, which is what a user sees in their editor.
    """
    rig.wire.drain()
    _refused(
        rig,
        rig.upload("broken.jsonl", _LINE + b"\n" + b"{not json\n"),
        400,
        f"Batch input file line 3 is not valid JSON. {_NOT_FORWARDED}",
        "file",
    )


def test_batch_upload_with_a_json_line_that_is_not_an_object_is_400(rig: Rig) -> None:
    """A batch upload's second line is the JSON array `[1, 2]`.

    The proxy answers 400, `param: file`, message `Batch input file line 2 must be a JSON object. ...`.
    """
    rig.wire.drain()
    _refused(
        rig,
        rig.upload("array.jsonl", _LINE + b"[1, 2]\n"),
        400,
        f"Batch input file line 2 must be a JSON object. {_NOT_FORWARDED}",
        "file",
    )


@pytest.mark.parametrize("missing", ["custom_id", "method", "url", "body"])
def test_batch_upload_line_missing_a_required_key_names_the_key_as_param(rig: Rig, missing: str) -> None:
    """A batch upload's only line lacks one of `custom_id`, `method`, `url`, `body`.

    The proxy answers 400 with `param` equal to the missing key and the message `Missing required parameter:
    '<key>' (batch input file line 1). Each line must be a JSON object with keys custom_id, method, url, body.
    The file was not forwarded to the provider.`
    """
    line: Final = {name: value for name, value in batch_line("gpt-4o-mini").items() if name != missing}
    rig.wire.drain()
    _refused(
        rig,
        rig.upload("missing.jsonl", jsonl(line)),
        400,
        f"Missing required parameter: '{missing}' (batch input file line 1). "
        f"Each line must be a JSON object with keys custom_id, method, url, body. {_NOT_FORWARDED}",
        missing,
    )


def test_non_batch_upload_is_not_checked_as_batch_input(rig: Rig) -> None:
    """A text file that is not JSONL is uploaded with `purpose=assistants`.

    Batch input validation applies only to `purpose=batch`, so the upload is accepted with 200 and the
    upstream receives the bytes unchanged.
    """
    stem: Final = unique("notes")
    rig.wire.drain()
    response: Final = rig.upload(f"{stem}.txt", b"plain notes, not json\n", purpose="assistants", content_type="text/plain")
    assert response.status_code == 200, response.text
    (request,) = rig.drained()
    assert multipart_field(request, "file") == b"plain notes, not json\n"


def test_filename_with_a_parent_directory_component_is_400(rig: Rig) -> None:
    """A file is uploaded under the name `../escape.jsonl`.

    The proxy answers 400, `param: file`, message `Filename '../escape.jsonl' is not allowed: directory
    traversal sequences are not permitted in uploaded file names. ...`. The check runs before anything else
    reads the file. Assumed: the multipart parser hands the handler the filename with its `..` component.
    """
    rig.wire.drain()
    _refused(
        rig,
        rig.upload("../escape.jsonl", _LINE),
        400,
        "Filename '../escape.jsonl' is not allowed: directory traversal sequences are not permitted in "
        f"uploaded file names. {_NOT_FORWARDED}",
        "file",
    )


def test_filename_starting_with_a_dot_is_accepted(rig: Rig) -> None:
    """A file named `.hidden-<id>.jsonl` is uploaded.

    Only a real `..` path component or a null byte is unsafe; a leading dot is an ordinary name, so the upload
    is accepted with 200.
    """
    response: Final = rig.upload(f".{unique('hidden')}.jsonl", _LINE)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    ("fields", "fragment"),
    [
        (
            {"expires_after[anchor]": "created_at"},
            "Both expires_after[anchor] and expires_after[seconds] must be provided if expires_after is specified",
        ),
        (
            {"expires_after[seconds]": "3600"},
            "Both expires_after[anchor] and expires_after[seconds] must be provided if expires_after is specified",
        ),
        (
            {"expires_after[anchor]": "last_active_at", "expires_after[seconds]": "3600"},
            "expires_after[anchor] must be 'created_at', got 'last_active_at'",
        ),
        (
            {"expires_after[anchor]": "created_at", "expires_after[seconds]": "soon"},
            "expires_after[seconds] must be a valid integer, got 'soon'",
        ),
    ],
    ids=["anchor-only", "seconds-only", "wrong-anchor", "seconds-not-an-integer"],
)
def test_malformed_expires_after_is_400(rig: Rig, fields: dict[str, str], fragment: str) -> None:
    """A file is uploaded with an incomplete or invalid `expires_after` pair.

    The proxy answers 400 and the error message contains the sentence for that case. Nothing is sent
    upstream. The message is the string form of a `{"error": ...}` detail, so the test matches the sentence
    rather than the whole string.
    """
    rig.wire.drain()
    response: Final = rig.upload("expiry.jsonl", _LINE, **fields)
    assert response.status_code == 400, response.text
    assert fragment in str(error_of(response)["message"])
    assert rig.drained() == ()


@pytest.mark.parametrize("missing", ["purpose", "file"])
def test_upload_missing_a_required_form_part_is_422(rig: Rig, missing: str) -> None:
    """A multipart upload omits `purpose` or omits `file`.

    Both are required form parts, so the framework rejects the request with 422 before the handler runs and
    nothing is sent upstream. UNVERIFIED: the 422 body shape; the handler is
    `otel_request_validation_exception_handler` in `litellm/proxy/proxy_server.py`.
    """
    data: Final = {} if missing == "purpose" else {"purpose": "batch"}
    files: Final = {"other": ("other.jsonl", _LINE, "application/jsonl")} if missing == "file" else {
        "file": ("in.jsonl", _LINE, "application/jsonl")
    }
    rig.wire.drain()
    response: Final = rig.proxy.client.post(
        "/v1/files", data=data, files=files, headers={"Authorization": f"Bearer {rig.proxy.key}"}
    )
    assert response.status_code == 422, response.text
    assert rig.drained() == ()


@pytest.mark.parametrize(
    ("limit", "message"),
    [
        ("0", "Invalid 'limit': integer below minimum value. Expected a value >= 1, but got 0 instead."),
        ("-5", "Invalid 'limit': integer below minimum value. Expected a value >= 1, but got -5 instead."),
        ("10001", "Invalid 'limit': integer above maximum value. Expected a value <= 10000, but got 10001 instead."),
    ],
)
def test_file_list_limit_outside_the_openai_range_is_400(rig: Rig, limit: str, message: str) -> None:
    """`GET /v1/files?limit=` is called with 0, a negative number, or 10001.

    The proxy answers 400 with `param: limit`, `type: invalid_request_error` and OpenAI's wording for the
    bound that was crossed. The accepted range is 1 to 10000 inclusive.
    """
    rig.wire.drain()
    _refused(rig, rig.proxy.request("GET", "/v1/files", params={"limit": limit}), 400, message, "limit")


def test_file_list_limit_that_is_not_an_integer_is_422(rig: Rig) -> None:
    """`GET /v1/files?limit=many` is called.

    `limit` is a typed query parameter, so a non-integer is refused with 422 by request validation rather than
    with the 400 above.
    """
    response: Final = rig.proxy.request("GET", "/v1/files", params={"limit": "many"})
    assert response.status_code == 422, response.text


def test_team_enforced_file_expiry_replaces_the_callers_expires_after(rig: Rig) -> None:
    """A team has metadata `enforced_file_expires_after = {"anchor": "created_at", "seconds": 7200}` and one of
    its keys uploads a file asking for 60 seconds.

    The upstream must receive `expires_after[seconds]=7200`: the team setting wins over the caller's value.
    Assumed: team metadata set at `/team/new` reaches `UserAPIKeyAuth.team_metadata`.
    """
    with rig.proxy.scenario() as scenario:
        team: Final = scenario.team(metadata={"enforced_file_expires_after": {"anchor": "created_at", "seconds": 7200}})
        key: Final = scenario.key(team_id=team)
        rig.wire.drain()
        response: Final = rig.upload(
            f"{unique('enforced')}.jsonl",
            _LINE,
            key=key,
            **{"expires_after[anchor]": "created_at", "expires_after[seconds]": "60"},
        )
        assert response.status_code == 200, response.text
        (request,) = rig.drained()
        assert multipart_field(request, "expires_after[anchor]") == b"created_at"
        assert multipart_field(request, "expires_after[seconds]") == b"7200"


def test_team_enforced_file_expiry_applies_when_the_caller_sends_none(rig: Rig) -> None:
    """A team enforces a 7200 second file expiry and its key uploads a file with no `expires_after` at all.

    The upstream must still receive `expires_after[anchor]=created_at` and `expires_after[seconds]=7200`.
    """
    with rig.proxy.scenario() as scenario:
        team: Final = scenario.team(metadata={"enforced_file_expires_after": {"anchor": "created_at", "seconds": 7200}})
        key: Final = scenario.key(team_id=team)
        rig.wire.drain()
        response: Final = rig.upload(f"{unique('enforced-none')}.jsonl", _LINE, key=key)
        assert response.status_code == 200, response.text
        (request,) = rig.drained()
        assert multipart_field(request, "expires_after[seconds]") == b"7200"


@pytest.mark.parametrize(
    ("setting", "fragment"),
    [
        ({"anchor": "created_at"}, "team metadata field 'enforced_file_expires_after' is malformed"),
        (
            {"anchor": "last_active_at", "seconds": 60},
            "team metadata field 'enforced_file_expires_after' has invalid anchor 'last_active_at'",
        ),
    ],
    ids=["missing-seconds", "wrong-anchor"],
)
def test_malformed_team_enforced_file_expiry_is_a_500_not_a_silent_skip(
    rig: Rig, setting: dict[str, str | int], fragment: str
) -> None:
    """A team's `enforced_file_expires_after` lacks `seconds` or names an anchor other than `created_at`.

    An upload by that team's key fails with 500 and a message that names the metadata field, and the file is
    not sent upstream. Failing closed matters: silently skipping the setting would store files the team said
    must expire.
    """
    with rig.proxy.scenario() as scenario:
        team: Final = scenario.team(metadata={"enforced_file_expires_after": dict(setting)})
        key: Final = scenario.key(team_id=team)
        rig.wire.drain()
        response: Final = rig.upload("malformed-team.jsonl", _LINE, key=key)
        assert response.status_code == 500, response.text
        assert fragment in str(error_of(response)["message"])
        assert rig.drained() == ()
