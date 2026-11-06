"""Tests for scripts/check_openapi_docs.py, the scan behind scripts/openapi_docs_gate.py."""

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import check_openapi_docs as checker
import pytest

_BODY: Final = {"content": {"application/json": {"schema": {"type": "object"}}}}
_RESPONSES: Final = {"200": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Out"}}}}}
_DOCUMENTED: Final = {"description": "Does a thing", "requestBody": _BODY, "responses": _RESPONSES}


def _post_with_media_types(request_media_type: str, response_media_type: str) -> Mapping[str, object]:
    return {
        **_DOCUMENTED,
        "requestBody": {"content": {request_media_type: {"schema": {"type": "string"}}}},
        "responses": {"200": {"content": {response_media_type: {"schema": {"type": "string"}}}}},
    }


@pytest.mark.parametrize(
    "request_type, response_type",
    [
        ("application/json", "application/json"),
        ("application/sdp", "application/sdp"),
        ("multipart/form-data", "application/json"),
    ],
)
def test_a_fully_documented_post_has_no_findings(request_type: str, response_type: str) -> None:
    operation: Final = _post_with_media_types(request_type, response_type)
    assert checker.violations({"paths": {"/v1/thing": {"post": operation}}}) == ()


@pytest.mark.parametrize("status", ["200", "201", "202", "206", "299"])
def test_any_documented_success_response_satisfies_the_response_rule(status: str) -> None:
    operation: Final = {**_DOCUMENTED, "responses": {status: _RESPONSES["200"]}}
    assert checker.violations({"paths": {"/v1/thing": {"post": operation}}}) == ()


@pytest.mark.parametrize("status", ["default", "100", "300", "400", "500", "2000"])
def test_only_success_statuses_can_supply_the_response_schema(status: str) -> None:
    operation: Final = {**_DOCUMENTED, "responses": {status: _RESPONSES["200"]}}
    assert checker.violations({"paths": {"/v1/thing": {"post": operation}}}) == (
        checker.Violation("POST", "/v1/thing", "OA003"),
    )


@pytest.mark.parametrize("status", ["201", "202"])
def test_success_without_a_schema_is_not_documented_by_an_error_response(status: str) -> None:
    operation: Final[Mapping[str, object]] = {
        **_DOCUMENTED,
        "responses": {
            status: {"content": {"application/json": {"schema": {}}}},
            "422": _RESPONSES["200"],
        },
    }
    assert checker.violations({"paths": {"/v1/thing": {"post": operation}}}) == (
        checker.Violation("POST", "/v1/thing", "OA003"),
    )


def test_a_schema_under_another_media_type_still_counts_as_undocumented() -> None:
    operation: Final = _post_with_media_types("application/octet-stream", "audio/wav")
    assert {v.code for v in checker.violations({"paths": {"/v1/thing": {"post": operation}}})} == {"OA001", "OA003"}


def test_each_missing_piece_is_its_own_rule() -> None:
    bare: Final = {"summary": "Thing", "responses": {"200": {"description": "Successful Response"}}}
    codes: Final = {v.code for v in checker.violations({"paths": {"/v1/thing": {"post": bare}}})}
    assert codes == {"OA001", "OA002", "OA003"}


@pytest.mark.parametrize("body", [{}, {"content": {"multipart/form-data": {}}}])
def test_only_the_missing_piece_is_reported(body: Mapping[str, object]) -> None:
    no_body: Final = {**_DOCUMENTED, "requestBody": body}
    blank_description: Final = {**_DOCUMENTED, "description": "   "}
    untyped_200: Final[Mapping[str, object]] = {
        **_DOCUMENTED,
        "responses": {"200": {"content": {"application/json": {"schema": {}}}}},
    }
    spec: Final[Mapping[str, object]] = {
        "paths": {
            "/a": {"post": no_body},
            "/b": {"post": blank_description},
            "/c": {"post": untyped_200},
        }
    }
    assert checker.violations(spec) == (
        checker.Violation("POST", "/a", "OA001"),
        checker.Violation("POST", "/b", "OA002"),
        checker.Violation("POST", "/c", "OA003"),
    )


def test_non_post_operations_are_not_gated() -> None:
    spec: Final = {"paths": {"/v1/thing": {"get": {}, "delete": {}, "put": {}, "post": _DOCUMENTED}}}
    assert checker.violations(spec) == ()


def test_findings_are_sorted_by_path_so_counts_are_stable() -> None:
    spec: Final[Mapping[str, object]] = {"paths": {"/z": {"post": {}}, "/a": {"post": {}}}}
    assert [v.path for v in checker.violations(spec)] == ["/a", "/a", "/a", "/z", "/z", "/z"]


def test_main_reads_a_spec_file_and_prints_one_line_per_finding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec_file: Final = tmp_path / "openapi.json"
    spec_file.write_text('{"paths": {"/v1/thing": {"post": {"description": "x", "requestBody": {}}}}}')
    assert checker.main([str(spec_file)]) == 0
    assert capsys.readouterr().out.splitlines() == ["POST /v1/thing OA001", "POST /v1/thing OA003"]
