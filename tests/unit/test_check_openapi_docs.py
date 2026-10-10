"""Tests for scripts/check_openapi_docs.py, the scan behind scripts/openapi_docs_gate.py."""

from typing import Final

import check_openapi_docs as checker

_BODY: Final = {"content": {"application/json": {"schema": {"type": "object"}}}}
_RESPONSES: Final = {"200": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Out"}}}}}
_DOCUMENTED: Final = {"description": "Does a thing", "requestBody": _BODY, "responses": _RESPONSES}


def test_a_fully_documented_post_has_no_findings() -> None:
    assert checker.violations({"paths": {"/v1/thing": {"post": _DOCUMENTED}}}) == ()


def test_each_missing_piece_is_its_own_rule() -> None:
    bare: Final = {"summary": "Thing", "responses": {"200": {"description": "Successful Response"}}}
    codes: Final = {v.code for v in checker.violations({"paths": {"/v1/thing": {"post": bare}}})}
    assert codes == {"OA001", "OA002", "OA003"}


def test_only_the_missing_piece_is_reported() -> None:
    no_body: Final = {**_DOCUMENTED, "requestBody": {}}
    blank_description: Final = {**_DOCUMENTED, "description": "   "}
    untyped_200: Final = {**_DOCUMENTED, "responses": {"200": {"content": {"application/json": {"schema": {}}}}}}
    spec: Final = {
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
    spec: Final = {"paths": {"/z": {"post": {}}, "/a": {"post": {}}}}
    assert [v.path for v in checker.violations(spec)] == ["/a", "/a", "/a", "/z", "/z", "/z"]


def test_main_reads_a_spec_file_and_prints_one_line_per_finding(tmp_path, capsys) -> None:
    spec_file: Final = tmp_path / "openapi.json"
    spec_file.write_text('{"paths": {"/v1/thing": {"post": {"description": "x", "requestBody": {}}}}}')
    assert checker.main([str(spec_file)]) == 0
    assert capsys.readouterr().out.splitlines() == ["POST /v1/thing OA001", "POST /v1/thing OA003"]
