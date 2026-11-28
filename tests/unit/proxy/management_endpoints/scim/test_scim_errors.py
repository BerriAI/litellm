import json
from http import HTTPStatus
from typing import Final

import pytest
from fastapi import HTTPException

from litellm.litellm_core_utils.bug_report import NOTICE_PREFIX
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.management_endpoints.scim.scim_errors import (
    ScimError,
    _scim_error_for,
    scim_detail_of,
    scim_error_from,
    scim_error_response,
)
from litellm.proxy.management_endpoints.scim.scim_v2 import RosterWriteFailure, SCIMRosterSyncError
from litellm.proxy.utils import handle_exception_on_proxy

_SCIM_ERROR_URN: Final = "urn:ietf:params:scim:api:messages:2.0:Error"


def _answer(exc: HTTPException | ProxyException) -> tuple[int, dict[str, object]]:
    response: Final = scim_error_response(scim_error_from(exc))
    return response.status_code, json.loads(bytes(response.body))


def _raised_while_handling(origin: Exception, raised: ProxyException) -> ProxyException:
    try:
        try:
            raise origin
        except Exception:
            raise raised
    except ProxyException as exc:
        return exc


def _proxy_exception(message: str, code: int | str | None) -> ProxyException:
    return ProxyException(message=message, type=ProxyErrorTypes.internal_server_error, param=None, code=code)


@pytest.mark.parametrize(
    "detail",
    [{"message": "a shape SCIM does not read"}, ["a", "list"], {"error": 404}],
    ids=["dict-without-error", "list", "non-string-error"],
)
def test_a_detail_with_no_readable_reason_answers_its_text(detail: object):
    assert _answer(HTTPException(status_code=404, detail=detail)) == (
        404,
        {"schemas": [_SCIM_ERROR_URN], "status": "404", "detail": str(detail)},
    )


def test_an_empty_detail_answers_the_status_phrase():
    assert _answer(HTTPException(status_code=404, detail="")) == (
        404,
        {"schemas": [_SCIM_ERROR_URN], "status": "404", "detail": HTTPStatus(404).phrase},
    )


def test_a_proxy_exception_stringified_from_an_unreadable_http_exception_detail_keeps_that_text():
    detail: Final = frozenset({"Master key must be a valid string. Current type=<class 'NoneType'>"})
    raised: Final = _raised_while_handling(
        HTTPException(status_code=500, detail=detail), _proxy_exception(str(detail), 500)
    )

    assert _answer(raised) == (500, {"schemas": [_SCIM_ERROR_URN], "status": "500", "detail": str(detail)})


def test_an_empty_proxy_exception_with_a_status_python_has_no_phrase_for_answers_the_code_itself():
    assert _answer(_proxy_exception("", 529)) == (529, {"schemas": [_SCIM_ERROR_URN], "status": "529", "detail": "529"})


def test_a_proxy_exception_without_a_numeric_code_answers_500():
    assert _answer(_proxy_exception("boom", None)) == (
        500,
        {"schemas": [_SCIM_ERROR_URN], "status": "500", "detail": "boom"},
    )


@pytest.mark.parametrize(
    "text",
    [
        "row decode failed for column metadata",
        f"row decode failed for column metadata\n{NOTICE_PREFIX} File it with one click: https://example.com/new",
    ],
    ids=["plain", "with-bug-report-notice"],
)
def test_the_text_of_an_exception_raised_without_a_status_answers_the_status_phrase(text: str):
    origin: Final = RuntimeError(text)

    assert _answer(_raised_while_handling(origin, handle_exception_on_proxy(origin))) == (
        500,
        {"schemas": [_SCIM_ERROR_URN], "status": "500", "detail": HTTPStatus.INTERNAL_SERVER_ERROR.phrase},
    )


def test_a_server_error_raised_with_its_own_status_keeps_its_reason():
    origin: Final = SCIMRosterSyncError(
        (
            RosterWriteFailure(description="add user1 to team-a: team not found", status_code=404),
            RosterWriteFailure(description="add user1 to team-b: write timed out", status_code=500),
        ),
        attempted=2,
    )

    assert _answer(_raised_while_handling(origin, handle_exception_on_proxy(origin))) == (
        500,
        {"schemas": [_SCIM_ERROR_URN], "status": "500", "detail": str(origin)},
    )


def _http_exception_raised_while_handling(origin: Exception, raised: HTTPException) -> HTTPException:
    try:
        try:
            raise origin
        except Exception:
            raise raised
    except HTTPException as exc:
        return exc


def _team_write_failure_wrapping(origin: Exception) -> HTTPException:
    return _http_exception_raised_while_handling(
        origin,
        HTTPException(
            status_code=500,
            detail={"error": f"Unable to add user - user1, to team - team-b, for reason - {origin}"},
        ),
    )


@pytest.mark.parametrize("converted", [False, True], ids=["as-raised", "converted-by-the-handler"])
def test_a_server_error_that_embeds_the_text_of_an_exception_raised_without_a_status_answers_the_status_phrase(
    converted: bool,
):
    wrapped: Final = _team_write_failure_wrapping(RuntimeError("The table `public.LiteLLM_TeamMembership` is missing"))
    raised: Final = _raised_while_handling(wrapped, handle_exception_on_proxy(wrapped)) if converted else wrapped

    assert _answer(raised) == (
        500,
        {"schemas": [_SCIM_ERROR_URN], "status": "500", "detail": HTTPStatus.INTERNAL_SERVER_ERROR.phrase},
    )


@pytest.mark.parametrize(
    "origin",
    [RuntimeError("row decode failed for column metadata"), RuntimeError()],
    ids=["unrelated-error", "error-with-no-text"],
)
def test_a_server_error_raised_while_handling_an_exception_it_does_not_quote_keeps_its_reason(origin: Exception):
    raised: Final = _http_exception_raised_while_handling(
        origin, HTTPException(status_code=500, detail={"error": "No database connected"})
    )

    assert _answer(raised) == (500, {"schemas": [_SCIM_ERROR_URN], "status": "500", "detail": "No database connected"})


def test_a_client_error_that_embeds_the_text_of_the_exception_it_was_raised_from_keeps_it():
    origin: Final = ValueError("team_id must not be empty")

    assert _answer(
        _http_exception_raised_while_handling(origin, HTTPException(status_code=400, detail=f"Invalid team, {origin}"))
    ) == (400, {"schemas": [_SCIM_ERROR_URN], "status": "400", "detail": "Invalid team, team_id must not be empty"})


def test_a_proxy_exception_raised_while_handling_an_unrelated_http_exception_keeps_its_own_message():
    raised: Final = _raised_while_handling(
        HTTPException(status_code=404, detail={"error": "Group not found with ID: g"}),
        _proxy_exception("Authentication Error, Invalid API key provided.", 401),
    )

    assert _answer(raised) == (
        401,
        {"schemas": [_SCIM_ERROR_URN], "status": "401", "detail": "Authentication Error, Invalid API key provided."},
    )


def test_the_http_exception_headers_reach_the_scim_response():
    response: Final = scim_error_response(
        scim_error_from(HTTPException(status_code=401, detail="no key", headers={"WWW-Authenticate": "Bearer"}))
    )

    assert (response.headers["www-authenticate"], response.headers["content-type"]) == (
        "Bearer",
        "application/scim+json",
    )


def test_direct_scim_error_preserves_explicit_scim_type():
    err: Final = ScimError(400, "Unsupported SCIM filter", scim_type="invalidFilter")
    assert _scim_error_for(err) is err
    assert scim_detail_of(err) == "Unsupported SCIM filter"

    response: Final = scim_error_response(err)
    assert response.status_code == 400
    assert json.loads(bytes(response.body)) == {
        "schemas": [_SCIM_ERROR_URN],
        "status": "400",
        "detail": "Unsupported SCIM filter",
        "scimType": "invalidFilter",
    }

    conflict_response: Final = scim_error_response(ScimError(409, "Already exists"))
    assert conflict_response.status_code == 409
    assert json.loads(bytes(conflict_response.body))["scimType"] == "uniqueness"
