import json
from http import HTTPStatus
from typing import Final

import pytest
from fastapi import HTTPException

from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.management_endpoints.scim.scim_errors import scim_error_from, scim_error_response

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
def test_a_detail_with_no_readable_reason_answers_the_status_phrase(detail: object):
    assert _answer(HTTPException(status_code=404, detail=detail)) == (
        404,
        {"schemas": [_SCIM_ERROR_URN], "status": "404", "detail": HTTPStatus(404).phrase},
    )


def test_an_empty_proxy_exception_with_a_status_python_has_no_phrase_for_answers_the_code_itself():
    assert _answer(_proxy_exception("", 529)) == (529, {"schemas": [_SCIM_ERROR_URN], "status": "529", "detail": "529"})


def test_a_proxy_exception_without_a_numeric_code_answers_500():
    assert _answer(_proxy_exception("boom", None)) == (
        500,
        {"schemas": [_SCIM_ERROR_URN], "status": "500", "detail": "boom"},
    )


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
