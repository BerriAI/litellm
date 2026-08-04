from typing import Annotated

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.testclient import TestClient

from litellm.proxy.management_endpoints.management_v1.common import (
    ManagementProblem,
    PROBLEM_CONTENT_TYPE,
    _declared_query_params,
    problem_response,
    reject_unknown_query_params,
)


def _client() -> TestClient:
    app = FastAPI()

    @app.exception_handler(ManagementProblem)
    async def _handle(_request: Request, exc: ManagementProblem):
        return problem_response(exc.problem)

    @app.get("/things/{thing_id}", dependencies=[Depends(reject_unknown_query_params)])
    def _handler(
        thing_id: str,
        request: Request,
        status: Annotated[str | None, Query(alias="filter[status]")] = None,
        page: Annotated[int, Query(ge=1)] = 1,
        x_trace: Annotated[str | None, Header()] = None,
    ) -> dict[str, bool]:
        return {"ok": True}

    return TestClient(app, raise_server_exceptions=False)


def test_a_declared_query_param_is_accepted_by_its_alias():
    response = _client().get("/things/abc", params={"filter[status]": "active", "page": "2"})
    assert response.status_code == 200, response.text


def test_an_unknown_query_param_is_rejected_as_a_problem():
    response = _client().get("/things/abc", params={"bogus": "x"})
    assert response.status_code == 400
    assert response.headers["content-type"].startswith(PROBLEM_CONTENT_TYPE)
    assert "bogus" in response.json()["detail"]


def test_a_path_param_name_is_not_a_declared_query_param():
    """The flatten step returns path+query+header together; only query names count as declared.

    If the ParamTypes.query filter were dropped, `thing_id` (a path param) would leak
    into the declared set and this request would be wrongly accepted.
    """
    response = _client().get("/things/abc", params={"thing_id": "x"})
    assert response.status_code == 400
    assert "thing_id" in response.json()["detail"]


def test_a_header_param_name_is_not_a_declared_query_param():
    response = _client().get("/things/abc", params={"x-trace": "x"})
    assert response.status_code == 400
    assert "x-trace" in response.json()["detail"]


def test_declared_query_params_isolates_query_aliases_from_other_param_types():
    captured: dict[str, frozenset[str]] = {}
    app = FastAPI()

    @app.get("/things/{thing_id}")
    def _handler(
        thing_id: str,
        request: Request,
        status: Annotated[str | None, Query(alias="filter[status]")] = None,
        page: Annotated[int, Query(ge=1)] = 1,
        x_trace: Annotated[str | None, Header()] = None,
    ) -> dict[str, bool]:
        captured["declared"] = _declared_query_params(request)
        return {"ok": True}

    TestClient(app).get("/things/abc")
    assert captured["declared"] == frozenset({"filter[status]", "page"})


def test_declared_query_params_is_empty_when_the_route_has_no_dependant():
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "root_path": "",
            "path": "/things/abc",
            "query_string": b"",
            "headers": [(b"host", b"testserver")],
        }
    )
    assert _declared_query_params(request) == frozenset()
