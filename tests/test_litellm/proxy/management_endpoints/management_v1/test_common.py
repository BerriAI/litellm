import ast
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Annotated

import fastapi.dependencies.utils as fastapi_dependency_utils
import pytest
from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.testclient import TestClient

import litellm.proxy.management_endpoints.management_v1.common as common_module
from litellm.proxy.management_endpoints.management_v1.common import (
    PROBLEM_CONTENT_TYPE,
    ManagementProblem,
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


# fastapi removed these in 0.140.7, which `pyproject.toml` still allows via
# `fastapi>=0.136.3,<1.0`. Add a name here whenever a supported release drops one.
FASTAPI_NAMES_REMOVED_IN_0_140_7 = frozenset({"get_flat_dependant"})

MANAGEMENT_V1_PACKAGE = Path(str(common_module.__file__)).parent


def _public_names(module: ModuleType) -> frozenset[str]:
    return frozenset(name for name in vars(module) if not name.startswith("_"))


def _fastapi_names_imported_by(source_file: Path) -> frozenset[str]:
    tree = ast.parse(source_file.read_text())
    return frozenset(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("fastapi")
        for alias in node.names
    )


@pytest.mark.parametrize(
    "source_file", sorted(MANAGEMENT_V1_PACKAGE.glob("*.py")), ids=lambda path: path.name
)
def test_no_module_imports_a_fastapi_name_removed_in_a_supported_release(source_file: Path):
    """`pyproject.toml` allows fastapi up to <1.0, but CI only ever resolves 0.136.3.

    Every other test here passes just as well against a module importing a name
    fastapi has since deleted, because the pinned fastapi still has it. On a user's
    fastapi>=0.140.7 that import is an ImportError, and `proxy_server` imports this
    package unguarded at module level, so it takes the whole proxy down rather than
    just these routes. Globbing the package means a new module is covered on sight.
    """
    assert not _fastapi_names_imported_by(source_file) & FASTAPI_NAMES_REMOVED_IN_0_140_7


def test_common_still_imports_when_fastapi_has_dropped_those_names(monkeypatch: pytest.MonkeyPatch):
    """The static check above cannot prove the module actually loads; this does.

    Behaviour cannot be asserted under the same simulation: on 0.136.3
    `get_flat_params` calls `get_flat_dependant` internally, so it raises NameError
    once the name is gone. Loading is the part this pins.
    """
    for name in FASTAPI_NAMES_REMOVED_IN_0_140_7:
        monkeypatch.delattr(fastapi_dependency_utils, name, raising=False)
    spec = importlib.util.spec_from_file_location(
        "management_v1_common__simulated_fastapi", Path(str(common_module.__file__))
    )
    assert spec is not None and spec.loader is not None
    reimported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reimported)
    assert _public_names(reimported) == _public_names(common_module)
