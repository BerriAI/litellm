import ast
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Annotated

import fastapi.dependencies.utils as fastapi_dependency_utils
import pytest
from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.testclient import TestClient

import litellm.proxy.list_api.common as common_module
from litellm.proxy.list_api.common import (
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
    """`_query_param_aliases` walks `Dependant.query_params`; if it were ever widened
    to also read `.path_params`, `thing_id` would leak into the declared set."""
    response = _client().get("/things/abc", params={"thing_id": "x"})
    assert response.status_code == 400
    assert "thing_id" in response.json()["detail"]


def test_a_header_param_name_is_not_a_declared_query_param():
    response = _client().get("/things/abc", params={"x-trace": "x"})
    assert response.status_code == 400
    assert "x-trace" in response.json()["detail"]


def test_declared_query_params_isolates_query_aliases_from_other_param_types():
    app = FastAPI()

    @app.get("/things/{thing_id}")
    def _handler(
        thing_id: str,
        request: Request,
        status: Annotated[str | None, Query(alias="filter[status]")] = None,
        page: Annotated[int, Query(ge=1)] = 1,
        x_trace: Annotated[str | None, Header()] = None,
    ) -> dict[str, list[str]]:
        return {"declared": sorted(_declared_query_params(request))}

    response = TestClient(app).get("/things/abc")
    assert frozenset(response.json()["declared"]) == frozenset({"filter[status]", "page"})


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


def _shared_pagination(page: Annotated[int, Query()] = 1) -> int:
    return page


def test_declared_query_params_includes_a_param_declared_only_on_a_shared_sub_dependency():
    """`page` sits on `_shared_pagination`'s own Dependant, nested under the route's
    `dependencies`, not on the route function's own `query_params` — the case a
    non-recursive walk over just the top-level Dependant would miss."""
    app = FastAPI()

    @app.get("/probe")
    def _handler(request: Request, page: Annotated[int, Depends(_shared_pagination)]) -> dict[str, list[str]]:
        return {"declared": sorted(_declared_query_params(request))}

    response = TestClient(app).get("/probe")
    assert frozenset(response.json()["declared"]) == frozenset({"page"})


LIST_API_PACKAGE = Path(str(common_module.__file__)).parent
PROXY_PACKAGE = LIST_API_PACKAGE.parent
GUARDED_PACKAGES = (
    LIST_API_PACKAGE,
    PROXY_PACKAGE / "management_endpoints" / "management_v1",
    PROXY_PACKAGE / "public_endpoints" / "public_v1",
)
FRAMEWORK_SOURCE_FILES = sorted(
    (path for package in GUARDED_PACKAGES for path in package.glob("*.py")),
    key=lambda path: (path.parent.name, path.name),
)


def _public_names(module: ModuleType) -> frozenset[str]:
    return frozenset(name for name in vars(module) if not name.startswith("_"))


def _modules_imported_from_by(source_file: Path) -> frozenset[str]:
    tree = ast.parse(source_file.read_text())
    return frozenset(node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module)


@pytest.mark.parametrize("source_file", FRAMEWORK_SOURCE_FILES, ids=lambda path: f"{path.parent.name}/{path.name}")
def test_no_module_imports_from_the_private_fastapi_dependencies_utils_module(source_file: Path):
    """fastapi.dependencies.utils is a private module fastapi has already removed a name
    from once (get_flat_dependant, in 0.140.7) without notice; the names it still exposes
    (e.g. get_flat_params) carry the same risk. `proxy_server` imports every one of these
    packages unguarded at module level, so an ImportError here takes down the whole proxy
    rather than just these routes. Globbing them means a new module is covered on sight.
    """
    assert "fastapi.dependencies.utils" not in _modules_imported_from_by(source_file)


def test_common_still_imports_with_fastapi_dependencies_utils_emptied_out(monkeypatch: pytest.MonkeyPatch):
    for name in ("get_flat_dependant", "get_flat_params"):
        monkeypatch.delattr(fastapi_dependency_utils, name, raising=False)
    spec = importlib.util.spec_from_file_location(
        "list_api_common__simulated_fastapi", Path(str(common_module.__file__))
    )
    assert spec is not None and spec.loader is not None
    reimported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reimported)
    assert _public_names(reimported) == _public_names(common_module)
