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


def _shared_pagination(page: Annotated[int, Query()] = 1) -> int:
    return page


def test_declared_query_params_includes_a_param_declared_only_on_a_shared_sub_dependency():
    """`page` sits on `_shared_pagination`'s own Dependant, nested under the route's
    `dependencies`, not on the route function's own `query_params` — the case a
    non-recursive walk over just the top-level Dependant would miss."""
    captured: dict[str, frozenset[str]] = {}
    app = FastAPI()

    @app.get("/probe")
    def _handler(request: Request, page: Annotated[int, Depends(_shared_pagination)]) -> dict[str, bool]:
        captured["declared"] = _declared_query_params(request)
        return {"ok": True}

    TestClient(app).get("/probe")
    assert captured["declared"] == frozenset({"page"})


MANAGEMENT_V1_PACKAGE = Path(str(common_module.__file__)).parent


def _public_names(module: ModuleType) -> frozenset[str]:
    return frozenset(name for name in vars(module) if not name.startswith("_"))


def _modules_imported_from_by(source_file: Path) -> frozenset[str]:
    tree = ast.parse(source_file.read_text())
    return frozenset(node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module)


@pytest.mark.parametrize("source_file", sorted(MANAGEMENT_V1_PACKAGE.glob("*.py")), ids=lambda path: path.name)
def test_no_module_imports_from_the_private_fastapi_dependencies_utils_module(source_file: Path):
    """fastapi.dependencies.utils is a private module fastapi has already removed a name
    from once (get_flat_dependant, in 0.140.7) without notice; the names it still exposes
    (e.g. get_flat_params) carry the same risk. Nothing here should import from it at all,
    only from the stable public Dependant dataclass in fastapi.dependencies.models.
    """
    assert "fastapi.dependencies.utils" not in _modules_imported_from_by(source_file)


def test_common_still_imports_with_fastapi_dependencies_utils_emptied_out(monkeypatch: pytest.MonkeyPatch):
    for name in ("get_flat_dependant", "get_flat_params"):
        monkeypatch.delattr(fastapi_dependency_utils, name, raising=False)
    spec = importlib.util.spec_from_file_location(
        "management_v1_common__simulated_fastapi", Path(str(common_module.__file__))
    )
    assert spec is not None and spec.loader is not None
    reimported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reimported)
    assert _public_names(reimported) == _public_names(common_module)
