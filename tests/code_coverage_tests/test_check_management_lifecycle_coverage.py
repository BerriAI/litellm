from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
from check_management_lifecycle_coverage import (
    GateFailed,
    GatePassed,
    StaleBaselineEntry,
    WriteRoute,
    evaluate,
    read_baseline,
    render,
    resource_name,
    write_baseline,
    write_routes_by_resource,
)
from pydantic import ValidationError

KEY_ROUTES: Final = (
    WriteRoute(method="POST", path="/key/generate", location="key_management_endpoints.py:10", resource="key"),
    WriteRoute(method="POST", path="/key/delete", location="key_management_endpoints.py:20", resource="key"),
)
ROUTES: Final = MappingProxyType({"key": KEY_ROUTES})
KEY_LIFECYCLE: Final = frozenset(
    {
        "mgmt.key.generate.persists",
        "mgmt.key.update.preserves_unrelated_fields",
        "mgmt.key.update.clear_persists",
        "mgmt.key.delete.persists",
    }
)


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)


def test_write_routes_are_grouped_by_resource_from_fixture_modules(tmp_path: Path) -> None:
    _write(
        tmp_path / "key_endpoints.py",
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n"
        '@router.post("/key/generate")\n'
        "async def generate(): ...\n"
        '@router.get("/key/info")\n'
        "async def info(): ...\n"
        '@router.post("/v2/key/info")\n'
        "async def info_v2(): ...\n"
        '@router.delete(path="/key/{key_id}")\n'
        "def delete(): ...\n",
    )
    _write(
        tmp_path / "mcp_endpoints.py",
        "from typing import Final\n"
        "from fastapi import APIRouter\n"
        'router: Final = APIRouter(prefix="/v1/mcp", tags=["mcp"])\n'
        '@router.post("/server/register")\n'
        "async def register(): ...\n"
        '@router.put("/server/{server_id}/approve")\n'
        "async def approve(): ...\n"
        '@router.patch("/toolset")\n'
        "async def patch_toolset(): ...\n",
    )
    _write(tmp_path / "v1" / "common.py", 'PREFIX: Final = "/management/v1"\n')
    _write(
        tmp_path / "v1" / "budgets.py",
        "from fastapi import APIRouter\n"
        "from .common import PREFIX\n"
        "router = APIRouter(prefix=PREFIX)\n"
        '@router.post("/budgets")\n'
        "async def create(): ...\n",
    )
    _write(
        tmp_path / "aliases.py",
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n"
        '@router.api_route("/widget/legacy", methods=["POST", "GET"])\n'
        "async def legacy(): ...\n"
        '@router.api_route("/widget/readonly", methods=["GET"])\n'
        "async def readonly(): ...\n"
        'router.add_api_route("/v1/widget", create, methods=["POST"])\n'
        'router.add_api_route("/v1/widget/{widget_id}", replace, methods=["PUT", "DELETE"])\n'
        'router.add_api_route("/v1/widget/{widget_id}", read, methods=["GET"])\n'
        'router.add_api_route("/v1/widget/{widget_id}", implicit_get)\n',
    )
    _write(
        tmp_path / "scim.py",
        "from fastapi import APIRouter\n"
        'scim_router = APIRouter(prefix="/scim/v2")\n'
        '@scim_router.post("/Users")\n'
        "async def create_user(): ...\n",
    )

    routes: Final = write_routes_by_resource(tmp_path)

    assert frozenset(routes) == frozenset(
        {"key", "mcp_server", "mcp_toolset", "management_budgets", "scim_users", "widget"}
    )
    assert routes["widget"] == (
        WriteRoute(method="POST", path="/widget/legacy", location="aliases.py:3", resource="widget"),
        WriteRoute(method="POST", path="/v1/widget", location="aliases.py:7", resource="widget"),
        WriteRoute(method="PUT", path="/v1/widget/{widget_id}", location="aliases.py:8", resource="widget"),
        WriteRoute(method="DELETE", path="/v1/widget/{widget_id}", location="aliases.py:8", resource="widget"),
    )
    assert routes["key"] == (
        WriteRoute(method="POST", path="/key/generate", location="key_endpoints.py:3", resource="key"),
        WriteRoute(method="POST", path="/v2/key/info", location="key_endpoints.py:7", resource="key"),
        WriteRoute(method="DELETE", path="/key/{key_id}", location="key_endpoints.py:9", resource="key"),
    )
    assert routes["mcp_server"] == (
        WriteRoute(method="POST", path="/v1/mcp/server/register", location="mcp_endpoints.py:4", resource="mcp_server"),
        WriteRoute(
            method="PUT",
            path="/v1/mcp/server/{server_id}/approve",
            location="mcp_endpoints.py:6",
            resource="mcp_server",
        ),
    )
    assert routes["mcp_toolset"] == (
        WriteRoute(method="PATCH", path="/v1/mcp/toolset", location="mcp_endpoints.py:8", resource="mcp_toolset"),
    )
    assert routes["management_budgets"] == (
        WriteRoute(
            method="POST", path="/management/v1/budgets", location="v1/budgets.py:4", resource="management_budgets"
        ),
    )
    assert routes["scim_users"] == (
        WriteRoute(method="POST", path="/scim/v2/Users", location="scim.py:3", resource="scim_users"),
    )


@pytest.mark.parametrize(
    "source",
    (
        "from fastapi import APIRouter\n"
        "router = APIRouter(prefix=compute_prefix())\n"
        '@router.post("/thing")\n'
        "async def create(): ...\n",
        'from typing import Final\nfrom .shared import router\n@router.post("/thing")\nasync def create(): ...\n',
        "from fastapi import APIRouter\n"
        "router = APIRouter(prefix=compute_prefix())\n"
        'router.add_api_route("/thing", create, methods=["POST"])\n',
    ),
    ids=("dynamic_prefix", "router_not_assigned_in_module", "dynamic_prefix_add_api_route"),
)
def test_write_route_whose_router_prefix_cannot_be_resolved_is_an_error(tmp_path: Path, source: str) -> None:
    _write(tmp_path / "dynamic.py", source)

    with pytest.raises(ValueError, match=r"dynamic\.py:3"):
        write_routes_by_resource(tmp_path)


@pytest.mark.parametrize(
    "registration",
    (
        'router.add_api_route("/thing", create, methods=build_methods())\n',
        'router.add_api_route("/thing", create, methods=[POST])\n',
        '@router.api_route("/thing", methods=build_methods())\nasync def create(): ...\n',
    ),
    ids=("computed_list", "non_literal_member", "decorator_computed_list"),
)
def test_route_registration_with_non_literal_methods_is_an_error(tmp_path: Path, registration: str) -> None:
    _write(tmp_path / "dynamic.py", "from fastapi import APIRouter\nrouter = APIRouter()\n" + registration)

    with pytest.raises(ValueError, match=r"dynamic\.py:3"):
        write_routes_by_resource(tmp_path)


def test_uncovered_resource_not_in_baseline_fails_naming_routes_and_missing_cells() -> None:
    verdict: Final = evaluate(
        ROUTES,
        registry_ids=KEY_LIFECYCLE,
        covered_markers=frozenset({"mgmt.key.generate.persists", "mgmt.key.delete.persists"}),
        baseline=frozenset(),
    )

    assert isinstance(verdict, GateFailed)
    assert tuple(u.resource for u in verdict.uncovered) == ("key",)
    assert verdict.uncovered[0].routes == KEY_ROUTES
    assert verdict.uncovered[0].missing_cells == (
        "mgmt.key.update.preserves_unrelated_fields",
        "mgmt.key.update.clear_persists",
    )
    assert verdict.stale == ()
    report: Final = render(verdict)
    assert "POST /key/generate" in report
    assert "key_management_endpoints.py:20" in report
    assert "mgmt.key.update.clear_persists" in report


def test_baselined_and_now_covered_fails_as_stale() -> None:
    verdict: Final = evaluate(
        ROUTES, registry_ids=KEY_LIFECYCLE, covered_markers=KEY_LIFECYCLE, baseline=frozenset({"key"})
    )

    assert verdict == GateFailed(uncovered=(), stale=(StaleBaselineEntry(resource="key", reason="covered"),))
    assert "delete" in render(verdict)
    assert "key" in render(verdict)


def test_baselined_resource_without_write_routes_fails_as_stale() -> None:
    verdict: Final = evaluate(
        ROUTES, registry_ids=KEY_LIFECYCLE, covered_markers=frozenset(), baseline=frozenset({"key", "ghost"})
    )

    assert verdict == GateFailed(uncovered=(), stale=(StaleBaselineEntry(resource="ghost", reason="no_write_routes"),))
    assert "ghost" in render(verdict)
    assert "delete" in render(verdict)


def test_covered_resource_passes() -> None:
    verdict: Final = evaluate(ROUTES, registry_ids=KEY_LIFECYCLE, covered_markers=KEY_LIFECYCLE, baseline=frozenset())

    assert verdict == GatePassed(covered=frozenset({"key"}), baselined=frozenset())


def test_baselined_and_uncovered_passes() -> None:
    verdict: Final = evaluate(
        ROUTES, registry_ids=KEY_LIFECYCLE, covered_markers=frozenset(), baseline=frozenset({"key"})
    )

    assert verdict == GatePassed(covered=frozenset(), baselined=frozenset({"key"}))


def test_marker_outside_the_registry_does_not_count() -> None:
    verdict: Final = evaluate(
        ROUTES,
        registry_ids=KEY_LIFECYCLE - frozenset({"mgmt.key.update.clear_persists"}),
        covered_markers=KEY_LIFECYCLE,
        baseline=frozenset(),
    )

    assert isinstance(verdict, GateFailed)
    assert verdict.uncovered[0].missing_cells == ("mgmt.key.update.clear_persists",)


def _lifecycle_for(resource: str, create: str) -> frozenset[str]:
    return frozenset(
        {
            f"mgmt.{resource}.{create}.persists",
            f"mgmt.{resource}.update.preserves_unrelated_fields",
            f"mgmt.{resource}.update.clear_persists",
            f"mgmt.{resource}.delete.persists",
        }
    )


@pytest.mark.parametrize("create", ("new", "generate", "add", "register"))
def test_the_create_verb_a_resource_route_uses_completes_its_lifecycle(create: str) -> None:
    routes: Final = MappingProxyType(
        {
            "thing": (
                WriteRoute(method="POST", path=f"/thing/{create}", location="thing.py:1", resource="thing"),
                WriteRoute(method="POST", path="/thing/delete", location="thing.py:2", resource="thing"),
            )
        }
    )
    lifecycle: Final = _lifecycle_for("thing", create)

    verdict: Final = evaluate(routes, registry_ids=lifecycle, covered_markers=lifecycle, baseline=frozenset())

    assert verdict == GatePassed(covered=frozenset({"thing"}), baselined=frozenset())


@pytest.mark.parametrize("create", ("new", "add", "register"))
def test_a_create_verb_the_resource_does_not_use_does_not_complete_its_lifecycle(create: str) -> None:
    lifecycle: Final = _lifecycle_for("key", create)

    verdict: Final = evaluate(ROUTES, registry_ids=lifecycle, covered_markers=lifecycle, baseline=frozenset())

    assert isinstance(verdict, GateFailed)
    assert verdict.uncovered[0].missing_cells == ("mgmt.key.generate.persists",)


def test_a_resource_with_no_create_verb_in_any_route_accepts_any_of_them() -> None:
    routes: Final = MappingProxyType(
        {
            "access_group": (
                WriteRoute(method="POST", path="/v1/access_group", location="ag.py:1", resource="access_group"),
                WriteRoute(method="DELETE", path="/v1/access_group/{id}", location="ag.py:2", resource="access_group"),
            )
        }
    )
    lifecycle: Final = _lifecycle_for("access_group", "new")

    verdict: Final = evaluate(routes, registry_ids=lifecycle, covered_markers=lifecycle, baseline=frozenset())

    assert verdict == GatePassed(covered=frozenset({"access_group"}), baselined=frozenset())


def test_a_create_cell_alone_is_not_a_lifecycle() -> None:
    verdict: Final = evaluate(
        ROUTES,
        registry_ids=KEY_LIFECYCLE,
        covered_markers=frozenset({"mgmt.key.generate.persists"}),
        baseline=frozenset(),
    )

    assert isinstance(verdict, GateFailed)
    assert verdict.uncovered[0].missing_cells == (
        "mgmt.key.update.preserves_unrelated_fields",
        "mgmt.key.update.clear_persists",
        "mgmt.key.delete.persists",
    )


@pytest.mark.parametrize(
    "content",
    ('{"key": true}', "[1, 2]", "not json", "[]extra"),
    ids=("object", "non_strings", "not_json", "trailing_garbage"),
)
def test_a_baseline_that_is_not_a_list_of_strings_is_rejected(tmp_path: Path, content: str) -> None:
    baseline_path: Final = tmp_path / "baseline.json"
    baseline_path.write_text(content)

    with pytest.raises(ValidationError):
        read_baseline(baseline_path)


def test_a_route_with_no_nameable_segment_is_an_error() -> None:
    with pytest.raises(ValueError, match=r"cannot name a resource"):
        resource_name("", "/")


def test_write_baseline_lists_uncovered_resources_sorted_and_reports_change(tmp_path: Path) -> None:
    baseline_path: Final = tmp_path / "baseline.json"
    routes: Final = MappingProxyType(
        {
            "team": (WriteRoute(method="POST", path="/team/new", location="team_endpoints.py:1", resource="team"),),
            "key": KEY_ROUTES,
            "budget": (WriteRoute(method="POST", path="/budget/new", location="budget.py:1", resource="budget"),),
        }
    )

    first: Final = write_baseline(baseline_path, routes, registry_ids=KEY_LIFECYCLE, covered_markers=KEY_LIFECYCLE)

    assert first == 1
    assert baseline_path.read_text() == '[\n  "budget",\n  "team"\n]\n'
    assert read_baseline(baseline_path) == frozenset({"budget", "team"})

    second: Final = write_baseline(baseline_path, routes, registry_ids=KEY_LIFECYCLE, covered_markers=KEY_LIFECYCLE)

    assert second == 0
