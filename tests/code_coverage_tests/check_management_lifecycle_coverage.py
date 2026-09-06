"""Every management resource with a write route needs full lifecycle e2e coverage,
or an entry in management_lifecycle_baseline.json.

A write route is any POST, PUT, PATCH or DELETE registration on a router in these
modules, whether it is written as a decorator or as an `add_api_route` call. A
resource is the first path segment of its write routes (`/key/generate` is `key`),
after any `/v1` or `/v2` version segment and behind its router's own prefix
(`/v1/mcp` + `/server/register` is `mcp_server`). Its lifecycle is covered when
`tests/e2e/coverage_registry/mgmt.yaml` holds, and a collected non-skipped e2e test
declares, `mgmt.<resource>.<create>.persists` for the create verb the resource's own
routes use (`new`, `generate`, `add` or `register`, so `/key/generate` wants
`mgmt.key.generate.persists`), `mgmt.<resource>.update.preserves_unrelated_fields`,
`mgmt.<resource>.update.clear_persists` and `mgmt.<resource>.delete.persists`.

The gate fails on a resource with write routes that is neither covered nor baselined,
and on a baseline entry that is now covered or no longer has write routes, so the
baseline never carries stale headroom. `--write-baseline` regenerates the file from the
current tree for local use and exits non-zero when it changed.

The gate itself only ever removes entries. Nothing here stops `--write-baseline` from
adding one for a brand new uncovered resource, so that growth is caught by a human
reading the one-line JSON diff in the pull request, not by this script.

    uv run --no-sync python tests/code_coverage_tests/check_management_lifecycle_coverage.py
"""

from __future__ import annotations

import ast
import json
import sys
from argparse import ArgumentParser
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, TypeAdapter
from typing_extensions import assert_never

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
MANAGEMENT_ENDPOINTS_DIR: Final = REPO_ROOT / "litellm" / "proxy" / "management_endpoints"
E2E_DIR: Final = REPO_ROOT / "tests" / "e2e"
BASELINE_PATH: Final = Path(__file__).resolve().with_name("management_lifecycle_baseline.json")

sys.path.insert(0, str(E2E_DIR))  # test-quality-ok: absolute path to the coverage_registry package rooted at tests/e2e

from coverage_registry.collector import collect_markers
from coverage_registry.registry import load_registry

WRITE_METHODS: Final = frozenset({"POST", "PUT", "PATCH", "DELETE"})
ROUTE_DECORATORS: Final = frozenset(method.lower() for method in WRITE_METHODS) | frozenset({"api_route"})
VERSION_SEGMENTS: Final = frozenset({"v1", "v2"})
CREATE_ROUTES: Final = ("new", "generate", "add", "register")
LIFECYCLE_TAILS: Final = ("update.preserves_unrelated_fields", "update.clear_persists", "delete.persists")

_BASELINE_ADAPTER: Final = TypeAdapter(tuple[str, ...])


@dataclass(frozen=True, slots=True)
class WriteRoute:
    method: str
    path: str
    location: str
    resource: str


@dataclass(frozen=True, slots=True)
class UncoveredResource:
    resource: str
    routes: tuple[WriteRoute, ...]
    missing_cells: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StaleBaselineEntry:
    resource: str
    reason: Literal["covered", "no_write_routes"]


@dataclass(frozen=True, slots=True)
class GatePassed:
    covered: frozenset[str]
    baselined: frozenset[str]


@dataclass(frozen=True, slots=True)
class GateFailed:
    uncovered: tuple[UncoveredResource, ...]
    stale: tuple[StaleBaselineEntry, ...]


Verdict: TypeAlias = GatePassed | GateFailed


def _segments(route: str) -> tuple[str, ...]:
    return tuple(segment for segment in route.split("/") if segment)


def resource_name(prefix: str, path: str) -> str:
    namespace: Final = tuple(segment for segment in _segments(prefix) if segment not in VERSION_SEGMENTS)
    segments: Final = _segments(path)
    unversioned: Final = segments[1:] if segments and segments[0] in VERSION_SEGMENTS else segments
    parts: Final = (*namespace, *unversioned[:1])
    if not parts:
        raise ValueError(f"cannot name a resource for route {prefix + path!r}")
    return "_".join(parts).lower()


def _string_assignment(stmt: ast.stmt) -> tuple[tuple[str, str], ...]:
    match stmt:
        case ast.Assign(targets=[ast.Name(id=name)], value=ast.Constant(value=str() as value)):
            return ((name, value),)
        case ast.AnnAssign(target=ast.Name(id=name), value=ast.Constant(value=str() as value)):
            return ((name, value),)
        case _:
            return ()


def _module_string_constants(tree: ast.Module) -> Mapping[str, str]:
    return MappingProxyType({name: value for stmt in tree.body for name, value in _string_assignment(stmt)})


def _relative_source(module_path: Path, node: ast.ImportFrom) -> Path | None:
    if node.level == 0 or node.module is None:
        return None
    source: Final = module_path.parents[node.level - 1].joinpath(*node.module.split(".")).with_suffix(".py")
    return source if source.is_file() else None


def _imported_string_constants(tree: ast.Module, module_path: Path) -> Mapping[str, str]:
    imports: Final = tuple(
        (stmt, source)
        for stmt in tree.body
        if isinstance(stmt, ast.ImportFrom)
        for source in (_relative_source(module_path, stmt),)
        if source is not None
    )
    return MappingProxyType(
        {
            alias.asname or alias.name: value
            for stmt, source in imports
            for constants in (_module_string_constants(_parse(source)),)
            for alias in stmt.names
            for value in (constants.get(alias.name),)
            if value is not None
        }
    )


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _is_api_router_call(call: ast.Call) -> bool:
    match call.func:
        case ast.Name(id="APIRouter") | ast.Attribute(attr="APIRouter"):
            return True
        case _:
            return False


def _router_prefix(call: ast.Call, constants: Mapping[str, str]) -> str | None:
    prefix: Final = next((keyword.value for keyword in call.keywords if keyword.arg == "prefix"), None)
    match prefix:
        case None:
            return ""
        case ast.Constant(value=str() as value):
            return value
        case ast.Name(id=name):
            return constants.get(name)
        case _:
            return None


def _router_assignment(stmt: ast.stmt, constants: Mapping[str, str]) -> tuple[tuple[str, str | None], ...]:
    match stmt:
        case ast.Assign(targets=[ast.Name(id=name)], value=ast.Call() as call) if _is_api_router_call(call):
            return ((name, _router_prefix(call, constants)),)
        case ast.AnnAssign(target=ast.Name(id=name), value=ast.Call() as call) if _is_api_router_call(call):
            return ((name, _router_prefix(call, constants)),)
        case _:
            return ()


def _routers(tree: ast.Module, module_path: Path) -> Mapping[str, str | None]:
    constants: Final = MappingProxyType(
        {**_imported_string_constants(tree, module_path), **_module_string_constants(tree)}
    )
    return MappingProxyType(
        {name: prefix for stmt in tree.body for name, prefix in _router_assignment(stmt, constants)}
    )


def _route_path(call: ast.Call) -> str | None:
    positional: Final = call.args[0] if call.args else None
    keyword: Final = next((keyword.value for keyword in call.keywords if keyword.arg == "path"), None)
    match positional or keyword:
        case ast.Constant(value=str() as value):
            return value
        case _:
            return None


def _declared_methods(call: ast.Call, location: str) -> tuple[str, ...]:
    methods: Final = next((keyword.value for keyword in call.keywords if keyword.arg == "methods"), None)
    match methods:
        case None:
            return ()
        case ast.List(elts=elements) | ast.Tuple(elts=elements) | ast.Set(elts=elements):
            named: Final = tuple(
                e.value.upper() for e in elements if isinstance(e, ast.Constant) and isinstance(e.value, str)
            )
            if len(named) != len(elements):
                raise ValueError(f"{location}: route methods are not all string literals")
            return tuple(method for method in named if method in WRITE_METHODS)
        case _:
            raise ValueError(f"{location}: route methods are not a literal list")


def _registration_methods(call: ast.Call, attribute: str, location: str) -> tuple[str, ...]:
    if attribute in ("api_route", "add_api_route"):
        return _declared_methods(call, location)
    return (attribute.upper(),)


def _route_registrations(tree: ast.Module) -> tuple[tuple[ast.Call, str, str], ...]:
    decorators: Final = tuple(
        (decorator, decorator.func.attr, decorator.func.value.id)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and isinstance(decorator.func.value, ast.Name)
        and decorator.func.attr in ROUTE_DECORATORS
    )
    added: Final = tuple(
        (node, node.func.attr, node.func.value.id)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.attr == "add_api_route"
    )
    return decorators + added


def _write_route(call: ast.Call, method: str, prefix: str | None, location: str) -> WriteRoute:
    path: Final = _route_path(call)
    if prefix is None:
        raise ValueError(f"{location}: cannot resolve the prefix of the router this write route is declared on")
    if path is None:
        raise ValueError(f"{location}: write route path is not a string literal")
    return WriteRoute(method=method, path=prefix + path, location=location, resource=resource_name(prefix, path))


def _module_write_routes(module_path: Path, package_dir: Path) -> tuple[WriteRoute, ...]:
    tree: Final = _parse(module_path)
    routers: Final = _routers(tree, module_path)
    relative: Final = module_path.relative_to(package_dir).as_posix()
    return tuple(
        _write_route(call, method, routers.get(router), f"{relative}:{call.lineno}")
        for call, attribute, router in _route_registrations(tree)
        for method in _registration_methods(call, attribute, f"{relative}:{call.lineno}")
    )


def write_routes_by_resource(package_dir: Path) -> Mapping[str, tuple[WriteRoute, ...]]:
    routes: Final = tuple(
        route
        for module_path in sorted(package_dir.rglob("*.py"))
        for route in _module_write_routes(module_path, package_dir)
    )
    return MappingProxyType(
        {
            resource: tuple(route for route in routes if route.resource == resource)
            for resource in sorted(frozenset(route.resource for route in routes))
        }
    )


def _create_verbs(routes: tuple[WriteRoute, ...]) -> tuple[str, ...]:
    declared: Final = frozenset(
        segment for route in routes for segment in _segments(route.path) if segment in CREATE_ROUTES
    )
    return tuple(verb for verb in CREATE_ROUTES if verb in declared) or CREATE_ROUTES


def _lifecycle_candidates(resource: str, routes: tuple[WriteRoute, ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        (f"mgmt.{resource}.{create}.persists", *(f"mgmt.{resource}.{tail}" for tail in LIFECYCLE_TAILS))
        for create in _create_verbs(routes)
    )


def _missing_cells(resource: str, routes: tuple[WriteRoute, ...], proven: frozenset[str]) -> tuple[str, ...]:
    closest: Final = min(_lifecycle_candidates(resource, routes), key=lambda cells: len(frozenset(cells) - proven))
    return tuple(cell for cell in closest if cell not in proven)


def evaluate(
    routes: Mapping[str, tuple[WriteRoute, ...]],
    registry_ids: frozenset[str],
    covered_markers: frozenset[str],
    baseline: frozenset[str],
) -> Verdict:
    proven: Final = registry_ids & covered_markers
    covered: Final = frozenset(
        resource for resource, declared in routes.items() if not _missing_cells(resource, declared, proven)
    )
    uncovered: Final = tuple(
        UncoveredResource(resource, routes[resource], _missing_cells(resource, routes[resource], proven))
        for resource in sorted(routes)
        if resource not in covered and resource not in baseline
    )
    stale: Final = tuple(
        StaleBaselineEntry(resource, "covered" if resource in covered else "no_write_routes")
        for resource in sorted(baseline)
        if resource in covered or resource not in routes
    )
    if uncovered or stale:
        return GateFailed(uncovered=uncovered, stale=stale)
    return GatePassed(covered=covered, baselined=baseline)


def _render_uncovered(entry: UncoveredResource) -> str:
    routes: Final = tuple(f"  {route.method} {route.path}  ({route.location})" for route in entry.routes)
    return "\n".join(
        (
            f"{entry.resource}: management write routes without full lifecycle e2e coverage, "
            f"and not in {BASELINE_PATH.name}",
            *routes,
            "  add these cells to tests/e2e/coverage_registry/mgmt.yaml and declare them with "
            "@pytest.mark.covers on a non-skipped tests/e2e test: " + ", ".join(entry.missing_cells),
        )
    )


def _render_stale(entry: StaleBaselineEntry) -> str:
    match entry.reason:
        case "covered":
            return (
                f"{entry.resource}: baseline entry is stale, the resource now has full lifecycle coverage; "
                f"delete it from {BASELINE_PATH.name}"
            )
        case "no_write_routes":
            return (
                f"{entry.resource}: baseline entry is stale, no management write route maps to it; "
                f"delete it from {BASELINE_PATH.name}"
            )
        case _:
            assert_never(entry.reason)


def render(verdict: Verdict) -> str:
    match verdict:
        case GatePassed(covered=covered, baselined=baselined):
            return (
                f"management lifecycle coverage: {len(covered)} resource(s) fully covered, "
                f"{len(baselined)} baselined without full lifecycle coverage"
            )
        case GateFailed(uncovered=uncovered, stale=stale):
            return "\n".join((*map(_render_uncovered, uncovered), *map(_render_stale, stale)))
        case _:
            assert_never(verdict)


def read_baseline(path: Path) -> frozenset[str]:
    return frozenset(_BASELINE_ADAPTER.validate_json(path.read_text(encoding="utf-8")))


def render_baseline(resources: Iterable[str]) -> str:
    return json.dumps(sorted(resources), indent=2) + "\n"


def write_baseline(
    path: Path,
    routes: Mapping[str, tuple[WriteRoute, ...]],
    registry_ids: frozenset[str],
    covered_markers: frozenset[str],
) -> int:
    proven: Final = registry_ids & covered_markers
    content: Final = render_baseline(
        resource for resource, declared in routes.items() if _missing_cells(resource, declared, proven)
    )
    previous: Final = path.read_text(encoding="utf-8") if path.is_file() else None
    path.write_text(content, encoding="utf-8")
    if content == previous:
        print(f"{path.name} is unchanged")
        return 0
    print(f"{path.name} rewritten from the current tree; commit it")
    return 1


class _CliArgs(BaseModel):
    write_baseline: bool


def main(argv: Sequence[str] | None = None) -> int:
    parser: Final = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Regenerate the baseline from the current tree and exit non-zero if it changed. Local use only.",
    )
    args: Final = _CliArgs.model_validate(vars(parser.parse_args(argv)))
    routes: Final = write_routes_by_resource(MANAGEMENT_ENDPOINTS_DIR)
    registry_ids: Final = frozenset(cell.id for cell in load_registry() if cell.module == "mgmt")
    markers: Final = collect_markers()
    if markers.collection_errors:
        print(
            f"{len(markers.collection_errors)} tests/e2e node(s) failed to import during collection, "
            "so lifecycle coverage cannot be computed:\n  " + "\n  ".join(markers.collection_errors)
        )
        return 1
    if args.write_baseline:
        return write_baseline(BASELINE_PATH, routes, registry_ids, markers.covered)
    verdict: Final = evaluate(routes, registry_ids, markers.covered, read_baseline(BASELINE_PATH))
    print(render(verdict))
    match verdict:
        case GatePassed():
            return 0
        case GateFailed():
            return 1
        case _:
            assert_never(verdict)


if __name__ == "__main__":
    sys.exit(main())
