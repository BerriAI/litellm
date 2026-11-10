import ast
from collections.abc import Iterator
from itertools import chain
from pathlib import Path
import re
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parents[4]
SOURCE_FILES: Final = (
    REPO_ROOT / "litellm/proxy/management_endpoints/key_management_endpoints.py",
    REPO_ROOT / "litellm/proxy/management_endpoints/team_endpoints.py",
)
TEST_DIRS: Final = (
    REPO_ROOT / "tests/integration/management",
    REPO_ROOT / "tests/integration/authorization",
)
SELF: Final = Path(__file__).resolve()

_ROUTE_DECORATOR = re.compile(
    r"@router\.(?:get|post|put|delete|patch)\(\s*[\"']([^\"']+)[\"']"
)


def _source_routes() -> frozenset[str]:
    return frozenset(chain.from_iterable(_ROUTE_DECORATOR.findall(path.read_text()) for path in SOURCE_FILES))


def _route_to_regex(route: str) -> re.Pattern[str]:
    def route_part(part: str) -> str:
        if part.startswith("{") and part.endswith("}"):
            return "[^?]+" if part.endswith(":path}") else "[^/?]+"
        return re.escape(part)

    pattern: Final = "".join(route_part(part) for part in re.split(r"(\{[^}]+\})", route))
    return re.compile(f"^{pattern}$")


def _url_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else "X"
            for value in node.values
        )
    return None


def _urls(path: Path) -> Iterator[str]:
    tree: Final = ast.parse(path.read_text())
    return (
        literal.split("?", 1)[0]
        for node in ast.walk(tree)
        if (literal := _url_literal(node)) is not None and literal.startswith("/")
    )


def _directory_urls(directory: Path) -> Iterator[str]:
    return chain.from_iterable(
        _urls(path)
        for path in sorted(directory.glob("test_*.py"))
        if path.resolve() != SELF
    )


def _test_urls() -> frozenset[str]:
    return frozenset(
        chain.from_iterable(_directory_urls(directory) for directory in TEST_DIRS)
    )


def _uncovered_routes(routes: frozenset[str], urls: frozenset[str]) -> Iterator[str]:
    return (
        route
        for route in routes
        if not any(_route_to_regex(route).match(url) for url in urls)
    )


def test_every_management_route_has_a_behavior_scenario() -> None:
    routes: Final = _source_routes()
    assert routes, "no @router routes parsed"

    urls: Final = _test_urls()
    uncovered: Final = tuple(sorted(_uncovered_routes(routes, urls)))
    assert (
        not uncovered
    ), "management routes with no behavior-suite scenario:\n  " + "\n  ".join(uncovered)
