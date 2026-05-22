"""PR3.M1 — codified route coverage.

Every route declared in the two management-endpoint source files must be
exercised by at least one behavior-suite scenario. This is a permanent
regression guard: a future route added without a behavior test fails CI here,
the same way test_no_management_imports.py codifies the G3 import grep.
"""

import ast
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SOURCE_FILES = [
    REPO_ROOT / "litellm/proxy/management_endpoints/key_management_endpoints.py",
    REPO_ROOT / "litellm/proxy/management_endpoints/team_endpoints.py",
]
TEST_DIR = pathlib.Path(__file__).resolve().parent
SELF = pathlib.Path(__file__).resolve()

# Captures the route literal from `@router.<method>("<literal>"` — `\s*` spans
# newlines so multi-line decorators are matched too.
_ROUTE_DECORATOR = re.compile(
    r"@router\.(?:get|post|put|delete|patch)\(\s*[\"']([^\"']+)[\"']"
)


def _source_routes() -> set:
    routes: set = set()
    for path in SOURCE_FILES:
        routes.update(_ROUTE_DECORATOR.findall(path.read_text()))
    return routes


def _route_to_regex(route: str) -> re.Pattern:
    # A plain path param ({team_id}) matches a single path segment; a Starlette
    # ':path' param ({key:path}) matches across '/'. Keeping plain params
    # slash-bounded stops a loose regex from falsely reporting a future
    # multi-segment route as already covered.
    pattern = ["^"]
    pos = 0
    for match in re.finditer(r"\{([^}]+)\}", route):
        pattern.append(re.escape(route[pos : match.start()]))
        pattern.append("[^?]+" if match.group(1).endswith(":path") else "[^/?]+")
        pos = match.end()
    pattern.append(re.escape(route[pos:]) + "$")
    return re.compile("".join(pattern))


def _test_urls() -> set:
    """Every request-URL string literal across the behavior test suite.

    f-strings are reconstructed with each interpolation collapsed to a single
    placeholder char, so f"/key/{target}/regenerate" becomes /key/X/regenerate.
    Query strings are dropped — coverage is a path-level property.
    """
    urls: set = set()
    for path in sorted(TEST_DIR.glob("test_*.py")):
        if path.resolve() == SELF:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            literal = None
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                literal = node.value
            elif isinstance(node, ast.JoinedStr):
                chunks = []
                for value in node.values:
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        chunks.append(value.value)
                    else:
                        chunks.append("X")  # interpolated path / query segment
                literal = "".join(chunks)
            if literal and literal.startswith("/"):
                urls.add(literal.split("?", 1)[0])
    return urls


def test_every_management_route_has_a_behavior_scenario():
    routes = _source_routes()
    assert routes, "no @router routes parsed — the decorator regex is stale"

    urls = _test_urls()
    uncovered = sorted(
        route
        for route in routes
        if not any(_route_to_regex(route).match(url) for url in urls)
    )
    assert (
        not uncovered
    ), "management routes with no behavior-suite scenario:\n  " + "\n  ".join(uncovered)
