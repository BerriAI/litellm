import re
from pathlib import Path
from typing import Final

import pytest

from backend.routes.allowlist import BACKEND_EXACT_PATHS, BACKEND_PATH_PREFIXES
from gateway.routes.allowlist import GATEWAY_EXACT_PATHS, GATEWAY_PATH_PREFIXES

_INGRESS_TEMPLATE: Final = Path(__file__).parents[4] / "helm" / "litellm" / "templates" / "ingress.yaml"

# A route template whose first segment is any provider name (/{provider}/v1/files). An Ingress Prefix cannot
# express it, so those requests reach the backend unless the provider has its own prefix above.
_PREFIXES_AN_INGRESS_CANNOT_ROUTE: Final = frozenset({"/{provider}/"})


def _template_list(name: str) -> frozenset[str]:
    match: Final = re.search(rf"\${name} := list(.*?)-}}}}", _INGRESS_TEMPLATE.read_text(), re.S)
    assert match is not None, f"${name} not found in {_INGRESS_TEMPLATE}"
    return frozenset(re.findall(r'"(/[^"]*)"', match.group(1)))


_CHART_PREFIXES: Final = _template_list("gatewayPrefixes")
_CHART_EXACT_PATHS: Final = _template_list("gatewayExactPaths")


def _backend_keeps(path: str) -> bool:
    return path in BACKEND_EXACT_PATHS or any(path.startswith(prefix) for prefix in BACKEND_PATH_PREFIXES)


_GATEWAY_ONLY_PREFIXES: Final = sorted(
    frozenset(prefix for prefix in GATEWAY_PATH_PREFIXES if not _backend_keeps(prefix))
    - _PREFIXES_AN_INGRESS_CANNOT_ROUTE
)
_GATEWAY_ONLY_EXACT_PATHS: Final = sorted(GATEWAY_EXACT_PATHS - BACKEND_EXACT_PATHS)


def _routed_to_gateway_by_chart_prefix(path: str) -> bool:
    trimmed: Final = path.rstrip("/")
    return any(trimmed == prefix or trimmed.startswith(f"{prefix}/") for prefix in _CHART_PREFIXES)


def _gateway_serves_something_under(chart_prefix: str) -> bool:
    under: Final = f"{chart_prefix}/"
    by_prefix: Final = any(
        chart_prefix.startswith(prefix) or prefix.rstrip("/") == chart_prefix or prefix.startswith(under)
        for prefix in GATEWAY_PATH_PREFIXES
    )
    return by_prefix or any(path == chart_prefix or path.startswith(under) for path in GATEWAY_EXACT_PATHS)


@pytest.mark.parametrize("prefix", _GATEWAY_ONLY_PREFIXES)
def test_ingress_sends_every_gateway_only_prefix_to_the_gateway(prefix: str) -> None:
    assert _routed_to_gateway_by_chart_prefix(prefix)


@pytest.mark.parametrize("path", _GATEWAY_ONLY_EXACT_PATHS)
def test_ingress_sends_every_gateway_only_exact_path_to_the_gateway(path: str) -> None:
    assert path in _CHART_EXACT_PATHS or _routed_to_gateway_by_chart_prefix(path)


@pytest.mark.parametrize("chart_prefix", sorted(_CHART_PREFIXES))
def test_ingress_sends_no_prefix_to_the_gateway_that_the_gateway_dropped(chart_prefix: str) -> None:
    assert _gateway_serves_something_under(chart_prefix)


@pytest.mark.parametrize("chart_path", sorted(_CHART_EXACT_PATHS))
def test_ingress_sends_no_exact_path_to_the_gateway_that_the_gateway_dropped(chart_path: str) -> None:
    assert chart_path in GATEWAY_EXACT_PATHS
