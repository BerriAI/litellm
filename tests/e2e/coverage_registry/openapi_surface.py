"""OpenAPI-backed e2e coverage by endpoint, provider, and request parameter.

The denominator is generated from the proxy OpenAPI schema. LLM endpoints expand
each endpoint/param pair by the provider families LiteLLM supports for that
endpoint; non-LLM proxy routes use provider="proxy". The numerator is collected
from pytest markers without running live tests:

    @pytest.mark.e2e_coverage(
        endpoint="/chat/completions",
        providers=("openai", "anthropic"),
        params=("model", "messages", "stream"),
    )

This intentionally rejects wildcards and "multiple" labels so uncovered rows can
name a concrete route, provider, and real OpenAPI parameter.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, cast

import pytest

E2E_DIR = Path(__file__).resolve().parent.parent

JsonMap = Mapping[str, object]

HTTP_METHODS: frozenset[str] = frozenset(
    {"get", "post", "put", "patch", "delete", "options", "head"}
)

DISPLAY_MODULES: Mapping[str, str] = {
    "core_llms": "Core LLMs",
    "non_core_llms": "Non-Core LLMs",
    "budgets": "Budgets",
    "management": "Management/UI",
    "spend_tracking": "Spend Tracking",
    "mcp": "MCPs",
    "logging": "Logging & Guardrails",
    "guardrails": "Logging & Guardrails",
    "other": "Other",
}

MODULE_ORDER: tuple[str, ...] = (
    "core_llms",
    "non_core_llms",
    "budgets",
    "management",
    "spend_tracking",
    "mcp",
    "logging",
    "other",
)

CORE_LLM_FAMILIES: frozenset[str] = frozenset(
    {"chat_completions", "messages", "responses"}
)

NON_CORE_LLM_FAMILIES: frozenset[str] = frozenset(
    {"embeddings", "rerank", "images", "audio", "batch", "files", "realtime"}
)

LLM_PROVIDER_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "chat_completions": (
        "openai",
        "azure",
        "anthropic",
        "bedrock",
        "vertex_ai",
        "gemini",
        "cohere",
        "together_ai",
        "deepseek",
        "xai",
    ),
    "messages": ("anthropic", "bedrock", "vertex_ai"),
    "responses": ("openai", "azure", "anthropic", "bedrock", "vertex_ai"),
    "embeddings": ("openai", "azure", "cohere", "vertex_ai", "gemini", "bedrock"),
    "rerank": ("cohere", "bedrock", "vertex_ai", "jina_ai", "voyage"),
    "images": (
        "openai",
        "azure",
        "bedrock",
        "vertex_ai",
        "gemini",
        "stability",
    ),
    "audio": ("openai", "azure"),
    "batch": ("openai", "azure", "bedrock", "vertex_ai"),
    "files": ("openai", "azure", "bedrock", "vertex_ai"),
    "realtime": ("openai", "azure", "gemini", "vertex_ai", "xai"),
}

REQUEST_BODY_EXPECTED_FAMILIES: frozenset[str] = frozenset(
    LLM_PROVIDER_FAMILIES.keys() - {"realtime"}
)


@dataclass(frozen=True, order=True, slots=True)
class Surface:
    module: str
    endpoint: str
    method: str
    provider: str
    param: str


@dataclass(frozen=True, order=True, slots=True)
class SchemaGap:
    module: str
    endpoint: str
    method: str
    reason: str


@dataclass(frozen=True, slots=True)
class MarkerEntry:
    nodeid: str
    endpoint: str
    method: str
    providers: tuple[str, ...]
    params: tuple[str, ...]


@dataclass(frozen=True, order=True, slots=True)
class InvalidMarker:
    nodeid: str
    reason: str


@dataclass(frozen=True, slots=True)
class OpenAPISurface:
    surfaces: frozenset[Surface]
    schema_gaps: tuple[SchemaGap, ...]


@dataclass(frozen=True, slots=True)
class ModuleCoverage:
    module: str
    covered: int
    total: int
    tests: int

    @property
    def coverage_percent(self) -> float:
        return percent(self.covered, self.total)


@dataclass(frozen=True, slots=True)
class SurfaceCoverageReport:
    modules: tuple[ModuleCoverage, ...]
    covered: int
    total: int
    tests: int
    uncovered: tuple[Surface, ...]
    schema_gaps: tuple[SchemaGap, ...]
    invalid_markers: tuple[InvalidMarker, ...]
    collection_errors: tuple[str, ...]

    @property
    def coverage_percent(self) -> float:
        return percent(self.covered, self.total)


class _CoverageSink:
    def __init__(self) -> None:
        self.markers: tuple[MarkerEntry, ...] = ()
        self.invalid_markers: tuple[InvalidMarker, ...] = ()
        self.collection_errors: tuple[str, ...] = ()

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        entries: list[MarkerEntry] = []
        invalid: list[InvalidMarker] = []
        for item in session.items:
            for marker in item.iter_markers(name="e2e_coverage"):
                parsed, error = marker_entry_from_pytest_marker(item.nodeid, marker)
                if error is not None:
                    invalid.append(InvalidMarker(nodeid=item.nodeid, reason=error))
                elif parsed is not None:
                    entries.append(parsed)
        self.markers = tuple(entries)
        self.invalid_markers = tuple(invalid)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.collection_errors = (*self.collection_errors, report.nodeid)


def percent(covered: int, total: int) -> float:
    return (100.0 * covered / total) if total else 0.0


def marker_entry_from_pytest_marker(
    nodeid: str, marker: pytest.Mark
) -> tuple[MarkerEntry | None, str | None]:
    endpoint = marker.kwargs.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint:
        return None, "e2e_coverage marker requires endpoint='...'"
    if "*" in endpoint:
        return None, f"endpoint must be concrete, got {endpoint!r}"

    method = marker.kwargs.get("method", "post")
    if not isinstance(method, str):
        return None, "method must be a string"

    providers = normalize_tuple(
        marker.kwargs.get("providers", marker.kwargs.get("provider"))
    )
    if not providers:
        return None, "e2e_coverage marker requires provider='...' or providers=(...)"
    if any(provider == "multiple" or "*" in provider for provider in providers):
        return None, f"providers must be concrete, got {providers!r}"

    params = normalize_tuple(marker.kwargs.get("params"))
    if not params:
        return None, "e2e_coverage marker requires params=(...)"
    if any(param == "multiple" or "*" in param for param in params):
        return None, f"params must be concrete, got {params!r}"

    return (
        MarkerEntry(
            nodeid=nodeid,
            endpoint=endpoint,
            method=method.upper(),
            providers=providers,
            params=params,
        ),
        None,
    )


def normalize_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        normalized: list[str] = []
        for item in value:
            if isinstance(item, str) and item:
                normalized.append(item)
        return tuple(normalized)
    return ()


def collect_markers(
    e2e_dir: Path = E2E_DIR,
) -> tuple[tuple[MarkerEntry, ...], tuple[InvalidMarker, ...], tuple[str, ...]]:
    sink = _CoverageSink()
    with contextlib.redirect_stdout(io.StringIO()):
        pytest.main(
            [
                "--collect-only",
                "-qq",
                "--continue-on-collection-errors",
                "-p",
                "no:cacheprovider",
                str(e2e_dir),
            ],
            plugins=[sink],
        )
    return sink.markers, sink.invalid_markers, sink.collection_errors


def load_proxy_openapi_schema() -> JsonMap:
    from litellm.proxy.proxy_server import app

    app.openapi_schema = None
    return cast(JsonMap, app.openapi())


def load_openapi_from_file(path: Path) -> JsonMap:
    return cast(JsonMap, json.loads(path.read_text()))


def build_openapi_surface(openapi_schema: JsonMap) -> OpenAPISurface:
    paths = as_mapping(openapi_schema.get("paths"))
    surfaces: set[Surface] = set()
    schema_gaps: set[SchemaGap] = set()

    for endpoint, raw_path_item in paths.items():
        if not isinstance(endpoint, str):
            continue
        path_item = as_mapping(raw_path_item)
        for method_lower, raw_operation in path_item.items():
            if method_lower not in HTTP_METHODS:
                continue
            operation = as_mapping(raw_operation)
            family = route_family_for_operation(operation)
            module = module_for_route(endpoint, family)
            providers = providers_for_route(endpoint, family)
            method = method_lower.upper()
            request_params = params_for_operation(openapi_schema, operation)

            if (
                family in REQUEST_BODY_EXPECTED_FAMILIES
                and method == "POST"
                and not has_json_request_body(operation)
            ):
                schema_gaps.add(
                    SchemaGap(
                        module=module,
                        endpoint=endpoint,
                        method=method,
                        reason="missing JSON requestBody in OpenAPI",
                    )
                )

            if not request_params:
                if family in LLM_PROVIDER_FAMILIES and method == "POST":
                    schema_gaps.add(
                        SchemaGap(
                            module=module,
                            endpoint=endpoint,
                            method=method,
                            reason="no request params exposed in OpenAPI",
                        )
                    )
                continue

            for provider in providers:
                for param in request_params:
                    surfaces.add(
                        Surface(
                            module=module,
                            endpoint=endpoint,
                            method=method,
                            provider=provider,
                            param=param,
                        )
                    )

    return OpenAPISurface(
        surfaces=frozenset(surfaces), schema_gaps=tuple(sorted(schema_gaps))
    )


def filter_openapi_surface(
    openapi_surface: OpenAPISurface, *, modules: frozenset[str] | None
) -> OpenAPISurface:
    if modules is None:
        return openapi_surface
    return OpenAPISurface(
        surfaces=frozenset(
            surface for surface in openapi_surface.surfaces if surface.module in modules
        ),
        schema_gaps=tuple(
            gap for gap in openapi_surface.schema_gaps if gap.module in modules
        ),
    )


def route_family_for_operation(operation: JsonMap) -> str | None:
    tags = tuple(tag for tag in operation.get("tags", ()) if isinstance(tag, str))
    operation_id = operation.get("operationId")
    operation_id = operation_id if isinstance(operation_id, str) else ""

    if "chat/completions" in tags:
        return "chat_completions"
    if operation_id.startswith("anthropic_response_"):
        return "messages"
    if operation_id.startswith("responses_api_"):
        return "responses"
    if "embeddings" in tags:
        return "embeddings"
    if "rerank" in tags:
        return "rerank"
    if "images" in tags:
        return "images"
    if "audio" in tags:
        return "audio"
    if "batch" in tags:
        return "batch"
    if "files" in tags:
        return "files"
    if "realtime" in operation_id:
        return "realtime"
    return None


def providers_for_route(endpoint: str, family: str | None) -> tuple[str, ...]:
    if family not in LLM_PROVIDER_FAMILIES:
        return ("proxy",)
    if endpoint.startswith("/openai/deployments/"):
        return ("azure",)
    if endpoint.startswith("/engines/"):
        return ("openai",)
    if endpoint.startswith("/openai/"):
        return ("openai",)
    return LLM_PROVIDER_FAMILIES[family]


def module_for_route(endpoint: str, family: str | None) -> str:
    if family in CORE_LLM_FAMILIES:
        return "core_llms"
    if family in NON_CORE_LLM_FAMILIES:
        return "non_core_llms"
    if endpoint.startswith(("/budget", "/budgets")):
        return "budgets"
    if endpoint.startswith(("/spend", "/global/spend", "/daily", "/provider")):
        return "spend_tracking"
    if endpoint.startswith(("/mcp", "/v1/mcp")):
        return "mcp"
    if endpoint.startswith(("/callback", "/guardrail", "/guardrails", "/logging")):
        return "logging"
    if endpoint.startswith(
        (
            "/key",
            "/keys",
            "/team",
            "/user",
            "/organization",
            "/customer",
            "/model",
            "/model_group",
            "/config",
            "/settings",
            "/tag",
            "/credential",
            "/credentials",
            "/cache",
            "/health",
        )
    ):
        return "management"
    return "other"


def params_for_operation(
    openapi_schema: JsonMap, operation: JsonMap
) -> tuple[str, ...]:
    params: set[str] = set()
    raw_parameters = operation.get("parameters")
    if isinstance(raw_parameters, Sequence) and not isinstance(raw_parameters, str):
        for raw_parameter in raw_parameters:
            parameter = resolve_ref(openapi_schema, raw_parameter)
            if parameter.get("in") in {"path", "query"}:
                name = parameter.get("name")
                if isinstance(name, str) and name:
                    params.add(name)

    for schema in request_body_schemas(operation):
        params.update(top_level_schema_properties(openapi_schema, schema))

    return tuple(sorted(params))


def request_body_schemas(operation: JsonMap) -> tuple[JsonMap, ...]:
    request_body = as_mapping(operation.get("requestBody"))
    content = as_mapping(request_body.get("content"))
    schemas: list[JsonMap] = []
    for content_type, raw_media_type in content.items():
        if not isinstance(content_type, str):
            continue
        if content_type != "application/json" and not content_type.startswith(
            "multipart/"
        ):
            continue
        media_type = as_mapping(raw_media_type)
        schema = as_mapping(media_type.get("schema"))
        if schema:
            schemas.append(schema)
    return tuple(schemas)


def has_json_request_body(operation: JsonMap) -> bool:
    request_body = as_mapping(operation.get("requestBody"))
    content = as_mapping(request_body.get("content"))
    return "application/json" in content


def top_level_schema_properties(openapi_schema: JsonMap, schema: JsonMap) -> set[str]:
    resolved = resolve_ref(openapi_schema, schema)
    properties = as_mapping(resolved.get("properties"))
    names = {name for name in properties if isinstance(name, str)}
    for combiner in ("allOf", "anyOf", "oneOf"):
        raw_options = resolved.get(combiner)
        if not isinstance(raw_options, Sequence) or isinstance(raw_options, str):
            continue
        for raw_option in raw_options:
            option = as_mapping(raw_option)
            if option:
                names.update(top_level_schema_properties(openapi_schema, option))
    return names


def resolve_ref(openapi_schema: JsonMap, value: object) -> JsonMap:
    data = as_mapping(value)
    ref = data.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return data
    current: object = openapi_schema
    for part in ref[2:].split("/"):
        if not isinstance(current, Mapping):
            return {}
        current = current.get(part)
    return as_mapping(current)


def as_mapping(value: object) -> JsonMap:
    if isinstance(value, Mapping):
        return cast(JsonMap, value)
    return {}


def compute_surface_coverage(
    openapi_surface: OpenAPISurface,
    marker_entries: tuple[MarkerEntry, ...],
    marker_errors: tuple[InvalidMarker, ...] = (),
    collection_errors: tuple[str, ...] = (),
) -> SurfaceCoverageReport:
    denominator = openapi_surface.surfaces
    denominator_by_key = {
        (surface.endpoint, surface.method, surface.provider, surface.param): surface
        for surface in denominator
    }
    covered: set[Surface] = set()
    invalid: list[InvalidMarker] = list(marker_errors)
    tests_by_module: dict[str, set[str]] = {module: set() for module in MODULE_ORDER}

    for marker in marker_entries:
        for provider in marker.providers:
            for param in marker.params:
                surface = denominator_by_key.get(
                    (marker.endpoint, marker.method, provider, param)
                )
                if surface is None:
                    invalid.append(
                        InvalidMarker(
                            nodeid=marker.nodeid,
                            reason=(
                                "marker does not match OpenAPI surface: "
                                f"{marker.method} {marker.endpoint} "
                                f"provider={provider} param={param}"
                            ),
                        )
                    )
                    continue
                covered.add(surface)
                tests_by_module[surface.module].add(marker.nodeid)

    modules = tuple(
        module_coverage
        for module in MODULE_ORDER
        for module_coverage in (
            ModuleCoverage(
                module=module,
                covered=sum(1 for surface in covered if surface.module == module),
                total=sum(1 for surface in denominator if surface.module == module),
                tests=len(tests_by_module[module]),
            ),
        )
        if module_coverage.total or module_coverage.tests
    )
    return SurfaceCoverageReport(
        modules=modules,
        covered=len(covered),
        total=len(denominator),
        tests=len(
            {
                nodeid
                for module_tests in tests_by_module.values()
                for nodeid in module_tests
            }
        ),
        uncovered=tuple(sorted(denominator - covered)),
        schema_gaps=openapi_surface.schema_gaps,
        invalid_markers=tuple(sorted(invalid)),
        collection_errors=collection_errors,
    )


def render_text(report: SurfaceCoverageReport, *, max_uncovered: int = 80) -> str:
    lines = [
        f"{'MODULE':24}{'COVERED':>12}{'COVERAGE':>12}{'TESTS':>8}",
    ]
    for module in report.modules:
        lines.append(
            f"{DISPLAY_MODULES[module.module]:24}"
            f"{module.covered:>5}/{module.total:<6}"
            f"{module.coverage_percent:>11.1f}%"
            f"{module.tests:>8}"
        )
    lines.extend(
        [
            "-" * 56,
            f"{'ALL':24}{report.covered:>5}/{report.total:<6}"
            f"{report.coverage_percent:>11.1f}%{report.tests:>8}",
            "",
            "Coverage formula: covered OpenAPI endpoint/provider/param surfaces "
            "/ total OpenAPI endpoint/provider/param surfaces.",
        ]
    )

    if report.uncovered:
        lines.extend(
            [
                "",
                f"Uncovered endpoint/provider/param surfaces "
                f"(showing {min(max_uncovered, len(report.uncovered))} of "
                f"{len(report.uncovered)}):",
                f"{'MODULE':16}{'METHOD':8}{'ENDPOINT':52}{'PROVIDER':14}PARAM",
            ]
        )
        for surface in report.uncovered[:max_uncovered]:
            lines.append(
                f"{surface.module:16}{surface.method:8}"
                f"{surface.endpoint[:51]:52}{surface.provider:14}{surface.param}"
            )

    if report.schema_gaps:
        lines.extend(
            [
                "",
                f"OpenAPI schema gaps ({len(report.schema_gaps)}):",
                f"{'MODULE':16}{'METHOD':8}{'ENDPOINT':52}REASON",
            ]
        )
        for gap in report.schema_gaps:
            lines.append(
                f"{gap.module:16}{gap.method:8}{gap.endpoint[:51]:52}{gap.reason}"
            )

    if report.invalid_markers:
        lines.extend(
            [
                "",
                f"Invalid coverage markers ({len(report.invalid_markers)}):",
            ]
        )
        lines.extend(f"  {m.nodeid}: {m.reason}" for m in report.invalid_markers)

    if report.collection_errors:
        lines.extend(
            [
                "",
                f"Collection errors ({len(report.collection_errors)}):",
            ]
        )
        lines.extend(f"  {nodeid}" for nodeid in report.collection_errors)

    return "\n".join(lines)


def report_to_dict(report: SurfaceCoverageReport) -> dict[str, object]:
    return {
        "covered": report.covered,
        "total": report.total,
        "coverage_percent": report.coverage_percent,
        "tests": report.tests,
        "modules": [
            {
                "module": module.module,
                "display_module": DISPLAY_MODULES[module.module],
                "covered": module.covered,
                "total": module.total,
                "coverage_percent": module.coverage_percent,
                "tests": module.tests,
            }
            for module in report.modules
        ],
        "uncovered": [
            {
                "module": surface.module,
                "endpoint": surface.endpoint,
                "method": surface.method,
                "provider": surface.provider,
                "param": surface.param,
            }
            for surface in report.uncovered
        ],
        "schema_gaps": [
            {
                "module": gap.module,
                "endpoint": gap.endpoint,
                "method": gap.method,
                "reason": gap.reason,
            }
            for gap in report.schema_gaps
        ],
        "invalid_markers": [
            {"nodeid": marker.nodeid, "reason": marker.reason}
            for marker in report.invalid_markers
        ],
        "collection_errors": list(report.collection_errors),
    }


def render_json(report: SurfaceCoverageReport) -> str:
    return json.dumps(report_to_dict(report), indent=2, sort_keys=True)


def render_loki(report: SurfaceCoverageReport) -> str:
    lines = [
        (
            f"OPENAPI_COVERAGE_TOTAL percent={report.coverage_percent:.1f} "
            f"covered={report.covered} total={report.total} tests={report.tests}"
        )
    ]
    lines.extend(
        (
            f"OPENAPI_COVERAGE_MODULE module={module.module} "
            f"percent={module.coverage_percent:.1f} "
            f"covered={module.covered} total={module.total} tests={module.tests}"
        )
        for module in report.modules
    )
    lines.extend(
        (
            f"OPENAPI_COVERAGE_UNCOVERED module={surface.module} method={surface.method} "
            f"endpoint={surface.endpoint} provider={surface.provider} "
            f"param={surface.param}"
        )
        for surface in report.uncovered
    )
    lines.extend(
        (
            f"OPENAPI_COVERAGE_SCHEMA_GAP module={gap.module} method={gap.method} "
            f"endpoint={gap.endpoint} reason={quote_loki(gap.reason)}"
        )
        for gap in report.schema_gaps
    )
    return "\n".join(lines)


def quote_loki(value: str) -> str:
    return json.dumps(value, separators=(",", ":"))


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--openapi-json", type=Path)
    parser.add_argument(
        "--format",
        choices=("text", "json", "loki"),
        default="text",
    )
    parser.add_argument(
        "--scope",
        choices=("all", "llm"),
        default="all",
        help="Use 'llm' for the Core/Non-Core LLM dashboard view.",
    )
    parser.add_argument("--max-uncovered", type=int, default=80)
    parser.add_argument("--fail-on-invalid-markers", action="store_true")
    parser.add_argument("--fail-on-collection-errors", action="store_true")
    args = parser.parse_args()

    openapi_schema = (
        load_openapi_from_file(args.openapi_json)
        if args.openapi_json
        else load_proxy_openapi_schema()
    )
    openapi_surface = build_openapi_surface(openapi_schema)
    openapi_surface = filter_openapi_surface(
        openapi_surface,
        modules=(
            frozenset({"core_llms", "non_core_llms"}) if args.scope == "llm" else None
        ),
    )
    markers, marker_errors, collection_errors = collect_markers()
    report = compute_surface_coverage(
        openapi_surface,
        markers,
        marker_errors=marker_errors,
        collection_errors=collection_errors,
    )

    if args.format == "json":
        print(render_json(report))  # noqa: T201
    elif args.format == "loki":
        print(render_loki(report))  # noqa: T201
    else:
        print(render_text(report, max_uncovered=args.max_uncovered))  # noqa: T201

    if args.fail_on_invalid_markers and report.invalid_markers:
        return 1
    if args.fail_on_collection_errors and report.collection_errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
