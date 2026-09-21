from __future__ import annotations

import importlib
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, TypeAlias

from pydantic import BaseModel, ConfigDict

from ...shared.tracing.pytest_usage import (
    PythonFunctionIdentity,
    RustFunctionIdentity,
    candidate_test_files,
    collect_python_function_tests,
)
from ...shared.tracing.steps import pipeline_projection
from ...shared.unit_runners.python_runner import collect_python_tests, contract_nodeid
from ...shared.unit_runners.rust_runner import RustTarget, RustTestIdentity, RustTestScope, enumerate_rust_tests
from .contracts import PythonFunctionDiscoverySpec, RustTestFamily, TestMapping, UnitTestContract

PythonInventory: TypeAlias = Callable[[Sequence[str], Path], frozenset[str]]
RustInventory: TypeAlias = Callable[[Path, tuple[RustTestScope, ...]], frozenset[RustTestIdentity]]


def _trace_functions(
    spec: PythonFunctionDiscoverySpec,
) -> tuple[tuple[PythonFunctionIdentity, ...], tuple[RustFunctionIdentity, ...]]:
    from ..trace_parity.models import RouteSpec, TraceExecutionFailure, TraceSuite
    from ..trace_parity.sdk.execution import collect_trace

    if spec.trace_module is None:
        return ()
    module: Final = importlib.import_module(spec.trace_module)
    suite: Final = getattr(module, "TRACE_SUITE", None)
    if not isinstance(suite, TraceSuite) or not isinstance(suite.route, RouteSpec):
        raise ValueError(f"{spec.trace_module} must export an SDK TRACE_SUITE")
    python_functions: Final[dict[str, PythonFunctionIdentity]] = {}
    rust_functions: Final[dict[str, RustFunctionIdentity]] = {}
    for scenario in suite.scenarios:
        for mode in scenario.modes:
            route: Final = RouteSpec(
                route=suite.route.route,
                python_entrypoints=suite.route.python_entrypoints,
                rust_entrypoints=suite.route.rust_entrypoints,
                fixture=scenario.fixture,
            )
            python_trace: Final = collect_trace(route, "python", asynchronous=mode == "async")
            rust_trace: Final = collect_trace(route, "rust", asynchronous=mode == "async")
            if isinstance(python_trace, TraceExecutionFailure):
                raise ValueError(f"Python trace discovery failed for {scenario.name}/{mode}: {python_trace.message}")
            if isinstance(rust_trace, TraceExecutionFailure):
                raise ValueError(f"Rust trace discovery failed for {scenario.name}/{mode}: {rust_trace.message}")
            mappings: Final = scenario.mappings_for(mode)
            python_projection: Final = pipeline_projection("python", python_trace, mappings)
            rust_projection: Final = pipeline_projection("rust", rust_trace, mappings)
            for step in python_projection.steps:
                if step.span in spec.trace_spans:
                    function: Final = PythonFunctionIdentity.from_trace(step.raw)
                    python_functions[function.raw] = function
            for step in rust_projection.steps:
                if step.span in spec.trace_spans:
                    function: Final = RustFunctionIdentity.from_trace(step.raw)
                    rust_functions[step.raw] = function
    if not python_functions or not rust_functions:
        raise ValueError(f"Python trace discovery found no functions for spans: {', '.join(spec.trace_spans)}")
    return (
        tuple(python_functions[key] for key in sorted(python_functions)),
        tuple(rust_functions[key] for key in sorted(rust_functions)),
    )


def collect_python_function_inventory(
    spec: PythonFunctionDiscoverySpec,
    repo_root: Path,
    traced_functions: Sequence[PythonFunctionIdentity] = (),
) -> frozenset[str]:
    source_root: Final = repo_root / "litellm"
    functions: Final = (
        tuple(reference.resolve(source_root) for reference in spec.functions)
        if spec.functions
        else tuple(traced_functions)
    )
    discovered: Final = candidate_test_files(
        functions,
        spec.search_roots,
        repo_root,
        exclude_roots=spec.exclude_roots,
    )
    selectors: Final = tuple(dict.fromkeys((*discovered, *spec.includes)))
    if not selectors:
        raise ValueError("Python function discovery found no candidate test files")
    report: Final = collect_python_function_tests(
        functions,
        selectors,
        repo_root,
        source_root=source_root,
        exclusions=spec.exclusions,
    )
    if report.exit_code or report.problems:
        details: Final = "\n".join(report.problems) or f"pytest exited with code {report.exit_code}"
        raise ValueError(f"Python function test discovery failed:\n{details}")
    return frozenset(contract_nodeid(nodeid) for usage in report.usages for nodeid in usage.tests)


def _colocated_rust_scope(mappings: Sequence[TestMapping]) -> tuple[RustTestScope, ...]:
    modules_by_target: Final[dict[str, set[str]]] = defaultdict(set)
    targets: Final[dict[str, RustTarget]] = {}
    for item in mappings:
        module, separator, _ = item.rust.name.partition("::tests::")
        if not separator:
            raise ValueError(f"Rust unit test is not colocated in a tests module: {item.rust.key}")
        target_key: Final = item.rust.target.key
        targets[target_key] = item.rust.target
        modules_by_target[target_key].add(f"{module}::tests")
    return tuple(
        RustTestScope(
            target=targets[target_key],
            modules=tuple(sorted(modules_by_target[target_key])),
        )
        for target_key in sorted(targets)
    )


def _traced_rust_scope(
    functions: Sequence[RustFunctionIdentity],
    targets: Sequence[RustTarget],
    repo_root: Path,
) -> tuple[RustTestScope, ...]:
    targets_by_name: Final = {target.name: target for target in targets}
    modules_by_target: Final[dict[str, set[str]]] = defaultdict(set)
    for function in functions:
        crate: Final = function.module_path.partition("::")[0]
        target: Final = targets_by_name.get(crate)
        if target is None:
            continue
        source_candidates: Final = (
            repo_root / "litellm-rust" / function.file,
            repo_root / function.file,
        )
        source: Final = next((path for path in source_candidates if path.is_file()), None)
        if source is None:
            raise ValueError(f"Traced Rust source does not exist: {function.file}")
        contents: Final = source.read_text()
        if "mod tests" in contents and "#[cfg(test)]" in contents:
            modules_by_target[target.key].add(function.test_module)
    selected_targets: Final = {target.key: target for target in targets}
    scopes: Final = tuple(
        RustTestScope(target=selected_targets[key], modules=tuple(sorted(modules)))
        for key, modules in sorted(modules_by_target.items())
        if modules
    )
    if not scopes:
        raise ValueError("Traced Rust functions have no colocated test modules")
    return scopes


def _merge_rust_scopes(scopes: Sequence[RustTestScope]) -> tuple[RustTestScope, ...]:
    targets: Final = {scope.target.key: scope.target for scope in scopes}
    modules: Final[dict[str, set[str]]] = defaultdict(set)
    features: Final[dict[str, set[str]]] = defaultdict(set)
    default_features: Final[dict[str, bool]] = {}
    for scope in scopes:
        modules[scope.target.key].update(scope.modules)
        features[scope.target.key].update(scope.features)
        default_features[scope.target.key] = default_features.get(scope.target.key, True) and scope.default_features
    return tuple(
        RustTestScope(
            target=targets[key],
            modules=tuple(
                sorted(
                    module
                    for module in modules[key]
                    if not any(module.startswith(f"{parent}::") for parent in modules[key])
                )
            ),
            features=tuple(sorted(features[key])),
            default_features=default_features[key],
        )
        for key in sorted(targets)
    )


def _owned_rust_tests(
    rust: RustTestIdentity | RustTestFamily,
    inventory: frozenset[RustTestIdentity],
) -> frozenset[RustTestIdentity]:
    if isinstance(rust, RustTestFamily):
        return frozenset(identity for identity in inventory if rust.contains(identity))
    return frozenset((rust,)) if rust in inventory else frozenset()


class MappingReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    python_tests: tuple[str, ...]
    rust_tests: tuple[str, ...]
    mapped_python_tests: tuple[str, ...]
    excluded_python_tests: tuple[str, ...]
    unmapped_python_tests: tuple[str, ...]
    rust_only_tests: tuple[str, ...]
    missing_python_tests: tuple[str, ...]
    missing_rust_tests: tuple[str, ...]
    duplicate_python_mappings: tuple[str, ...]
    duplicate_rust_mappings: tuple[str, ...]
    invalid_mapping_exclusions: tuple[str, ...]
    mapped_and_excluded_python_tests: tuple[str, ...]
    invalid_unit_parity_exclusions: tuple[str, ...]

    @property
    def mapped_count(self) -> int:
        return len(self.mapped_python_tests)

    @property
    def total_count(self) -> int:
        return len(self.python_tests)

    @property
    def percentage(self) -> float:
        return 0.0 if not self.total_count else round(100.0 * self.mapped_count / self.total_count, 1)

    @property
    def is_valid(self) -> bool:
        return not (
            self.missing_python_tests
            or self.missing_rust_tests
            or self.duplicate_python_mappings
            or self.duplicate_rust_mappings
            or self.invalid_mapping_exclusions
            or self.mapped_and_excluded_python_tests
            or self.invalid_unit_parity_exclusions
        )


def audit_mapping(
    contract: UnitTestContract,
    repo_root: Path,
    *,
    python_inventory: PythonInventory = collect_python_tests,
    rust_inventory: RustInventory = enumerate_rust_tests,
) -> MappingReport:
    mapping: Final = contract.mapping
    traced_python: tuple[PythonFunctionIdentity, ...] = ()
    traced_rust: tuple[RustFunctionIdentity, ...] = ()
    if mapping.python_functions is not None and mapping.python_functions.trace_module is not None:
        traced_python, traced_rust = _trace_functions(mapping.python_functions)
    python_tests: Final = (
        collect_python_function_inventory(mapping.python_functions, repo_root, traced_python)
        if mapping.python_functions is not None
        else python_inventory(mapping.python_selectors, repo_root)
    )
    unit_parity_tests: Final = python_inventory(contract.unit_parity.python_selectors, repo_root)
    traced_scope: Final = _traced_rust_scope(traced_rust, mapping.rust_targets, repo_root) if traced_rust else ()
    rust_scope: Final = _merge_rust_scopes(
        (*mapping.rust_scope, *traced_scope, *_colocated_rust_scope(mapping.mappings))
    )
    rust_tests: Final = rust_inventory(repo_root, rust_scope)
    mapped_python: Final = frozenset(item.python for item in mapping.mappings)
    excluded_python: Final = frozenset(exclusion.nodeid for exclusion in mapping.exclusions)
    rust_ownership: Final = tuple((item.rust, _owned_rust_tests(item.rust, rust_tests)) for item in mapping.mappings)
    mapped_rust: Final = frozenset(identity for _, identities in rust_ownership for identity in identities)
    duplicate_python: Final = tuple(
        sorted(nodeid for nodeid, count in Counter(item.python for item in mapping.mappings).items() if count > 1)
    )
    duplicate_exact_rust: Final = frozenset(
        identity.key
        for identity, count in Counter(
            item.rust for item in mapping.mappings if isinstance(item.rust, RustTestIdentity)
        ).items()
        if count > 1
    )
    duplicate_owned_rust: Final = frozenset(
        identity.key
        for identity, count in Counter(identity for _, identities in rust_ownership for identity in identities).items()
        if count > 1
    )
    duplicate_rust: Final = tuple(sorted(duplicate_exact_rust | duplicate_owned_rust))
    return MappingReport(
        python_tests=tuple(sorted(python_tests)),
        rust_tests=tuple(sorted(identity.key for identity in rust_tests)),
        mapped_python_tests=tuple(sorted(python_tests & mapped_python)),
        excluded_python_tests=tuple(sorted((python_tests & excluded_python) - mapped_python)),
        unmapped_python_tests=tuple(sorted(python_tests - mapped_python - excluded_python)),
        rust_only_tests=tuple(sorted(identity.key for identity in rust_tests - mapped_rust)),
        missing_python_tests=tuple(sorted(mapped_python - python_tests)),
        missing_rust_tests=tuple(sorted(rust.key for rust, identities in rust_ownership if not identities)),
        duplicate_python_mappings=duplicate_python,
        duplicate_rust_mappings=duplicate_rust,
        invalid_mapping_exclusions=tuple(sorted(excluded_python - python_tests)),
        mapped_and_excluded_python_tests=tuple(sorted(mapped_python & excluded_python)),
        invalid_unit_parity_exclusions=tuple(
            sorted(
                exclusion.nodeid
                for exclusion in contract.unit_parity.exclusions
                if exclusion.nodeid not in unit_parity_tests
            )
        ),
    )
