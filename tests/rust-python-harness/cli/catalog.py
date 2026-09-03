from __future__ import annotations

import hashlib
import importlib
import importlib.util
import pkgutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

from .. import strategies as _strategies_package
from ..shared.reporting.models import SDK_FUNCTIONS, SURFACES, CaseDisposition, HarnessCase, Strategy
from ..shared.reporting.strategy import StrategyDefinition

_STRATEGIES_PACKAGE: Final = _strategies_package
STRATEGIES_ROOT: Final = Path(_STRATEGIES_PACKAGE.__path__[0])


def _load_strategy_module(name: str, folder: Path, prefix: str | None) -> ModuleType:
    if prefix is not None:
        return importlib.import_module(f"{prefix}.{name}")
    module_name: Final = _synthetic_module_name(folder)
    spec: Final = importlib.util.spec_from_file_location(
        module_name, folder / "__init__.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"{folder}: cannot load strategy package")
    module: Final = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        del sys.modules[module_name]
        raise ValueError(f"{folder}: cannot import strategy package: {error}") from error
    return module


def _synthetic_module_name(folder: Path) -> str:
    digest: Final = hashlib.sha1(str(folder.resolve()).encode()).hexdigest()[:8]
    return f"_harness_strategy_{folder.name}_{digest}"


def _load_strategy(name: str, folder: Path, prefix: str | None) -> Strategy:
    module: Final = _load_strategy_module(name, folder, prefix)
    definition: Final = getattr(module, "STRATEGY", None)
    if not isinstance(definition, StrategyDefinition):
        raise ValueError(f"{folder}: __init__.py must export STRATEGY: StrategyDefinition")
    if definition.id != name:
        raise ValueError(f"{folder}: strategy id {definition.id!r} must match folder name {name!r}")
    if definition.directory.resolve() != folder.resolve():
        raise ValueError(f"{folder}: strategy directory must be {folder}")
    if len(set(definition.surfaces)) != len(definition.surfaces) or any(
        surface not in SURFACES for surface in definition.surfaces
    ):
        raise ValueError(f"{folder}: invalid strategy surfaces: {definition.surfaces}")
    keys: Final = tuple((case.surface, case.sdk_function) for case in definition.cases)
    duplicates: Final = tuple(sorted(key for key in set(keys) if keys.count(key) > 1))
    if duplicates:
        raise ValueError(f"{folder}: duplicate strategy cases: {duplicates}")
    expected: Final = frozenset(
        (surface, function)
        for surface in (definition.surfaces or (None,))
        for function in SDK_FUNCTIONS
    )
    actual: Final = frozenset(keys)
    if actual != expected:
        missing: Final = tuple(sorted(expected - actual))
        extra: Final = tuple(sorted(actual - expected))
        raise ValueError(
            f"{folder}: strategy cases must exactly match its declared matrix; missing={missing}, extra={extra}"
        )
    incompatible: Final = tuple(
        (case.surface, case.sdk_function)
        for case in definition.cases
        if case.spec.disposition is CaseDisposition.RUNNABLE
        and not isinstance(case.spec, definition.runnable_spec)
    )
    if incompatible:
        raise ValueError(f"{folder}: runnable cases do not match {definition.runnable_spec.__name__}: {incompatible}")
    cases: Final = tuple(
        HarnessCase(
            strategy_id=definition.id,
            strategy_label=definition.label,
            sdk_function=case.sdk_function,
            spec=case.spec,
            surface=case.surface,
        )
        for case in definition.cases
    )
    return Strategy(
        definition.order,
        definition.id,
        definition.label,
        definition.description,
        definition.directory,
        cases,
        definition,
    )


def load_catalog(root: Path | None = None) -> tuple[Strategy, ...]:
    resolved: Final = STRATEGIES_ROOT if root is None else root
    prefix: Final = _STRATEGIES_PACKAGE.__name__ if resolved == STRATEGIES_ROOT else None
    folders: Final = tuple(
        info.name for info in pkgutil.iter_modules([str(resolved)]) if info.ispkg
    )
    if not folders:
        raise ValueError(f"No strategy packages found below {resolved}")
    strategies: Final = tuple(
        _load_strategy(name, resolved / name, prefix) for name in sorted(folders)
    )
    ids: Final = [strategy.id for strategy in strategies]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate strategy id in {resolved}")
    return tuple(sorted(strategies, key=lambda strategy: (strategy.order, strategy.id)))
