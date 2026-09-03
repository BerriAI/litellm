from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import pkgutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from .. import strategies as _strategies_package
from ..shared.reporting.models import SDK_FUNCTIONS, HarnessCase, Strategy
from ..shared.reporting.strategy import StrategyDefinition

_STRATEGIES_PACKAGE: Final = _strategies_package
STRATEGIES_ROOT: Final = Path(_STRATEGIES_PACKAGE.__path__[0])


class StrategySpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    order: int
    id: str
    label: str
    description: str
    functions: dict[str, dict[str, object]]
    gateway: dict[str, dict[str, object]] = {}


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
    definition = getattr(module, "STRATEGY", None)
    if not isinstance(definition, StrategyDefinition):
        raise ValueError(f"{folder}: __init__.py must export STRATEGY: StrategyDefinition")
    manifest: Final = definition.directory / "strategy.json"
    if not manifest.exists():
        raise ValueError(f"{definition.directory}: missing strategy.json")
    try:
        data: Final = StrategySpec.model_validate_json(manifest.read_text(encoding="utf-8"))
        adapter: Final = TypeAdapter(dict[str, definition.case_spec])
        functions: Final = adapter.validate_python(data.functions)
        gateway: Final = adapter.validate_python(data.gateway)
    except (ValidationError, json.JSONDecodeError) as error:
        raise ValueError(f"{manifest}: {error}") from error
    if data.id != name:
        raise ValueError(f"{manifest}: manifest id {data.id!r} must match folder name {name!r}")
    if set(functions) != set(SDK_FUNCTIONS):
        raise ValueError(f"{manifest}: functions must exactly match {SDK_FUNCTIONS}")
    cases: Final = (
        *(
            HarnessCase(
                strategy_id=data.id,
                strategy_label=data.label,
                sdk_function=function,
                spec=functions[function],
                surface="sdk",
            )
            for function in SDK_FUNCTIONS
        ),
        *(
            HarnessCase(
                strategy_id=data.id,
                strategy_label=data.label,
                sdk_function=function,
                spec=spec,
                surface="gateway",
            )
            for function, spec in gateway.items()
        ),
    )
    return Strategy(
        data.order, data.id, data.label, data.description, definition.directory, cases, definition
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
