from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, TypeAlias

from pydantic import BaseModel, ConfigDict

from ...shared.unit_runners.python_runner import collect_python_tests
from ...shared.unit_runners.rust_runner import RustTestIdentity, RustTestScope, enumerate_rust_tests
from .contracts import UnitTestContract

PythonInventory: TypeAlias = Callable[[Sequence[str], Path], frozenset[str]]
RustInventory: TypeAlias = Callable[[Path, tuple[RustTestScope, ...]], frozenset[RustTestIdentity]]


class MappingReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    python_tests: tuple[str, ...]
    rust_tests: tuple[str, ...]
    mapped_python_tests: tuple[str, ...]
    unmapped_python_tests: tuple[str, ...]
    rust_only_tests: tuple[str, ...]
    missing_python_tests: tuple[str, ...]
    missing_rust_tests: tuple[str, ...]
    duplicate_python_mappings: tuple[str, ...]
    duplicate_rust_mappings: tuple[str, ...]
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
            or self.invalid_unit_parity_exclusions
        )


def _selected(nodeid: str, selectors: Sequence[str]) -> bool:
    source: Final = nodeid.partition("::")[0]
    return any(source == selector or source.startswith(f"{selector.rstrip('/')}/") for selector in selectors)


def audit_mapping(
    contract: UnitTestContract,
    repo_root: Path,
    *,
    python_inventory: PythonInventory = collect_python_tests,
    rust_inventory: RustInventory = enumerate_rust_tests,
) -> MappingReport:
    mapping: Final = contract.mapping
    python_tests: Final = python_inventory(mapping.python_selectors, repo_root)
    unit_parity_tests: Final = frozenset(
        nodeid for nodeid in python_tests if _selected(nodeid, contract.unit_parity.python_selectors)
    )
    rust_tests: Final = rust_inventory(repo_root, mapping.rust_scope)
    mapped_python: Final = frozenset(item.python for item in mapping.mappings)
    mapped_rust: Final = frozenset(item.rust for item in mapping.mappings)
    duplicate_python: Final = tuple(
        sorted(nodeid for nodeid, count in Counter(item.python for item in mapping.mappings).items() if count > 1)
    )
    duplicate_rust: Final = tuple(
        sorted(identity.key for identity, count in Counter(item.rust for item in mapping.mappings).items() if count > 1)
    )
    return MappingReport(
        python_tests=tuple(sorted(python_tests)),
        rust_tests=tuple(sorted(identity.key for identity in rust_tests)),
        mapped_python_tests=tuple(sorted(python_tests & mapped_python)),
        unmapped_python_tests=tuple(sorted(python_tests - mapped_python)),
        rust_only_tests=tuple(sorted(identity.key for identity in rust_tests - mapped_rust)),
        missing_python_tests=tuple(sorted(mapped_python - python_tests)),
        missing_rust_tests=tuple(sorted(identity.key for identity in mapped_rust - rust_tests)),
        duplicate_python_mappings=duplicate_python,
        duplicate_rust_mappings=duplicate_rust,
        invalid_unit_parity_exclusions=tuple(
            sorted(
                exclusion.nodeid
                for exclusion in contract.unit_parity.exclusions
                if exclusion.nodeid not in unit_parity_tests
            )
        ),
    )
