#!/usr/bin/env python3
"""Print the unit test files a pull request's changed files point at, one per line.

Reads the changed paths on stdin. A changed `tests/unit/**/test_*.py` file is selected,
and so is every `tests/unit/<dir>/test_<module>.py` or `test_<module>_*.py` mirroring a
changed `litellm/<dir>/<module>.py`. Any change inside a package whose tests do not mirror
its layout, migration SQL included, selects that package's whole test directory, and a cost
map change selects the cost map tests, both listed in `UNMIRRORED_TESTS`. Files the unit-core
shards already run are left out,
as are files whose full-matrix shard needs settings this lane does not have: serial
execution or another Python version.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Final

import yaml

UNIT_ROOT: Final = PurePosixPath("tests/unit")
SOURCE_ROOT: Final = PurePosixPath("litellm")
COST_MAP_TESTS: Final = f"{UNIT_ROOT}/test_model_prices_and_context_window.py"
UNMIRRORED_TESTS: Final = (
    (PurePosixPath("litellm/proxy-extras"), f"{UNIT_ROOT}/litellm_proxy_extras/test_*.py"),
    (PurePosixPath("model_prices_and_context_window.json"), COST_MAP_TESTS),
    (PurePosixPath("litellm/model_prices_and_context_window_backup.json"), COST_MAP_TESTS),
)


def _shard_paths(shard: Mapping[str, object]) -> tuple[PurePosixPath, ...]:
    tokens: Final = str(shard.get("test-path", "")).split()
    return tuple(PurePosixPath(token.split("::", 1)[0]) for token in tokens if not token.startswith("-"))


def _shards(workflow: Mapping[str, object], job: str) -> tuple[Mapping[str, object], ...]:
    jobs: Final = workflow.get("jobs")
    spec: Final = jobs.get(job) if isinstance(jobs, dict) else None
    strategy: Final = spec.get("strategy") if isinstance(spec, dict) else None
    matrix: Final = strategy.get("matrix") if isinstance(strategy, dict) else None
    include: Final = matrix.get("include") if isinstance(matrix, dict) else None
    return tuple(shard for shard in include if isinstance(shard, dict)) if isinstance(include, list) else ()


def _incompatible(shard: Mapping[str, object], python: str) -> bool:
    return str(shard.get("workers")) == "0" or str(shard.get("python-version")) != python


def exclusions(workflow: Mapping[str, object], python: str, include_core: bool = False) -> tuple[PurePosixPath, ...]:
    core: Final = () if include_core else _shards(workflow, "unit-core")
    special: Final = (shard for shard in _shards(workflow, "unit") if _incompatible(shard, python))
    return tuple(itertools.chain.from_iterable(_shard_paths(shard) for shard in itertools.chain(core, special)))


def _is_unit_test(path: PurePosixPath) -> bool:
    return path.suffix == ".py" and path.name.startswith("test_") and path.is_relative_to(UNIT_ROOT)


def _mirror_patterns(path: PurePosixPath) -> tuple[str, ...]:
    if path.suffix != ".py" or path.name == "__init__.py" or not path.is_relative_to(SOURCE_ROOT):
        return ()
    directory: Final = UNIT_ROOT / path.parent.relative_to(SOURCE_ROOT)
    return (f"{directory}/test_{path.stem}.py", f"{directory}/test_{path.stem}_*.py")


def _mirrors(path: PurePosixPath, root: Path) -> Iterator[PurePosixPath]:
    for pattern in _mirror_patterns(path):
        yield from (PurePosixPath(match.relative_to(root).as_posix()) for match in root.glob(pattern))


def _unmirrored_tests(path: PurePosixPath, root: Path) -> Iterator[PurePosixPath]:
    for source, pattern in UNMIRRORED_TESTS:
        if path.is_relative_to(source):
            yield from (PurePosixPath(match.relative_to(root).as_posix()) for match in root.glob(pattern))


def _excluded(path: PurePosixPath, excluded: tuple[PurePosixPath, ...]) -> bool:
    return any(path == entry or path.is_relative_to(entry) for entry in excluded)


def select(changed: Iterable[str], excluded: tuple[PurePosixPath, ...], root: Path) -> tuple[str, ...]:
    paths: Final = tuple(PurePosixPath(line.strip()) for line in changed if line.strip())
    direct: Final = (path for path in paths if _is_unit_test(path))
    mirrored: Final = itertools.chain.from_iterable(_mirrors(path, root) for path in paths)
    unmirrored: Final = itertools.chain.from_iterable(_unmirrored_tests(path, root) for path in paths)
    candidates: Final = frozenset(itertools.chain(direct, mirrored, unmirrored))
    return tuple(
        sorted(path.as_posix() for path in candidates if (root / path).is_file() and not _excluded(path, excluded))
    )


def main() -> None:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, default=Path(".github/workflows/test-unit.yml"))
    parser.add_argument("--python", default="3.12")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--include-core", action="store_true", help="Select tests normally run by unit-core too")
    args: Final = parser.parse_args()
    workflow: Final = yaml.safe_load(args.workflow.read_text(encoding="utf-8"))
    selected: Final = select(sys.stdin, exclusions(workflow, args.python, args.include_core), args.root)
    sys.stdout.write("".join(f"{path}\n" for path in selected))


if __name__ == "__main__":
    main()
