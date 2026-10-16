from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError

_Paths: TypeAlias = Annotated[tuple[str, ...], Field(min_length=1)]
_Sha: TypeAlias = Annotated[str, Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")]
CheckpointState: TypeAlias = Literal["unknown", "drifted", "aligned"]


class _Mapping(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Annotated[str, Field(min_length=1)]
    python_paths: _Paths
    rust_paths: _Paths


class _Manifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    checkpoint_sha: _Sha | None
    checkpoint_state: CheckpointState = "unknown"
    candidate_paths: _Paths
    mappings: Annotated[tuple[_Mapping, ...], Field(min_length=1)]


@dataclass(frozen=True, slots=True)
class DriftError:
    message: str


@dataclass(frozen=True, slots=True)
class _PathChange:
    status: str
    old_path: str | None
    path: str

    @property
    def paths(self) -> tuple[str, ...]:
        return (self.path,) if self.old_path is None else (self.old_path, self.path)


class DriftChange(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    old_path: str | None
    path: str
    mapping_ids: tuple[str, ...]
    sides: tuple[Literal["python", "rust"], ...]
    patch: str


class WorkingTreeChange(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    old_path: str | None
    path: str


class DriftReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["unreviewed", "changes_found", "no_changes"]
    target_sha: str
    checkpoint_sha: str | None
    checkpoint_state: CheckpointState
    comparison_sha: str | None
    comparison_source: Literal["checkpoint", "override", "none"]
    manifest_path: str
    manifest_sha256: str
    mappings: tuple[_Mapping, ...]
    changes: tuple[DriftChange, ...]
    unclassified: tuple[DriftChange, ...]
    working_tree_changes: tuple[WorkingTreeChange, ...]


def _git(repo_root: Path, *args: str) -> str | DriftError:
    try:
        result: Final = subprocess.run(
            ("git", "--no-optional-locks", "--no-pager", "--literal-pathspecs", *args),
            cwd=repo_root,
            capture_output=True,
            check=False,
            timeout=60,
        )
        stdout: Final = result.stdout.decode("utf-8")
        stderr: Final = result.stderr.decode("utf-8")
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        return DriftError(f"Git operation failed: {error}")
    if result.returncode:
        return DriftError(f"Git {' '.join(args)} failed: {stderr.strip()}")
    return stdout


def _commit(repo_root: Path, ref: str) -> str | DriftError:
    result: Final = _git(repo_root, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}")
    return result if isinstance(result, DriftError) else result.strip()


def _read_manifest(source: str, label: str) -> _Manifest | DriftError:
    try:
        manifest: Final = _Manifest.model_validate_json(source)
    except ValidationError as error:
        return DriftError(f"Invalid manifest {label}: {error}")
    if manifest.checkpoint_sha is None and manifest.checkpoint_state != "unknown":
        return DriftError(f"Invalid manifest {label}: checkpoint state requires a checkpoint SHA")
    selectors: Final = tuple(chain(manifest.candidate_paths, _mapping_paths(manifest.mappings)))
    invalid: Final = next((path for path in selectors if not _valid_selector(path)), None)
    if invalid is not None:
        return DriftError(f"Invalid manifest path {invalid!r}: use a relative file or directory prefix ending in /")
    ids: Final = tuple(mapping.id for mapping in manifest.mappings)
    if len(frozenset(ids)) != len(ids) or any(not name.strip() for name in ids):
        return DriftError(f"Invalid manifest {label}: mapping IDs must be nonblank and unique")
    return manifest


def _valid_selector(path: str) -> bool:
    return bool(
        path
        and not path.startswith("/")
        and not any(character in path for character in ("*", "?", "[", "]", "\\", "\0"))
        and all(part not in {"", ".", ".."} for part in path.removesuffix("/").split("/"))
    )


def _mapping_paths(mappings: tuple[_Mapping, ...]) -> Iterator[str]:
    return chain.from_iterable(chain(mapping.python_paths, mapping.rust_paths) for mapping in mappings)


def _manifest_at(repo_root: Path, sha: str, path: str) -> tuple[_Manifest, ...] | DriftError:
    listing: Final = _git(repo_root, "ls-tree", "-z", sha, "--", path)
    if isinstance(listing, DriftError):
        return listing
    if not listing:
        return ()
    source: Final = _git(repo_root, "show", f"{sha}:{path}")
    if isinstance(source, DriftError):
        return source
    manifest: Final = _read_manifest(source, f"{sha}:{path}")
    return manifest if isinstance(manifest, DriftError) else (manifest,)


def _matches(paths: tuple[str, ...], selectors: tuple[str, ...]) -> bool:
    return any(_matches_path(path, selectors) for path in paths)


def _matches_path(path: str, selectors: tuple[str, ...]) -> bool:
    return any(path == selector or (selector.endswith("/") and path.startswith(selector)) for selector in selectors)


def _parse_change(status: str, tokens: Iterator[str]) -> _PathChange:
    first: Final = next(tokens)
    if status.startswith(("R", "C")):
        return _PathChange(status, first, next(tokens))
    return _PathChange(status, None, first)


def _parse_changes(source: str) -> Iterator[_PathChange]:
    tokens: Final = iter(source.split("\0")[:-1])
    return (_parse_change(status, tokens) for status in tokens)


def _parse_working_change(record: str, tokens: Iterator[str]) -> _PathChange:
    status: Final = record[:2]
    path: Final = record[3:]
    old_path: Final = next(tokens) if "R" in status or "C" in status else None
    return _PathChange(status, old_path, path)


def _working_changes(repo_root: Path, selectors: tuple[str, ...]) -> tuple[WorkingTreeChange, ...] | DriftError:
    source: Final = _git(repo_root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if isinstance(source, DriftError):
        return source
    tokens: Final = iter(source.split("\0")[:-1])
    parsed: Final = (_parse_working_change(record, tokens) for record in tokens)
    return tuple(
        WorkingTreeChange(status=change.status, old_path=change.old_path, path=change.path)
        for change in parsed
        if _matches(change.paths, selectors)
    )


def _change_report(
    repo_root: Path, baseline: str, target: str, change: _PathChange, mappings: tuple[_Mapping, ...]
) -> DriftChange | DriftError:
    patch: Final = _git(
        repo_root, "diff", "--no-ext-diff", "--no-textconv", "--find-renames=50%", baseline, target, "--", *change.paths
    )
    if isinstance(patch, DriftError):
        return patch
    python: Final = any(_matches(change.paths, mapping.python_paths) for mapping in mappings)
    rust: Final = any(_matches(change.paths, mapping.rust_paths) for mapping in mappings)
    sides: Final[tuple[Literal["python", "rust"], ...]] = (("python",) if python else ()) + (("rust",) if rust else ())
    ids: Final = tuple(
        sorted(
            frozenset(
                mapping.id
                for mapping in mappings
                if _matches(change.paths, (*mapping.python_paths, *mapping.rust_paths))
            )
        )
    )
    return DriftChange(
        status=change.status, old_path=change.old_path, path=change.path, mapping_ids=ids, sides=sides, patch=patch
    )


def _ancestor(repo_root: Path, baseline: str, target: str) -> DriftError | None:
    result: Final = _git(repo_root, "merge-base", "--is-ancestor", baseline, target)
    if isinstance(result, DriftError):
        return DriftError(f"Comparison commit must be an ancestor of the target: {result.message}")
    return None


def _changed_files(
    repo_root: Path, baseline: str | None, target: str, selectors: tuple[str, ...], mappings: tuple[_Mapping, ...]
) -> tuple[DriftChange, ...] | DriftError:
    if baseline is None:
        return ()
    ancestry: Final = _ancestor(repo_root, baseline, target)
    if ancestry is not None:
        return ancestry
    source: Final = _git(
        repo_root,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--find-renames=50%",
        "--name-status",
        "-z",
        baseline,
        target,
    )
    if isinstance(source, DriftError):
        return source
    reports: Final = tuple(
        _change_report(repo_root, baseline, target, change, mappings)
        for change in _parse_changes(source)
        if _matches(change.paths, selectors)
    )
    error: Final = next((report for report in reports if isinstance(report, DriftError)), None)
    if error is not None:
        return error
    return tuple(report for report in reports if isinstance(report, DriftChange))


def _merged_mapping(identifier: str, mappings: tuple[_Mapping, ...]) -> _Mapping:
    matching: Final = tuple(mapping for mapping in mappings if mapping.id == identifier)
    python: Final = chain.from_iterable(mapping.python_paths for mapping in matching)
    rust: Final = chain.from_iterable(mapping.rust_paths for mapping in matching)
    return _Mapping(
        id=identifier, python_paths=tuple(sorted(frozenset(python))), rust_paths=tuple(sorted(frozenset(rust)))
    )


def _report(
    repo_root: Path, path: str, raw: str, manifest: _Manifest, target: str, baseline: str | None, override: bool
) -> DriftReport | DriftError:
    previous: Final = () if baseline is None else _manifest_at(repo_root, baseline, path)
    if isinstance(previous, DriftError):
        return previous
    current: Final = _manifest_at(repo_root, target, path)
    if isinstance(current, DriftError):
        return current
    checkpoint_policy: Final = (
        ()
        if manifest.checkpoint_sha is None or manifest.checkpoint_sha in {baseline, target}
        else _manifest_at(repo_root, manifest.checkpoint_sha, path)
    )
    if isinstance(checkpoint_policy, DriftError):
        return checkpoint_policy
    policies: Final = (*previous, *checkpoint_policy, *current, manifest)
    all_mappings: Final = tuple(chain.from_iterable(policy.mappings for policy in policies))
    ids: Final = tuple(sorted(frozenset(mapping.id for mapping in all_mappings)))
    mappings: Final = tuple(_merged_mapping(identifier, all_mappings) for identifier in ids)
    candidates: Final = chain.from_iterable(policy.candidate_paths for policy in policies)
    selectors: Final = tuple(chain(candidates, _mapping_paths(mappings), (path,)))
    changes: Final = _changed_files(repo_root, baseline, target, selectors, mappings)
    if isinstance(changes, DriftError):
        return changes
    working: Final = _working_changes(repo_root, selectors)
    if isinstance(working, DriftError):
        return working
    status: Final = "unreviewed" if manifest.checkpoint_sha is None else "changes_found" if changes else "no_changes"
    return DriftReport(
        status=status,
        target_sha=target,
        checkpoint_sha=manifest.checkpoint_sha,
        checkpoint_state=manifest.checkpoint_state,
        comparison_sha=baseline,
        comparison_source="override" if override else "checkpoint" if baseline else "none",
        manifest_path=path,
        manifest_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        mappings=mappings,
        changes=changes,
        unclassified=tuple(change for change in changes if not change.mapping_ids),
        working_tree_changes=working,
    )


def inspect_drift(
    repo_root: Path, manifest_path: Path, target_ref: str = "HEAD", since_ref: str | None = None
) -> DriftReport | DriftError:
    root: Final = repo_root.resolve()
    try:
        resolved_path: Final = (root / manifest_path).resolve()
        relative: Final = resolved_path.relative_to(root).as_posix()
        raw: Final = resolved_path.read_bytes().decode("utf-8")
    except (OSError, RuntimeError, ValueError) as error:
        return DriftError(f"Cannot read manifest inside repository: {error}")
    if not _valid_selector(relative) or PurePosixPath(relative).as_posix() != relative:
        return DriftError("Manifest must be a repository-relative file")
    manifest: Final = _read_manifest(raw, relative)
    if isinstance(manifest, DriftError):
        return manifest
    target: Final = _commit(root, target_ref)
    if isinstance(target, DriftError):
        return target
    checkpoint: Final = None if manifest.checkpoint_sha is None else _commit(root, manifest.checkpoint_sha)
    if isinstance(checkpoint, DriftError):
        return checkpoint
    checkpoint_ancestry: Final = None if checkpoint is None else _ancestor(root, checkpoint, target)
    if checkpoint_ancestry is not None:
        return checkpoint_ancestry
    reference: Final = since_ref if since_ref is not None else checkpoint
    baseline: Final = None if reference is None else _commit(root, reference)
    if isinstance(baseline, DriftError):
        return baseline
    return _report(root, relative, raw, manifest, target, baseline, since_ref is not None)
