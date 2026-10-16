from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

from .runner import DriftError, DriftReport, inspect_drift


@dataclass(frozen=True, slots=True)
class Repository:
    root: Path
    baseline: str


def _git(root: Path, *arguments: str) -> str:
    result: Final = subprocess.run(
        (
            "git",
            "-c",
            "user.name=Drift Test",
            "-c",
            "user.email=drift@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            *arguments,
        ),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write(root: Path, path: str, content: str) -> None:
    destination: Final = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")


def _commit(root: Path) -> str:
    _git(root, "add", "--all")
    _git(root, "commit", "--quiet", "-m", "test: update fixture behavior")
    return _git(root, "rev-parse", "HEAD")


def _manifest(
    root: Path,
    checkpoint: str | None,
    *,
    checkpoint_state: str | None = None,
    pretty: bool = False,
    candidates: tuple[str, ...] = ("litellm/", "litellm-rust/"),
    mappings: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
        ("messages", ("litellm/messages/", "litellm/common.py"), ("litellm-rust/messages.rs",)),
        ("responses", ("litellm/responses/", "litellm/common.py"), ("litellm-rust/responses.rs",)),
    ),
) -> Path:
    path: Final = root / "drift.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint_sha": checkpoint,
                **({"checkpoint_state": checkpoint_state} if checkpoint_state is not None else {}),
                "candidate_paths": candidates,
                "mappings": tuple(
                    {"id": identifier, "python_paths": python_paths, "rust_paths": rust_paths}
                    for identifier, python_paths, rust_paths in mappings
                ),
            },
            indent=2 if pretty else None,
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def repository(tmp_path: Path) -> Repository:
    root: Final = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "--quiet", "--initial-branch=main")
    _write(root, "litellm/messages/handler.py", "def result():\n    return 'before'\n")
    _write(root, "litellm/common.py", "TIMEOUT = 10\n")
    _write(root, "litellm-rust/messages.rs", 'pub const RESULT: &str = "before";\n')
    return Repository(root=root, baseline=_commit(root))


def test_reports_python_rust_shared_and_unmapped_changes_with_real_patches(repository: Repository) -> None:
    _write(repository.root, "litellm/messages/handler.py", "def result():\n    return 'after'\n")
    _write(repository.root, "litellm/common.py", "TIMEOUT = 20\n")
    _write(repository.root, "litellm-rust/messages.rs", 'pub const RESULT: &str = "after";\n')
    _write(repository.root, "litellm/new_endpoint.py", "ENABLED = True\n")
    target: Final = _commit(repository.root)
    manifest: Final = _manifest(repository.root, repository.baseline)
    original_manifest: Final = manifest.read_bytes()

    result: Final = inspect_drift(repository.root, manifest)

    assert isinstance(result, DriftReport), result
    assert result.status == "changes_found"
    assert (result.target_sha, result.checkpoint_sha, result.comparison_sha) == (
        target,
        repository.baseline,
        repository.baseline,
    )
    assert frozenset((change.path, change.status, change.mapping_ids, change.sides) for change in result.changes) == {
        ("litellm/messages/handler.py", "M", ("messages",), ("python",)),
        ("litellm/common.py", "M", ("messages", "responses"), ("python",)),
        ("litellm-rust/messages.rs", "M", ("messages",), ("rust",)),
        ("litellm/new_endpoint.py", "A", (), ()),
    }
    assert tuple(change.path for change in result.unclassified) == ("litellm/new_endpoint.py",)
    python_change: Final = next(change for change in result.changes if change.path == "litellm/messages/handler.py")
    assert "-    return 'before'" in python_change.patch
    assert "+    return 'after'" in python_change.patch
    assert result.manifest_sha256 == hashlib.sha256(original_manifest).hexdigest()
    assert manifest.read_bytes() == original_manifest


def test_patch_is_usable_when_git_diff_colors_are_forced(repository: Repository) -> None:
    _write(repository.root, "litellm/common.py", "TIMEOUT = 20\n")
    _commit(repository.root)
    _git(repository.root, "config", "color.ui", "always")
    _git(repository.root, "config", "color.diff", "always")
    manifest: Final = _manifest(repository.root, repository.baseline)

    result: Final = inspect_drift(repository.root, manifest)

    assert isinstance(result, DriftReport), result
    assert len(result.changes) == 1
    patch: Final = result.changes[0].patch
    assert "-TIMEOUT = 10\n+TIMEOUT = 20\n" in patch
    patch_path: Final = repository.root.parent / "change.patch"
    patch_path.write_text(patch, encoding="utf-8")
    _git(repository.root, "apply", "--check", "--reverse", str(patch_path))


def test_detects_deletions_and_renames_outside_candidate_directories(repository: Repository) -> None:
    _git(repository.root, "mv", "litellm/messages/handler.py", "renamed_handler.py")
    (repository.root / "litellm-rust/messages.rs").unlink()
    _commit(repository.root)
    manifest: Final = _manifest(repository.root, repository.baseline)

    result: Final = inspect_drift(repository.root, manifest)

    assert isinstance(result, DriftReport), result
    assert frozenset((change.status, change.old_path, change.path) for change in result.changes) == {
        ("R100", "litellm/messages/handler.py", "renamed_handler.py"),
        ("D", None, "litellm-rust/messages.rs"),
    }
    rename: Final = next(change for change in result.changes if change.old_path is not None)
    assert rename.mapping_ids == ("messages",)
    assert rename.sides == ("python",)
    assert "rename from litellm/messages/handler.py" in rename.patch
    assert "rename to renamed_handler.py" in rename.patch


def test_rust_only_changes_still_require_review(repository: Repository) -> None:
    _write(repository.root, "litellm-rust/messages.rs", 'pub const RESULT: &str = "changed independently";\n')
    _commit(repository.root)
    manifest: Final = _manifest(repository.root, repository.baseline)

    result: Final = inspect_drift(repository.root, manifest)

    assert isinstance(result, DriftReport), result
    assert result.status == "changes_found"
    assert tuple((change.path, change.sides) for change in result.changes) == (("litellm-rust/messages.rs", ("rust",)),)


def test_preserves_spaces_and_newlines_in_changed_filenames(repository: Repository) -> None:
    filename: Final = "litellm/messages/handler with space\nand newline.py"
    _write(repository.root, filename, "ENABLED = True\n")
    _commit(repository.root)
    manifest: Final = _manifest(repository.root, repository.baseline)

    result: Final = inspect_drift(repository.root, manifest)

    assert isinstance(result, DriftReport), result
    assert tuple((change.path, change.mapping_ids) for change in result.changes) == ((filename, ("messages",)),)
    assert "+ENABLED = True" in result.changes[0].patch


def test_exact_file_selectors_do_not_capture_files_with_the_same_prefix(repository: Repository) -> None:
    _write(repository.root, "litellm/common.py.backup", "unrelated copy\n")
    _commit(repository.root)
    manifest: Final = _manifest(repository.root, repository.baseline, candidates=("litellm/common.py",))

    result: Final = inspect_drift(repository.root, manifest)

    assert isinstance(result, DriftReport), result
    assert result.status == "no_changes"
    assert result.changes == ()


def test_null_checkpoint_cannot_claim_no_changes(repository: Repository) -> None:
    manifest: Final = _manifest(repository.root, None)

    result: Final = inspect_drift(repository.root, manifest)

    assert isinstance(result, DriftReport), result
    assert result.status == "unreviewed"
    assert result.checkpoint_sha is None
    assert result.checkpoint_state == "unknown"
    assert result.comparison_sha is None
    assert result.changes == ()


@pytest.mark.parametrize("state", ("unknown", "drifted", "aligned"))
def test_no_changes_preserves_recorded_checkpoint_state(repository: Repository, state: str) -> None:
    manifest: Final = _manifest(repository.root, repository.baseline, checkpoint_state=state)
    original_manifest: Final = manifest.read_bytes()

    result: Final = inspect_drift(repository.root, manifest)

    assert isinstance(result, DriftReport), result
    assert result.status == "no_changes"
    assert result.checkpoint_state == state
    assert manifest.read_bytes() == original_manifest


@pytest.mark.parametrize(("state", "with_checkpoint"), (("invalid", True), ("aligned", False), ("drifted", False)))
def test_rejects_invalid_checkpoint_state_without_rewriting_manifest(
    repository: Repository, state: str, with_checkpoint: bool
) -> None:
    manifest: Final = _manifest(
        repository.root, repository.baseline if with_checkpoint else None, checkpoint_state=state
    )
    original_manifest: Final = manifest.read_bytes()

    result: Final = inspect_drift(repository.root, manifest)

    assert isinstance(result, DriftError), result
    assert result.message
    assert manifest.read_bytes() == original_manifest


def test_manifest_digest_preserves_crlf_bytes(repository: Repository) -> None:
    manifest: Final = _manifest(repository.root, repository.baseline)
    raw: Final = manifest.read_bytes().replace(b", ", b",\r\n")
    manifest.write_bytes(raw)

    result: Final = inspect_drift(repository.root, manifest)

    assert isinstance(result, DriftReport), result
    assert result.manifest_sha256 == hashlib.sha256(raw).hexdigest()
    assert manifest.read_bytes() == raw


def test_explicit_target_ignores_later_committed_work(repository: Repository) -> None:
    _write(repository.root, "litellm/common.py", "TIMEOUT = 30\n")
    _commit(repository.root)
    manifest: Final = _manifest(repository.root, repository.baseline)

    result: Final = inspect_drift(repository.root, manifest, target_ref=repository.baseline)

    assert isinstance(result, DriftReport), result
    assert result.status == "no_changes"
    assert result.target_sha == repository.baseline
    assert result.changes == ()


def test_since_override_does_not_create_or_advance_checkpoint(repository: Repository) -> None:
    _write(repository.root, "litellm/common.py", "TIMEOUT = 30\n")
    _commit(repository.root)
    manifest: Final = _manifest(repository.root, None)
    original_manifest: Final = manifest.read_bytes()

    result: Final = inspect_drift(repository.root, manifest, since_ref=repository.baseline[:12])

    assert isinstance(result, DriftReport), result
    assert result.status == "unreviewed"
    assert result.checkpoint_sha is None
    assert result.comparison_sha == repository.baseline
    assert tuple(change.path for change in result.changes) == ("litellm/common.py",)
    assert manifest.read_bytes() == original_manifest


@pytest.mark.parametrize("checkpoint", ("f" * 40, "HEAD", "1234567"))
@pytest.mark.parametrize("use_override", (False, True))
def test_missing_or_nonfull_checkpoint_is_an_error(repository: Repository, checkpoint: str, use_override: bool) -> None:
    manifest: Final = _manifest(repository.root, checkpoint)

    result: Final = inspect_drift(repository.root, manifest, since_ref=repository.baseline if use_override else None)

    assert isinstance(result, DriftError), result
    assert result.message


@pytest.mark.parametrize("use_override", (False, True))
def test_unrelated_checkpoint_is_an_error(repository: Repository, use_override: bool) -> None:
    _git(repository.root, "checkout", "--quiet", "--orphan", "unrelated")
    _git(repository.root, "rm", "--quiet", "-rf", ".")
    _write(repository.root, "other.py", "unrelated = True\n")
    unrelated: Final = _commit(repository.root)
    _git(repository.root, "checkout", "--quiet", "main")
    manifest: Final = _manifest(repository.root, unrelated)

    result: Final = inspect_drift(repository.root, manifest, since_ref=repository.baseline if use_override else None)

    assert isinstance(result, DriftError), result
    assert result.message


def test_removed_policy_paths_stay_visible_from_checkpoint_and_target(repository: Repository) -> None:
    _manifest(
        repository.root,
        None,
        candidates=("litellm/",),
        mappings=(("previous", ("litellm/messages/",), ("legacy-native/",)),),
    )
    checkpoint: Final = _commit(repository.root)
    _write(repository.root, "litellm/messages/handler.py", "def result():\n    return 'after'\n")
    _write(repository.root, "litellm-rust/messages.rs", 'pub const RESULT: &str = "after";\n')
    _manifest(
        repository.root,
        checkpoint,
        candidates=("litellm-rust/",),
        mappings=(("target", ("litellm/responses/",), ("litellm-rust/messages.rs",)),),
    )
    _commit(repository.root)
    manifest: Final = _manifest(
        repository.root,
        checkpoint,
        candidates=("unrelated/",),
        mappings=(("local", ("unrelated/",), ("unrelated-native/",)),),
    )

    result: Final = inspect_drift(repository.root, manifest)

    assert isinstance(result, DriftReport), result
    assert result.status == "changes_found"
    assert ("litellm/messages/handler.py", ("previous",)) in tuple(
        (change.path, change.mapping_ids) for change in result.changes
    )
    assert ("litellm-rust/messages.rs", ("target",)) in tuple(
        (change.path, change.mapping_ids) for change in result.changes
    )
    assert "drift.json" in tuple(change.path for change in result.changes)


def test_report_exposes_merged_selectors_for_the_same_mapping_id(repository: Repository) -> None:
    _manifest(
        repository.root,
        None,
        mappings=(("messages", ("litellm/messages/",), ("old-native/",)),),
    )
    checkpoint: Final = _commit(repository.root)
    _write(repository.root, "litellm/messages/handler.py", "def result():\n    return 'after'\n")
    _manifest(
        repository.root,
        checkpoint,
        mappings=(("messages", ("litellm/common.py",), ("litellm-rust/messages.rs",)),),
    )
    _commit(repository.root)
    manifest: Final = _manifest(
        repository.root,
        checkpoint,
        mappings=(("messages", ("litellm/new-messages/",), ("new-native/",)),),
    )

    result: Final = inspect_drift(repository.root, manifest)

    assert isinstance(result, DriftReport), result
    assert tuple((mapping.id, mapping.python_paths, mapping.rust_paths) for mapping in result.mappings) == (
        (
            "messages",
            ("litellm/common.py", "litellm/messages/", "litellm/new-messages/"),
            ("litellm-rust/messages.rs", "new-native/", "old-native/"),
        ),
    )
    assert ("litellm/messages/handler.py", ("messages",)) in tuple(
        (change.path, change.mapping_ids) for change in result.changes
    )


def test_since_override_keeps_policy_from_the_stored_checkpoint(repository: Repository) -> None:
    _manifest(
        repository.root,
        None,
        candidates=("litellm/",),
        mappings=(("previous", ("litellm/messages/",), ("old-native/",)),),
    )
    checkpoint: Final = _commit(repository.root)
    manifest: Final = _manifest(
        repository.root,
        checkpoint,
        candidates=("unrelated/",),
        mappings=(("current", ("unrelated/",), ("new-native/",)),),
    )
    comparison: Final = _commit(repository.root)
    _write(repository.root, "litellm/messages/handler.py", "def result():\n    return 'after'\n")
    _commit(repository.root)

    result: Final = inspect_drift(repository.root, manifest, since_ref=comparison)

    assert isinstance(result, DriftReport), result
    assert result.checkpoint_sha == checkpoint
    assert result.comparison_sha == comparison
    assert tuple((change.path, change.mapping_ids) for change in result.changes) == (
        ("litellm/messages/handler.py", ("previous",)),
    )


@pytest.mark.parametrize("rename_state", ("none", "committed", "staged"))
@pytest.mark.parametrize("use_override", (False, True))
def test_renamed_manifest_keeps_historical_watch_paths(
    repository: Repository, rename_state: str, use_override: bool
) -> None:
    _write(repository.root, "legacy-owned/handler.py", "ENABLED = False\n")
    _manifest(
        repository.root,
        None,
        candidates=("legacy-owned/",),
        mappings=(("messages", ("legacy-owned/",), ("native/",)),),
        pretty=True,
    )
    checkpoint: Final = _commit(repository.root)
    _manifest(
        repository.root,
        checkpoint,
        candidates=("unrelated/",),
        mappings=(("messages", ("unrelated/",), ("native/",)),),
        pretty=True,
    )
    (repository.root / "policy").mkdir()
    _git(repository.root, "mv", "drift.json", "policy/drift.json")
    comparison: Final = _commit(repository.root)
    _write(repository.root, "legacy-owned/handler.py", "ENABLED = True\n")
    _commit(repository.root)
    if rename_state != "none":
        _git(repository.root, "mv", "policy/drift.json", "policy/current-drift.json")
    if rename_state == "committed":
        _commit(repository.root)
    if rename_state == "staged":
        _git(repository.root, "config", "status.renames", "false")
    manifest: Final = repository.root / ("policy/drift.json" if rename_state == "none" else "policy/current-drift.json")

    result: Final = inspect_drift(repository.root, manifest, since_ref=comparison if use_override else None)

    assert isinstance(result, DriftReport), result
    assert result.checkpoint_sha == checkpoint
    assert result.comparison_sha == (comparison if use_override else checkpoint)
    assert ("legacy-owned/handler.py", ("messages",)) in tuple(
        (change.path, change.mapping_ids) for change in result.changes
    )
    mapping: Final = next(mapping for mapping in result.mappings if mapping.id == "messages")
    assert mapping.python_paths == ("legacy-owned/", "unrelated/")
    if rename_state == "staged":
        assert ("policy/drift.json", "policy/current-drift.json") in tuple(
            (change.old_path, change.path) for change in result.working_tree_changes
        )


def test_manifest_rename_history_survives_later_large_policy_rewrites(repository: Repository) -> None:
    _write(repository.root, "legacy-owned/topic0/handler.py", "ENABLED = False\n")
    _manifest(
        repository.root,
        None,
        candidates=("legacy-owned/",),
        mappings=tuple(
            (f"topic{index}", (f"legacy-owned/topic{index}/",), (f"native/topic{index}/",)) for index in range(10)
        ),
        pretty=True,
    )
    checkpoint: Final = _commit(repository.root)
    _git(repository.root, "mv", "drift.json", "intermediate-drift.json")
    comparison: Final = _commit(repository.root)
    replacement: Final = _manifest(
        repository.root,
        checkpoint,
        candidates=("unrelated/",),
        mappings=(("current", ("unrelated/",), ("native/",)),),
        pretty=True,
    )
    (repository.root / "intermediate-drift.json").write_bytes(replacement.read_bytes())
    replacement.unlink()
    _write(repository.root, "legacy-owned/topic0/handler.py", "ENABLED = True\n")
    _commit(repository.root)
    _git(repository.root, "mv", "intermediate-drift.json", "current-drift.json")
    _commit(repository.root)

    result: Final = inspect_drift(repository.root, repository.root / "current-drift.json", since_ref=comparison)

    assert isinstance(result, DriftReport), result
    assert ("legacy-owned/topic0/handler.py", ("topic0",)) in tuple(
        (change.path, change.mapping_ids) for change in result.changes
    )
    assert tuple(mapping.id for mapping in result.mappings) == ("current", *(f"topic{index}" for index in range(10)))


def test_future_target_recovers_renamed_policy_from_an_older_checkout(repository: Repository) -> None:
    _manifest(
        repository.root,
        None,
        candidates=("legacy-owned/",),
        mappings=(("legacy", ("legacy-owned/",), ("native/",)),),
        pretty=True,
    )
    checkpoint: Final = _commit(repository.root)
    _manifest(
        repository.root,
        checkpoint,
        candidates=("future-owned/",),
        mappings=(("future", ("future-owned/",), ("native/",)),),
        pretty=True,
    )
    _git(repository.root, "mv", "drift.json", "future-drift.json")
    _write(repository.root, "future-owned/handler.py", "ENABLED = True\n")
    target: Final = _commit(repository.root)
    _git(repository.root, "checkout", "--quiet", "--detach", checkpoint)
    manifest: Final = _manifest(
        repository.root,
        checkpoint,
        candidates=("legacy-owned/",),
        mappings=(("legacy", ("legacy-owned/",), ("native/",)),),
        pretty=True,
    )

    result: Final = inspect_drift(repository.root, manifest, target_ref=target)

    assert isinstance(result, DriftReport), result
    assert result.target_sha == target
    assert ("future-owned/handler.py", ("future",)) in tuple(
        (change.path, change.mapping_ids) for change in result.changes
    )


def test_reports_relevant_staged_unstaged_and_untracked_changes_without_diffing_them(repository: Repository) -> None:
    _git(repository.root, "mv", "litellm/messages/handler.py", "renamed_handler.py")
    _write(repository.root, "litellm/common.py", "TIMEOUT = 99\n")
    _write(repository.root, "litellm/new_endpoint.py", "ENABLED = True\n")
    _write(repository.root, "unrelated.txt", "outside the candidate paths\n")
    manifest: Final = _manifest(repository.root, repository.baseline)

    result: Final = inspect_drift(repository.root, manifest)

    assert isinstance(result, DriftReport), result
    assert result.target_sha == repository.baseline
    assert result.changes == ()
    assert frozenset((change.status, change.old_path, change.path) for change in result.working_tree_changes) == {
        ("R ", "litellm/messages/handler.py", "renamed_handler.py"),
        (" M", None, "litellm/common.py"),
        ("??", None, "litellm/new_endpoint.py"),
        ("??", None, "drift.json"),
    }


@pytest.mark.parametrize("selector", ("../litellm/", "/litellm/", "litellm/**", "litellm/../outside/"))
def test_rejects_ambiguous_or_outside_repository_selectors(repository: Repository, selector: str) -> None:
    manifest: Final = _manifest(repository.root, repository.baseline, candidates=(selector,))

    result: Final = inspect_drift(repository.root, manifest)

    assert isinstance(result, DriftError), result
    assert result.message
