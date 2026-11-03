from collections.abc import Iterator
from fnmatch import fnmatch
from pathlib import Path
from typing import Final

import pytest
import yaml

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_SKILLS_DIR: Final = _REPO_ROOT / "skills"
_SKILL_DIRECTORIES: Final = tuple(sorted(skill_file.parent for skill_file in _SKILLS_DIR.glob("*/SKILL.md")))
_FULL_DEPTH_MAX_DEPTH: Final = 5
_FULL_DEPTH_SKIPPED_NAMES: Final = frozenset(("node_modules", ".git", "dist", "build", "__pycache__"))
_GITIGNORED_NAME_PATTERNS: Final = tuple(
    line.strip().rstrip("/")
    for line in (_REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith(("#", "!")) and "/" not in line.strip().rstrip("/")
)


def _absent_from_a_clone_or_skipped_by_the_cli(name: str) -> bool:
    return name in _FULL_DEPTH_SKIPPED_NAMES or any(fnmatch(name, pattern) for pattern in _GITIGNORED_NAME_PATTERNS)


def _full_depth_skill_directories(directory: Path, depth: int) -> Iterator[Path]:
    if (directory / "SKILL.md").is_file():
        yield directory
    if depth == _FULL_DEPTH_MAX_DEPTH:
        return
    for child in sorted(directory.iterdir()):
        if child.is_dir() and not _absent_from_a_clone_or_skipped_by_the_cli(child.name):
            yield from _full_depth_skill_directories(child, depth + 1)


_NON_END_USER_SKILL_DIRECTORIES: Final = tuple(
    directory for directory in _full_depth_skill_directories(_REPO_ROOT, 0) if directory.parent != _SKILLS_DIR
)


def _frontmatter(skill_directory: Path) -> object:
    opening, frontmatter, _body = (skill_directory / "SKILL.md").read_text(encoding="utf-8").split("---\n", 2)
    assert opening == "", f"{skill_directory.name}/SKILL.md must open with a --- frontmatter block"
    return yaml.safe_load(frontmatter)


@pytest.mark.parametrize("skill_directory", _SKILL_DIRECTORIES, ids=lambda path: path.name)
def test_skill_frontmatter_has_string_name_matching_its_directory_and_a_description(skill_directory: Path) -> None:
    frontmatter: Final = _frontmatter(skill_directory)
    assert isinstance(frontmatter, dict), frontmatter
    assert frontmatter.get("name") == skill_directory.name, frontmatter
    description: Final = frontmatter.get("description")
    assert isinstance(description, str) and description.strip(), frontmatter


@pytest.mark.parametrize(
    "skill_directory",
    _NON_END_USER_SKILL_DIRECTORIES,
    ids=lambda path: str(path.relative_to(_REPO_ROOT)),
)
def test_skill_outside_skills_folder_is_hidden_from_end_user_skill_installs(skill_directory: Path) -> None:
    frontmatter: Final = _frontmatter(skill_directory)
    assert isinstance(frontmatter, dict), frontmatter
    metadata: Final = frontmatter.get("metadata")
    assert isinstance(metadata, dict) and metadata.get("internal") is True, (
        f"{skill_directory} needs `metadata: {{internal: true}}`, or `npx skills add BerriAI/litellm` installs it "
        "for end users, from a root dot-folder always and from anywhere with --full-depth (skills CLI 1.7.4, "
        "checked 2026-10-28)"
    )
