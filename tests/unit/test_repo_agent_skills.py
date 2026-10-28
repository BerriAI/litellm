import subprocess
from pathlib import Path
from typing import Final

import pytest
import yaml

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_SKILLS_DIR: Final = _REPO_ROOT / "skills"
_SKILL_DIRECTORIES: Final = tuple(sorted(skill_file.parent for skill_file in _SKILLS_DIR.glob("*/SKILL.md")))
_TRACKED_SKILL_FILES: Final = tuple(
    _REPO_ROOT / tracked_path
    for tracked_path in subprocess.run(
        ["git", "ls-files", "SKILL.md", "*/SKILL.md"], cwd=_REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
)
_NON_END_USER_SKILL_DIRECTORIES: Final = tuple(
    sorted(skill_file.parent for skill_file in _TRACKED_SKILL_FILES if skill_file.parent.parent != _SKILLS_DIR)
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
