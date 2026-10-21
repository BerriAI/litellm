from pathlib import Path
from typing import Final

import pytest
import yaml

_SKILLS_DIR: Final = Path(__file__).resolve().parents[2] / "skills"
_SKILL_DIRECTORIES: Final = tuple(sorted(skill_file.parent for skill_file in _SKILLS_DIR.glob("*/SKILL.md")))


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
