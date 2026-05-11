# Tiny pin test for the README marker comment added by the workflow smoke test.
# Asserts the HTML comment present in README.md so the marker can't be silently
# removed.
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_readme_contains_hello_marker_comment():
    readme = REPO_ROOT / "README.md"
    assert readme.is_file(), f"expected README.md at {readme}"
    content = readme.read_text(encoding="utf-8")
    assert "<!-- hello from shin -->" in content, (
        "README.md is missing the expected one-line marker comment "
        "'<!-- hello from shin -->'"
    )
