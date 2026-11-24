"""Tests for .github/scripts/restore_handwritten_description.py."""

import importlib.util
from pathlib import Path
from typing import Final

import pytest

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_MODULE_PATH: Final = _REPO_ROOT / ".github" / "scripts" / "restore_handwritten_description.py"
_spec: Final = importlib.util.spec_from_file_location("restore_handwritten_description", _MODULE_PATH)
restore: Final = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(restore)

_AUTHOR_TEXT: Final = "A bad caller token no longer cools the model down for everyone.\n\nIt still returns 401."


def _body(handwritten: str, tldr: str = "- old bullet") -> str:
    return (
        f"## Handwritten 1-3 sentence description\n\n{handwritten}\n\n"
        f"<details>\n<summary>Details</summary>\n\n## TLDR\n\n{tldr}\n\n</details>\n"
    )


def test_placeholder_written_over_the_authors_text_is_replaced_and_the_rest_of_the_edit_kept():
    edited: Final = _body(restore.PLACEHOLDER, tldr="- new bullet")
    assert restore.restored_body(_body(_AUTHOR_TEXT), edited) == _body(_AUTHOR_TEXT, tldr="- new bullet")


def test_template_comment_around_the_placeholder_does_not_hide_the_reset():
    edited: Final = _body(f"<!-- agents never write here -->\n\n{restore.PLACEHOLDER}")
    assert restore.restored_body(_body(_AUTHOR_TEXT), edited) == _body(
        f"<!-- agents never write here -->\n\n{_AUTHOR_TEXT}"
    )


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(_body(_AUTHOR_TEXT), _body("A rewritten description."), id="author rewrote it"),
        pytest.param(_body(_AUTHOR_TEXT), _body(""), id="author emptied it"),
        pytest.param(_body(restore.PLACEHOLDER), _body(restore.PLACEHOLDER), id="never written"),
        pytest.param(_body(""), _body(restore.PLACEHOLDER), id="was empty"),
        pytest.param("no template at all", _body(restore.PLACEHOLDER), id="heading was missing before"),
        pytest.param(_body(_AUTHOR_TEXT), "no template at all", id="heading is missing after"),
    ],
)
def test_edits_that_did_not_reset_real_text_to_the_placeholder_are_left_alone(before: str, after: str):
    assert restore.restored_body(before, after) is None
