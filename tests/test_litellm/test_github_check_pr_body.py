"""Unit tests for `.github/scripts/check_pr_body.py`."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "check_pr_body.py"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("check_pr_body", SCRIPT_PATH)
    assert spec and spec.loader, f"Could not load spec for {SCRIPT_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_pr_body"] = module
    spec.loader.exec_module(module)
    return module


COMPLIANT_BODY = """## TLDR

Problem this solves:

- Any counts keep drifting toward their ceilings

How it solves it:

- Real types at each Any source across 56 files

## Caveats (if any)

- Rate limiter no longer crashes on usage-less responses

### Final Attestation

- [ ] The tests check the right things, including the edge cases, and regressions in the respective real-world customer use-cases are not possible after this PR
"""


def test_compliant_body_passes(checker):
    assert checker.check_body(COMPLIANT_BODY, ("litellm/main.py",)) == ()


def test_prose_caveats_fails(checker):
    body = "## Caveats (if any)\n\nThree micro-hardenings ride along with the typing because honest annotation exposed them.\n"
    violations = checker.check_body(body, ())
    assert len(violations) == 1
    assert "must be a short bullet" in violations[0].detail


def test_overlong_bullet_fails(checker):
    bullet = "- " + " ".join(f"word{i}" for i in range(15))
    violations = checker.check_body(f"## Caveats (if any)\n\n{bullet}\n", ())
    assert len(violations) == 1
    assert "15 words" in violations[0].detail


def test_wrapped_bullet_continuation_counts_into_its_bullet(checker):
    body = "## Caveats (if any)\n\n- short bullet that\n  wraps onto an indented line\n"
    assert checker.check_body(body, ()) == ()


def test_none_line_in_caveats_passes(checker):
    assert checker.check_body("## Caveats (if any)\n\nNone\n", ()) == ()


def test_qa_runbook_without_e2e_changes_fails(checker):
    body = "## QA runbook\n\n- some manual step\n"
    violations = checker.check_body(body, ("litellm/main.py",))
    assert len(violations) == 1
    assert "delete this section" in violations[0].detail


def test_qa_runbook_with_e2e_changes_passes(checker):
    body = "## QA runbook\n\n- some manual step\n"
    assert checker.check_body(body, ("tests/e2e/test_thing.py",)) == ()


def test_leftover_placeholder_fails(checker):
    violations = checker.check_body("## TLDR\n\nProblem this solves:\n\n- <blah>\n- ...\n", ())
    assert len(violations) == 2
    assert all("placeholder" in violation.detail for violation in violations)


def test_html_comments_are_ignored(checker):
    body = "## Caveats (if any)\n\n<!-- Short bullet points, just like the TLDR: one line per bullet -->\n"
    assert checker.check_body(body, ()) == ()


def test_final_attestation_checkbox_is_not_a_caveat(checker):
    body = (
        "## Caveats (if any)\n\n- a real caveat bullet\n\n### Final Attestation\n\n"
        "- [ ] The tests check the right things, including the edge cases, and regressions"
        " in the respective real-world customer use-cases are not possible after this PR\n"
    )
    assert checker.check_body(body, ()) == ()


def test_empty_body_passes(checker):
    assert checker.check_body("", ()) == ()


def test_multiline_html_comment_spanning_section_is_stripped(checker):
    body = "## QA runbook\n\n<!-- Only needed when your PR edits tests/e2e; delete this section otherwise\n\nExample:\n\n- step one\n-->\n"
    violations = checker.check_body(body, ("litellm/main.py",))
    assert len(violations) == 1
    assert violations[0].section == "QA runbook"


def test_bare_ellipsis_in_proof_output_is_not_a_placeholder(checker):
    body = (
        "## Screenshots / Proof of Fix\n\n"
        "```\n"
        "$ curl http://localhost:4000/v1/chat/completions ...\n"
        "{\"id\": \"chatcmpl-abc\",\n"
        " ...\n"
        " \"usage\": {\"prompt_tokens\": 5}}\n"
        "```\n"
    )
    assert checker.check_body(body, ()) == ()


def test_headings_inside_fenced_block_do_not_open_new_sections(checker):
    body = (
        "## Screenshots / Proof of Fix\n\n"
        "Quoted template excerpt:\n\n"
        "```markdown\n"
        "## Caveats (if any)\n\n"
        "Some prose paragraph that is not a bullet at all\n"
        "```\n"
    )
    assert checker.check_body(body, ()) == ()
