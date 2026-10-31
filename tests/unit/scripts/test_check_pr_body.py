from pathlib import Path
from typing import Final

from scripts.check_pr_body import problems

TEMPLATE: Final = Path(".github/PULL_REQUEST_TEMPLATE/general.md").read_text()
GOOD_BODY: Final = "## Handwritten 1-3 sentence description\nAdds X so Y works\n## TLDR\n- fix\n## Type\nBug Fix\n"


def test_template_body_with_required_sections_passes():
    assert problems(GOOD_BODY, TEMPLATE) == ()


def test_free_form_body_is_ignored():
    assert problems("Just a quick typo fix.", TEMPLATE) == ()


def test_tags_without_handwritten_heading_fail():
    body = "`<make-me-human-readable>`\nAdds X\n`</make-me-human-readable>`\n## TLDR\n- fix\n## Type\nBug Fix\n"
    assert problems(body, TEMPLATE) == (
        "missing section '## Handwritten 1-3 sentence description'",
        "the summary under '## Handwritten 1-3 sentence description' is still AI-written: the PR author rewrites it "
        "in their own words and deletes the <make-me-human-readable> tag lines",
    )


def test_ai_written_summary_still_tagged_fails():
    body = GOOD_BODY.replace("Adds X so Y works", "`<make-me-human-readable>`\nAdds X\n`</make-me-human-readable>`")
    assert problems(body, TEMPLATE) == (
        "the summary under '## Handwritten 1-3 sentence description' is still AI-written: the PR author rewrites it "
        "in their own words and deletes the <make-me-human-readable> tag lines",
    )


def test_handwritten_heading_must_come_first():
    body = "## TLDR\n- fix\n## Handwritten 1-3 sentence description\nAdds X\n## Type\nBug Fix\n"
    assert problems(body, TEMPLATE) == (
        "'## Handwritten 1-3 sentence description' must be the first section, found '## TLDR' first",
    )


def test_leftover_placeholder_fails():
    assert problems(GOOD_BODY.replace("Adds X so Y works", "<blah>"), TEMPLATE) == (
        "template placeholder '<blah>' left in the body",
    )
