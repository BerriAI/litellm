"""Unit tests for tools/import_skills.py (skills.sh / git-repo skill importer)."""

import sys
import textwrap
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import import_skills as imp  # noqa: E402

# ---------------------------------------------------------------------------
# parse_skill_md — Anthropic Agent Skills SKILL.md (YAML frontmatter + body)
# ---------------------------------------------------------------------------

SKILL_MD = textwrap.dedent("""\
    ---
    name: frontend-design
    description: Guidance for distinctive, high-quality frontend UI.
    metadata:
      version: "2.1"
    ---

    # Frontend design

    Use bold typography and a restrained palette.
    """)


def test_parse_skill_md_extracts_frontmatter_and_body():
    frontmatter, body = imp.parse_skill_md(SKILL_MD)
    assert frontmatter["name"] == "frontend-design"
    assert frontmatter["description"].startswith("Guidance for")
    assert body.startswith("# Frontend design")
    assert "restrained palette" in body


def test_parse_skill_md_without_frontmatter_raises():
    with pytest.raises(ValueError):
        imp.parse_skill_md("# Just a readme\nNo frontmatter here.\n")


def test_parse_skill_md_missing_name_raises():
    with pytest.raises(ValueError):
        imp.parse_skill_md("---\ndescription: no name\n---\nbody\n")


# ---------------------------------------------------------------------------
# detect_license — permissive-only import policy
# ---------------------------------------------------------------------------

MIT_TEXT = "MIT License\n\nPermission is hereby granted, free of charge, ..."
APACHE_TEXT = "Apache License\nVersion 2.0, January 2004\n..."
PROPRIETARY_TEXT = "All rights reserved. Internal use only."


def test_detect_license_mit():
    assert imp.detect_license(MIT_TEXT) == "MIT"


def test_detect_license_apache():
    assert imp.detect_license(APACHE_TEXT) == "Apache-2.0"


def test_detect_license_unknown_returns_none():
    assert imp.detect_license(PROPRIETARY_TEXT) is None


def test_permissive_set_membership():
    assert "MIT" in imp.PERMISSIVE_LICENSES
    assert "Apache-2.0" in imp.PERMISSIVE_LICENSES


# ---------------------------------------------------------------------------
# license_from_label / resolve_skill_license — per-skill license resolution.
# Real repos (anthropics/skills, vercel-labs/agent-skills) carry no root
# LICENSE; the license lives in SKILL.md frontmatter or a per-skill file.
# ---------------------------------------------------------------------------


def test_license_from_label_spdx_short_forms():
    assert imp.license_from_label("MIT") == "MIT"
    assert imp.license_from_label("Apache-2.0") == "Apache-2.0"
    assert imp.license_from_label("apache 2.0") == "Apache-2.0"


def test_license_from_label_prose_pointer_returns_none():
    assert imp.license_from_label("Complete terms in LICENSE.txt") is None
    assert imp.license_from_label(None) is None


def test_resolve_skill_license_frontmatter_wins():
    assert (
        imp.resolve_skill_license(
            {"license": "MIT"}, skill_dir_license=None, repo_license=None
        )
        == "MIT"
    )


def test_resolve_skill_license_falls_back_to_skill_dir_then_repo():
    assert (
        imp.resolve_skill_license(
            {"license": "Complete terms in LICENSE.txt"},
            skill_dir_license="Apache-2.0",
            repo_license="MIT",
        )
        == "Apache-2.0"
    )
    assert (
        imp.resolve_skill_license({}, skill_dir_license=None, repo_license="MIT")
        == "MIT"
    )


# ---------------------------------------------------------------------------
# find_skill_dirs — locate SKILL.md folders inside a cloned repo
# ---------------------------------------------------------------------------


def test_find_skill_dirs(tmp_path):
    (tmp_path / "skills" / "alpha").mkdir(parents=True)
    (tmp_path / "skills" / "alpha" / "SKILL.md").write_text("---\nname: a\n---\nx")
    (tmp_path / "beta").mkdir()
    (tmp_path / "beta" / "SKILL.md").write_text("---\nname: b\n---\ny")
    (tmp_path / "not_a_skill").mkdir()
    (tmp_path / "not_a_skill" / "README.md").write_text("nope")
    # hidden dirs (.git internals, .github mirror copies) must be ignored
    (tmp_path / ".git" / "deep").mkdir(parents=True)
    (tmp_path / ".git" / "deep" / "SKILL.md").write_text("---\nname: c\n---\nz")
    (tmp_path / ".github" / "skills" / "alpha").mkdir(parents=True)
    (tmp_path / ".github" / "skills" / "alpha" / "SKILL.md").write_text(
        "---\nname: a\n---\nx"
    )

    found = imp.find_skill_dirs(tmp_path)
    rel = sorted(str(p.relative_to(tmp_path)) for p in found)
    assert rel == ["beta", "skills/alpha"]


# ---------------------------------------------------------------------------
# build_skill_payload — map onto POST /v1/xct-skills (XCTSkillCreate)
# ---------------------------------------------------------------------------


def test_build_skill_payload_maps_xct_skill_create_fields():
    frontmatter, body = imp.parse_skill_md(SKILL_MD)
    payload = imp.build_skill_payload(
        frontmatter,
        body,
        repo_url="https://github.com/anthropics/skills",
        rel_path="skills/frontend-design",
        license_id="Apache-2.0",
    )
    assert payload["display_title"] == "frontend-design"
    assert payload["description"].startswith("Guidance for")
    assert payload["instructions"] == body
    assert payload["is_public"] is True
    meta = payload["xct_metadata"]
    assert meta["origin_repo"] == "https://github.com/anthropics/skills"
    assert meta["origin_path"] == "skills/frontend-design"
    assert meta["license"] == "Apache-2.0"
    assert meta["marketplace_source"] == "skills.sh"


def test_build_skill_payload_version_from_frontmatter_metadata():
    frontmatter, body = imp.parse_skill_md(SKILL_MD)
    payload = imp.build_skill_payload(
        frontmatter, body, repo_url="r", rel_path="p", license_id="MIT"
    )
    assert payload["version"] == "2.1"


# ---------------------------------------------------------------------------
# skills.sh leaderboard discovery + popularity ordering
# ---------------------------------------------------------------------------

LEADERBOARD_HTML = (
    '<a href="/docs/api">API</a>'
    '<a href="/site/example.com/some-skill">external</a>'
    '<a class="row" href="/vercel-labs/skills/find-skills"><h3>find-skills</h3></a>'
    '<a class="row" href="/anthropics/skills/frontend-design"><h3>x</h3></a>'
    '<a class="row" href="/anthropics/skills/frontend-design"><h3>dup</h3></a>'
    '<a class="row" href="/microsoft/azure-skills/microsoft-foundry"><h3>y</h3></a>'
)


def test_parse_skills_sh_leaderboard_ranked_unique_rows():
    rows = imp.parse_skills_sh_leaderboard(LEADERBOARD_HTML)
    assert rows == [
        ("vercel-labs", "skills", "find-skills"),
        ("anthropics", "skills", "frontend-design"),
        ("microsoft", "azure-skills", "microsoft-foundry"),
    ]


SITEMAP_XML = (
    '<?xml version="1.0"?><urlset>'
    "<url><loc>https://www.skills.sh/vercel-labs/skills/find-skills</loc></url>"
    "<url><loc>https://www.skills.sh/anthropics/skills/frontend-design</loc></url>"
    "<url><loc>https://www.skills.sh/site/example.com/external-skill</loc></url>"
    "</urlset>"
)


def test_parse_skills_sh_sitemap_rows():
    rows = imp.parse_skills_sh_sitemap(SITEMAP_XML)
    assert rows == [
        ("vercel-labs", "skills", "find-skills"),
        ("anthropics", "skills", "frontend-design"),
    ]


def test_order_candidates_by_leaderboard_rank():
    leaderboard = [
        ("anthropics", "skills", "frontend-design"),
        ("vercel-labs", "agent-skills", "vercel-react-best-practices"),
    ]
    on_board = {
        "display_title": "vercel-react-best-practices",
        "xct_metadata": {"origin_repo": "https://github.com/vercel-labs/agent-skills"},
    }
    top = {
        "display_title": "frontend-design",
        "xct_metadata": {"origin_repo": "https://github.com/anthropics/skills"},
    }
    off_board = {
        "display_title": "docx",
        "xct_metadata": {"origin_repo": "https://github.com/anthropics/skills"},
    }
    ordered = imp.order_candidates([off_board, on_board, top], leaderboard)
    # Skills explicitly on the leaderboard come first, in rank order; the
    # rest follow by their repo's rank.
    assert [c["display_title"] for c in ordered] == [
        "frontend-design",
        "vercel-react-best-practices",
        "docx",
    ]


# ---------------------------------------------------------------------------
# select_skills — dedupe against existing titles, cap at limit
# ---------------------------------------------------------------------------


def _candidate(title):
    return {"display_title": title, "xct_metadata": {}}


def test_select_skills_skips_existing_titles_and_respects_limit():
    candidates = [_candidate(f"skill-{i}") for i in range(10)]
    selected = imp.select_skills(
        candidates, limit=5, existing_titles={"skill-0", "skill-3"}
    )
    titles = [c["display_title"] for c in selected]
    assert "skill-0" not in titles
    assert "skill-3" not in titles
    assert len(selected) == 5


def test_select_skills_dedupes_within_batch():
    candidates = [_candidate("dup"), _candidate("dup"), _candidate("uniq")]
    selected = imp.select_skills(candidates, limit=10, existing_titles=set())
    assert [c["display_title"] for c in selected] == ["dup", "uniq"]
