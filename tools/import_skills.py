#!/usr/bin/env python3
"""Bulk-import Agent Skills (SKILL.md format) into a LiteLLM proxy.

skills.sh is a directory over plain git repos, so the import path is:
clone an allowlisted repo shallowly, find every folder carrying a
``SKILL.md`` (Anthropic Agent Skills open standard: YAML frontmatter with
``name``/``description`` + markdown instructions), check the repo license
(permissive only), and POST each skill to ``/v1/xct-skills``.

Only repos from the allowlist are imported — skills can carry executable
scripts and prompt-injection payloads, so the default list is limited to
first-party publishers. Extend at runtime with --repos / --repos-file;
additions there are a deliberate operator decision.

Usage:
    # Review what would be imported (writes a YAML manifest, no POSTs):
    python tools/import_skills.py --dry-run --limit 500

    # Import (idempotent — existing display_titles are skipped):
    LITELLM_ADMIN_KEY=sk-... \\
    python tools/import_skills.py --limit 500 \\
        --base-url https://tokenhub.xcity.one \\
        --repos-file skill_repos.txt

    # Import a previously reviewed manifest verbatim:
    LITELLM_ADMIN_KEY=sk-... \\
    python tools/import_skills.py --from-manifest skills_import_manifest.yaml
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import yaml

USER_AGENT = "xcity-litellm-skill-importer/1.0"

# First-party publishers whose org names are already established in this
# codebase. Anything beyond these is an operator decision via --repos /
# --repos-file at runtime.
DEFAULT_REPOS = [
    "https://github.com/anthropics/skills",
    "https://github.com/vercel-labs/agent-skills",
    "https://github.com/microsoft/azure-skills",
]

PERMISSIVE_LICENSES = {
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Unlicense",
    "CC0-1.0",
}

_LICENSE_PATTERNS: List[Tuple[str, str]] = [
    (r"permission is hereby granted, free of charge", "MIT"),
    (r"\bmit license\b", "MIT"),
    (r"apache license[\s,]+version 2\.0", "Apache-2.0"),
    (r"redistribution and use in source and binary forms", "BSD-3-Clause"),
    (r"permission to use, copy, modify, and/or distribute", "ISC"),
    (r"free and unencumbered software", "Unlicense"),
    (r"cc0 1\.0", "CC0-1.0"),
]

_LICENSE_FILENAMES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING")

_LICENSE_LABELS = {
    "mit": "MIT",
    "apache-2.0": "Apache-2.0",
    "apache 2.0": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "isc": "ISC",
    "unlicense": "Unlicense",
    "cc0-1.0": "CC0-1.0",
}


def detect_license(text: str) -> Optional[str]:
    """SPDX id from license text, or None when unrecognized."""
    lowered = text.lower()
    for pattern, spdx in _LICENSE_PATTERNS:
        if re.search(pattern, lowered):
            return spdx
    return None


def license_from_label(value: Optional[str]) -> Optional[str]:
    """SPDX id from a short frontmatter label like ``license: MIT``.

    Prose pointers ("Complete terms in LICENSE.txt") return None so the
    caller falls through to file-based detection.
    """
    if not value:
        return None
    return _LICENSE_LABELS.get(str(value).strip().lower())


def resolve_skill_license(
    frontmatter: Dict[str, Any],
    *,
    skill_dir_license: Optional[str],
    repo_license: Optional[str],
) -> Optional[str]:
    """Per-skill license: frontmatter label > skill-dir file > repo file.

    Real-world repos (anthropics/skills, vercel-labs/agent-skills) carry no
    root LICENSE — the license lives on each skill.
    """
    return (
        license_from_label(frontmatter.get("license"))
        or skill_dir_license
        or repo_license
    )


def parse_skill_md(text: str) -> Tuple[Dict[str, Any], str]:
    """Split a SKILL.md into (frontmatter, body).

    Raises ValueError when the YAML frontmatter is missing, malformed, or
    lacks the mandatory ``name`` field.
    """
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", text, re.DOTALL)
    if not match:
        raise ValueError("SKILL.md has no YAML frontmatter")
    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError as e:
        raise ValueError(f"SKILL.md frontmatter is not valid YAML: {e}") from e
    if (
        not isinstance(frontmatter, dict)
        or not str(frontmatter.get("name") or "").strip()
    ):
        raise ValueError("SKILL.md frontmatter must include a non-empty 'name'")
    return frontmatter, match.group(2).strip()


def find_skill_dirs(root: Path) -> List[Path]:
    """Directories under root containing a SKILL.md.

    Hidden directories are skipped: .git internals, and mirror copies some
    repos keep under .github (azure-skills duplicates every skill there).
    """
    dirs = [
        p.parent
        for p in root.rglob("SKILL.md")
        if not any(part.startswith(".") for part in p.relative_to(root).parts)
    ]
    return sorted(dirs)


def build_skill_payload(
    frontmatter: Dict[str, Any],
    body: str,
    *,
    repo_url: str,
    rel_path: str,
    license_id: Optional[str],
) -> Dict[str, Any]:
    """Map a parsed SKILL.md onto a POST /v1/xct-skills body (XCTSkillCreate)."""
    metadata = frontmatter.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    version = str(metadata.get("version") or frontmatter.get("version") or "1")
    return {
        "display_title": str(frontmatter["name"]).strip(),
        "description": frontmatter.get("description"),
        "instructions": body,
        "version": version,
        "is_public": True,
        "xct_metadata": {
            "origin_repo": repo_url,
            "origin_path": rel_path,
            "license": license_id,
            "marketplace_source": "skills.sh",
        },
    }


SKILLS_SH_URL = "https://www.skills.sh/"

# owner segments on skills.sh that are site sections, not GitHub owners
# ("site" prefixes external-domain skill routes like /site/example.com/x)
_SKIP_OWNERS = {"docs", "api", "blog", "_next", "site"}

_GITHUB_REPO_RE = re.compile(r"github\.com/([^/]+/[^/]+?)(?:\.git)?/?$", re.IGNORECASE)


def parse_skills_sh_leaderboard(html: str) -> List[Tuple[str, str, str]]:
    """Ranked (owner, repo, skill) rows from the skills.sh leaderboard page."""
    rows: List[Tuple[str, str, str]] = []
    seen = set()
    for match in re.finditer(
        r"href=\"/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)\"", html
    ):
        row = match.groups()
        if row[0] in _SKIP_OWNERS or row in seen:
            continue
        seen.add(row)
        rows.append(row)
    return rows


def parse_skills_sh_sitemap(xml_text: str) -> List[Tuple[str, str, str]]:
    """Ranked (owner, repo, skill) rows from a skills.sh sitemap shard.

    The skill sitemaps are popularity-ordered, so they extend the homepage
    leaderboard with long-tail supply in the same rank order.
    """
    rows: List[Tuple[str, str, str]] = []
    seen = set()
    for match in re.finditer(
        r"<loc>https://(?:www\.)?skills\.sh/"
        r"([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)\s*</loc>",
        xml_text,
    ):
        row = match.groups()
        if row[0] in _SKIP_OWNERS or row in seen:
            continue
        seen.add(row)
        rows.append(row)
    return rows


def order_candidates(
    candidates: List[Dict[str, Any]], leaderboard: List[Tuple[str, str, str]]
) -> List[Dict[str, Any]]:
    """Order by skills.sh popularity: leaderboard skills first (by rank),
    then remaining skills by their repo's best rank. Stable otherwise."""
    skill_rank: Dict[Tuple[str, str], int] = {}
    repo_rank: Dict[str, int] = {}
    for i, (owner, repo, skill) in enumerate(leaderboard):
        repo_key = f"{owner}/{repo}".lower()
        skill_rank.setdefault((repo_key, skill.lower()), i)
        repo_rank.setdefault(repo_key, i)

    fallback = len(leaderboard)

    def sort_key(pair: Tuple[int, Dict[str, Any]]):
        index, candidate = pair
        origin = (candidate.get("xct_metadata") or {}).get("origin_repo") or ""
        match = _GITHUB_REPO_RE.search(origin)
        repo_key = match.group(1).lower() if match else ""
        title = (candidate.get("display_title") or "").lower()
        if (repo_key, title) in skill_rank:
            return (0, skill_rank[(repo_key, title)], index)
        return (1, repo_rank.get(repo_key, fallback), index)

    return [c for _, c in sorted(enumerate(candidates), key=sort_key)]


def select_skills(
    candidates: List[Dict[str, Any]], limit: int, existing_titles: Set[str]
) -> List[Dict[str, Any]]:
    """Drop already-imported / duplicate titles, cap at limit."""
    selected: List[Dict[str, Any]] = []
    seen = set(existing_titles)
    for candidate in candidates:
        if len(selected) >= limit:
            break
        title = candidate["display_title"]
        if title in seen:
            continue
        seen.add(title)
        selected.append(candidate)
    return selected


# ---------------------------------------------------------------------------
# Repo handling (live git/HTTP — not exercised by unit tests)
# ---------------------------------------------------------------------------


SKILLS_SH_SITEMAPS = [
    "https://www.skills.sh/sitemap-skills-1.xml",
    "https://www.skills.sh/sitemap-skills-2.xml",
]


def _fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", "replace")


def fetch_leaderboard() -> List[Tuple[str, str, str]]:
    """Homepage leaderboard rows, extended by the popularity-ordered skill
    sitemaps for long-tail supply. Order is the priority signal."""
    rows: List[Tuple[str, str, str]] = []
    seen = set()

    def add(new_rows: List[Tuple[str, str, str]]) -> None:
        for row in new_rows:
            if row not in seen:
                seen.add(row)
                rows.append(row)

    try:
        add(parse_skills_sh_leaderboard(_fetch_text(SKILLS_SH_URL)))
    except (urllib.error.URLError, OSError) as e:
        print(f"  ! skills.sh leaderboard fetch failed: {e}", file=sys.stderr)
    for sitemap_url in SKILLS_SH_SITEMAPS:
        try:
            add(parse_skills_sh_sitemap(_fetch_text(sitemap_url)))
        except (urllib.error.URLError, OSError) as e:
            print(f"  ! sitemap fetch failed {sitemap_url}: {e}", file=sys.stderr)
    print(f"  skills.sh discovery: {len(rows)} ranked skills")
    return rows


def _repo_license(repo_dir: Path) -> Optional[str]:
    for filename in _LICENSE_FILENAMES:
        path = repo_dir / filename
        if path.is_file():
            return detect_license(path.read_text(encoding="utf-8", errors="replace"))
    return None


def collect_repo_skills(
    repo_url: str, workdir: Path, allow_unknown_license: bool = False
) -> List[Dict[str, Any]]:
    dest = workdir / re.sub(r"[^a-z0-9]+", "_", repo_url.lower()).strip("_")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", repo_url, str(dest)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            f"  ! clone failed {repo_url}: {result.stderr.strip()[:200]}",
            file=sys.stderr,
        )
        return []

    repo_license = _repo_license(dest)

    payloads = []
    for skill_dir in find_skill_dirs(dest):
        rel_path = str(skill_dir.relative_to(dest))
        try:
            text = (skill_dir / "SKILL.md").read_text(
                encoding="utf-8", errors="replace"
            )
            frontmatter, body = parse_skill_md(text)
        except (OSError, ValueError) as e:
            print(f"  ! skip {repo_url}:{rel_path}: {e}", file=sys.stderr)
            continue
        skill_license = resolve_skill_license(
            frontmatter,
            skill_dir_license=_repo_license(skill_dir),
            repo_license=repo_license,
        )
        if skill_license not in PERMISSIVE_LICENSES and not allow_unknown_license:
            print(
                f"  ! skip {repo_url}:{rel_path}: skill-level license "
                f"{skill_license or 'unknown'} is not permissive",
                file=sys.stderr,
            )
            continue
        payloads.append(
            build_skill_payload(
                frontmatter,
                body,
                repo_url=repo_url,
                rel_path=rel_path,
                license_id=skill_license,
            )
        )
    print(f"  {repo_url}: {len(payloads)} importable skills")
    return payloads


# ---------------------------------------------------------------------------
# Proxy API (idempotency + import)
# ---------------------------------------------------------------------------


def _request(url: str, key: str, method: str = "GET", body: Optional[bytes] = None):
    req = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except urllib.error.URLError as e:
        return 0, str(e)


def fetch_existing_titles(base_url: str, key: str) -> Set[str]:
    titles: Set[str] = set()
    cursor: Optional[str] = None
    while True:
        url = f"{base_url.rstrip('/')}/v1/xct-skills?limit=200"
        if cursor:
            url += f"&cursor={urllib.parse.quote(cursor)}"
        status, text = _request(url, key)
        if status != 200:
            print(
                f"  ! listing existing skills failed: {status} {text[:200]}",
                file=sys.stderr,
            )
            return titles
        page = json.loads(text)
        for skill in page.get("data") or []:
            if skill.get("display_title"):
                titles.add(skill["display_title"])
        if not page.get("has_more") or not page.get("next_cursor"):
            return titles
        cursor = page["next_cursor"]


def import_payloads(base_url: str, key: str, payloads: List[Dict[str, Any]]) -> int:
    created = failed = 0
    for payload in payloads:
        title = payload.get("display_title", "?")
        status, text = _request(
            f"{base_url.rstrip('/')}/v1/xct-skills",
            key,
            method="POST",
            body=json.dumps(payload).encode("utf-8"),
        )
        if status in (200, 201):
            created += 1
            print(f"  ✓ {title}")
        else:
            failed += 1
            print(f"  ✗ {title} -> {status} {text[:200]}")
    print(f"\nDone. created={created} failed={failed} total={len(payloads)}")
    return failed


def _load_repo_list(args) -> List[str]:
    repos: List[str] = []
    if args.repos:
        repos.extend(r.strip() for r in args.repos.split(",") if r.strip())
    if args.repos_file:
        with open(args.repos_file, "r", encoding="utf-8") as f:
            repos.extend(
                line.strip() for line in f if line.strip() and not line.startswith("#")
            )
    return repos or list(DEFAULT_REPOS)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument(
        "--base-url",
        default=os.environ.get("LITELLM_BASEURL", "https://tokenhub.xcity.one"),
    )
    ap.add_argument(
        "--repos", help="comma-separated git URLs (overrides default allowlist)"
    )
    ap.add_argument("--repos-file", help="file with one git URL per line, # comments")
    ap.add_argument(
        "--no-discover",
        action="store_true",
        help="skip skills.sh leaderboard discovery; use only --repos/--repos-file/defaults",
    )
    ap.add_argument(
        "--max-repos",
        type=int,
        default=60,
        help="cap on how many leaderboard repos get cloned",
    )
    ap.add_argument("--allow-unknown-license", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--manifest", default="skills_import_manifest.yaml")
    ap.add_argument("--from-manifest")
    args = ap.parse_args()

    key = (
        os.environ.get("LITELLM_ADMIN_KEY")
        or os.environ.get("LITELLM_MASTER_KEY")
        or os.environ.get("PROXY_MASTER_KEY")
        or os.environ.get("LITELLM_API_KEY")
    )
    if not key and not args.dry_run:
        sys.exit(
            "Set LITELLM_ADMIN_KEY / LITELLM_MASTER_KEY (proxy admin key) "
            "in the environment."
        )

    if args.from_manifest:
        with open(args.from_manifest, "r", encoding="utf-8") as f:
            candidates = yaml.safe_load(f) or []
        print(f"Loaded {len(candidates)} payloads from {args.from_manifest}")
    else:
        repos = _load_repo_list(args)
        leaderboard: List[Tuple[str, str, str]] = []
        if not args.no_discover:
            leaderboard = fetch_leaderboard()
            for owner, repo, _skill in leaderboard:
                url = f"https://github.com/{owner}/{repo}"
                if url.lower() not in {r.lower() for r in repos}:
                    repos.append(url)
        repos = repos[: args.max_repos]
        print(f"Cloning {len(repos)} repos ...")
        candidates = []
        with tempfile.TemporaryDirectory(prefix="skill_import_") as tmp:
            for repo_url in repos:
                candidates.extend(
                    collect_repo_skills(
                        repo_url,
                        Path(tmp),
                        allow_unknown_license=args.allow_unknown_license,
                    )
                )
        candidates = order_candidates(candidates, leaderboard)

    existing_titles: Set[str] = set()
    if key:
        existing_titles = fetch_existing_titles(args.base_url, key)
        print(f"{len(existing_titles)} skills already on the proxy")

    selected = select_skills(
        candidates, limit=args.limit, existing_titles=existing_titles
    )
    print(f"Selected {len(selected)} of {len(candidates)} candidate skills")

    if args.dry_run:
        with open(args.manifest, "w", encoding="utf-8") as f:
            yaml.safe_dump(selected, f, sort_keys=False, allow_unicode=True)
        print(f"[dry-run] wrote {len(selected)} payloads to {args.manifest}")
        return

    print(f"Importing into {args.base_url} ...")
    if import_payloads(args.base_url, key, selected):
        sys.exit(1)


if __name__ == "__main__":
    main()
