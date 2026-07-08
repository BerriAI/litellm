"""Regression tests for the GitHub release pagination in `run_daily.sh`.

The cron job resolves "newest LiteLLM v*-stable" via the GitHub Releases
API. A previous version of the loop broke as soon as the current page
contained ANY v*-stable tag. The Releases endpoint orders by
`created_at`, NOT by semver, so a backport on an older series cut today
(e.g. v1.80.1-stable) can land on an earlier page than a higher-version
release cut two weeks ago (e.g. v1.83.0-stable). The early-break would
silently pin the cron to a stale tag because the higher-version release
on a later page never made it into the merged set the final `sort_by`
consumed.

These tests pin two things:

  1. The buggy early-break-on-first-stable pattern must not return.
  2. The loop still terminates early on the standard "empty page" guard
     so a quiet release feed doesn't burn API quota.

The shell loop itself is exercised end-to-end with a fake `curl` that
serves canned page JSON, demonstrating that the resolved tag is the
highest-semver stable across all pages even when the highest tag lives
on page 2+.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_DAILY = REPO_ROOT / "tests" / "e2e" / "claude_code" / "cron_vm" / "run_daily.sh"

# The extracted snippet starts AFTER `log`/`die` are defined in run_daily.sh,
# so the test harness has to provide its own stubs. Without them, a failure
# inside the snippet (e.g. jq returning an empty LITELLM_VERSION) would crash
# with `bash: die: command not found` (exit 127) instead of the intended
# diagnostic, making test failures unnecessarily hard to debug.
_PREAMBLE = (
    "set -Eeuo pipefail\n"
    "log() { printf '==> %s\\n' \"$*\" >&2; }\n"
    "die() { printf 'ERROR: %s\\n' \"$*\" >&2; exit 1; }\n"
)


def test_run_daily_does_not_early_break_on_first_stable_page() -> None:
    """The regex pattern `select(test("...stable$"))] | length > 0` followed
    by `break` is exactly the buggy early-stop. If it ever returns the
    cron will silently start testing against a stale stable tag.
    """
    body = RUN_DAILY.read_text()
    assert (
        "length > 0" not in body
        or "break" not in body
        or (
            # If both substrings exist, make sure they aren't both inside the
            # same release-pagination loop. The current loop only contains
            # a `break` for the empty-page guard, not for any "length > 0"
            # condition.
            not _shares_loop_body(body, "length > 0", "break")
        )
    ), (
        "run_daily.sh contains the old early-break-on-stable pattern. The "
        "Releases endpoint orders by created_at, not semver, so breaking "
        "on first-stable-seen can miss higher-versioned releases sitting "
        "on later pages."
    )


def _shares_loop_body(body: str, needle_a: str, needle_b: str) -> bool:
    """Heuristic: do both needles live inside a `for page in ...; do ... done`
    block? Used as a defensive guard for the static check above."""
    in_loop = False
    saw_a = False
    saw_b = False
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("for page in"):
            in_loop = True
            saw_a = False
            saw_b = False
            continue
        if in_loop and stripped == "done":
            if saw_a and saw_b:
                return True
            in_loop = False
            continue
        if in_loop:
            if needle_a in line:
                saw_a = True
            if needle_b in line:
                saw_b = True
    return False


def test_run_daily_keeps_empty_page_break_guard() -> None:
    """The empty-page break is the only break that should remain in the
    pagination loop — without it a quiet release feed wastes API quota
    walking past the last real page."""
    body = RUN_DAILY.read_text()
    assert "jq 'length' \"${PAGE_JSON}\"" in body, (
        "run_daily.sh must still detect empty pages via `jq 'length' "
        "${PAGE_JSON}`; without this the loop walks the full 5-page cap "
        "even when there are no more releases."
    )
    assert (
        '== "0"' in body
    ), 'The empty-page guard must compare jq\'s length output to "0".'


def _make_fake_curl(scratch: Path, pages: dict[int, str]) -> Path:
    """Build a fake `curl` shim that serves the canned page JSON for
    each `page=N` request and an empty array for any page past the
    last canned one.

    The shim mimics just enough of curl's CLI surface for the cron
    script: it accepts the headers + URL we pass, ignores everything
    we don't need, and writes the canned body to either stdout or the
    --output target if one is given.
    """
    pages_dir = scratch / "pages"
    pages_dir.mkdir()
    for page_num, body in pages.items():
        (pages_dir / f"page{page_num}.json").write_text(body)

    curl_path = scratch / "curl"
    curl_path.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            # Fake curl for run_daily.sh release pagination tests. Serves
            # page JSON from {pages_dir} keyed by the `page=` query value,
            # and returns "[]" for pages past the last canned one (which
            # is exactly how the real GitHub API behaves past the end).
            url=""
            output=""
            while [[ $# -gt 0 ]]; do
              case "$1" in
                -fsS|-fsSL|-H|-o|--output)
                  if [[ "$1" == "-o" || "$1" == "--output" ]]; then
                    output="$2"; shift 2
                  elif [[ "$1" == "-H" ]]; then
                    shift 2
                  else
                    shift
                  fi
                  ;;
                http*)
                  url="$1"; shift
                  ;;
                *)
                  shift
                  ;;
              esac
            done
            page="$(printf '%s' "$url" | sed -n 's/.*[?&]page=\\([0-9]*\\).*/\\1/p')"
            [[ -z "$page" ]] && page=1
            file="{pages_dir}/page${{page}}.json"
            if [[ -f "$file" ]]; then
              if [[ -n "$output" ]]; then cp "$file" "$output"; else cat "$file"; fi
            else
              if [[ -n "$output" ]]; then printf '[]' > "$output"; else printf '[]'; fi
            fi
            """
        )
    )
    curl_path.chmod(0o755)
    return curl_path


def _extract_resolution_snippet() -> str:
    """Pull the pagination + sort_by + assignment block out of run_daily.sh
    so the test exercises the actual production code path (not a copy).

    The block is everything from the GH_AUTH_HEADER setup down through
    the LITELLM_VERSION emission.
    """
    body = RUN_DAILY.read_text()
    start = body.index("GH_AUTH_HEADER=()")
    end = body.index('log "resolved litellm:')
    return body[start:end]


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq not available")
def test_run_daily_resolves_highest_semver_across_pages(tmp_path: Path) -> None:
    """End-to-end: drive the actual run_daily.sh pagination loop with a
    fake curl whose page 1 contains a freshly-cut LOW-version backport
    (v1.80.1-stable) and page 2 contains a two-weeks-old HIGH-version
    release (v1.83.0-stable). The correct behavior is to resolve
    v1.83.0-stable. The pre-fix behavior would resolve v1.80.1-stable
    because the early-break consumed only page 1.
    """
    pages = {
        # Page 1: most-recently-created releases. The order here matches
        # what /releases?page=1 returns: created-at descending. The
        # freshly-cut v1.80.1-stable backport sits at the top, plus a
        # bunch of non-stable releases.
        1: """[
            {"tag_name": "v1.84.0-nightly.1"},
            {"tag_name": "v1.80.1-stable"},
            {"tag_name": "v1.84.0-nightly.0"}
        ]""",
        # Page 2: older releases. The HIGHER-version stable lives here
        # because it was cut two weeks ago, before the v1.80.1 backport.
        2: """[
            {"tag_name": "v1.83.0-rc.5"},
            {"tag_name": "v1.83.0-stable"},
            {"tag_name": "v1.82.4-stable"}
        ]""",
        # Page 3+: empty -> the loop's empty-page guard fires here.
    }
    fake_curl_dir = tmp_path / "shim"
    fake_curl_dir.mkdir()
    _make_fake_curl(fake_curl_dir, pages)

    workdir = tmp_path / "work"
    workdir.mkdir()

    snippet = _extract_resolution_snippet()
    script = (
        _PREAMBLE
        + f"WORKDIR={workdir!s}\n"
        + snippet
        + 'printf "%s" "${LITELLM_VERSION}"\n'
    )

    env = {
        **os.environ,
        "PATH": f"{fake_curl_dir}:{os.environ.get('PATH', '')}",
    }
    # Make sure the loop hits the fake curl, not the system one.
    env.pop("GITHUB_TOKEN", None)
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert result.stdout == "v1.83.0-stable", (
        f"Expected the highest-semver stable across pages 1-2, got "
        f"{result.stdout!r}. stderr={result.stderr!r}"
    )


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq not available")
def test_run_daily_terminates_on_empty_page(tmp_path: Path) -> None:
    """The empty-page guard must fire so we don't always walk all 5
    pages. With a single populated page and an empty page 2 we should
    stop after fetching page 2 (the first empty response)."""
    pages = {1: '[{"tag_name": "v1.50.0-stable"}]'}
    fake_curl_dir = tmp_path / "shim"
    fake_curl_dir.mkdir()
    _make_fake_curl(fake_curl_dir, pages)

    workdir = tmp_path / "work"
    workdir.mkdir()

    snippet = _extract_resolution_snippet()
    script = (
        _PREAMBLE
        + f"WORKDIR={workdir!s}\n"
        + snippet
        + 'printf "%s" "${LITELLM_VERSION}"\n'
    )

    env = {
        **os.environ,
        "PATH": f"{tmp_path}/shim:{os.environ.get('PATH', '')}",
    }
    env.pop("GITHUB_TOKEN", None)
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert result.stdout == "v1.50.0-stable"
    # Only pages 1 and 2 should have been fetched (2 is empty -> break).
    assert (workdir / "releases.page2.json").exists()
    assert not (workdir / "releases.page3.json").exists(), (
        "Empty-page guard didn't fire — the loop kept walking past the "
        "first empty response."
    )
