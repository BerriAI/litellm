#!/usr/bin/env python3
"""
Detect and close duplicate GitHub issues using title similarity.

Modes:
  --scan           Compare all open issues against each other (batch)
  --issue-number N Check a single issue against older open issues

Requires the `gh` CLI to be authenticated.
"""

import argparse
import difflib
import json
import re
import subprocess
import sys


def normalize_title(title: str) -> str:
    """Strip common prefixes, lowercase, and collapse whitespace."""
    title = re.sub(
        r"^\[?(bug|feature request|enhancement|question|docs)[:\]]?\s*",
        "",
        title,
        flags=re.IGNORECASE,
    )
    return " ".join(title.lower().split())


def gh(*args: str) -> str:
    """Run a gh CLI command and return stdout."""
    result = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def fetch_open_issues(repo: str | None) -> list[dict]:
    """Fetch all open issues (excluding PRs) via gh api --paginate."""
    endpoint = "repos/{owner}/{repo}/issues?state=open&per_page=100&sort=created&direction=asc"
    cmd = ["api", "--paginate", endpoint]
    if repo:
        cmd.extend(["-f", f"owner={repo.split('/')[0]}", "-f", f"repo={repo.split('/')[1]}"])
        endpoint = f"repos/{repo}/issues?state=open&per_page=100&sort=created&direction=asc"
        cmd = ["api", "--paginate", endpoint]
    else:
        cmd = ["api", "--paginate", "repos/{owner}/{repo}/issues?state=open&per_page=100&sort=created&direction=asc"]

    raw = gh(*cmd)
    # gh --paginate concatenates JSON arrays, so we may get multiple arrays
    issues = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if isinstance(parsed, list):
            issues.extend(parsed)
        else:
            issues.append(parsed)

    # Filter out pull requests (they also appear in the issues endpoint)
    return [i for i in issues if "pull_request" not in i]


def close_as_duplicate(
    issue_number: int, duplicate_of: int, repo: str | None, dry_run: bool
) -> None:
    """Close an issue as duplicate of another, adding a comment and label."""
    repo_args = ["--repo", repo] if repo else []

    if dry_run:
        print(f"  [DRY RUN] Would close #{issue_number} as duplicate of #{duplicate_of}")
        return

    # Add comment
    comment_body = (
        f"Closing as duplicate of #{duplicate_of}.\n\n"
        "If you believe this is not a duplicate, please reopen and add context "
        "explaining how this differs."
    )
    gh("issue", "comment", str(issue_number), "--body", comment_body, *repo_args)

    # Add label
    gh("issue", "edit", str(issue_number), "--add-label", "duplicate", *repo_args)

    # Close with not_planned reason
    gh(
        "api",
        f"repos/{repo or '{owner}/{repo}'}/issues/{issue_number}",
        "-X",
        "PATCH",
        "-f",
        "state=closed",
        "-f",
        "state_reason=not_planned",
    )

    print(f"  Closed #{issue_number} as duplicate of #{duplicate_of}")


def find_duplicate(
    issue: dict, candidates: list[dict], threshold: float
) -> dict | None:
    """Return the first candidate whose normalized title is above threshold."""
    norm = normalize_title(issue["title"])
    for candidate in candidates:
        if candidate["number"] == issue["number"]:
            continue
        cand_norm = normalize_title(candidate["title"])
        ratio = difflib.SequenceMatcher(None, norm, cand_norm).ratio()
        if ratio >= threshold:
            return candidate
    return None


def scan_all(issues: list[dict], threshold: float, repo: str | None, dry_run: bool) -> int:
    """Compare every issue against all older issues. Returns count of duplicates found."""
    # Sort oldest first
    issues.sort(key=lambda i: i["number"])
    closed_count = 0

    for idx, issue in enumerate(issues):
        older = issues[:idx]
        if not older:
            continue
        dup = find_duplicate(issue, older, threshold)
        if dup:
            ratio = difflib.SequenceMatcher(
                None,
                normalize_title(issue["title"]),
                normalize_title(dup["title"]),
            ).ratio()
            print(
                f"#{issue['number']}: \"{issue['title']}\"\n"
                f"  -> duplicate of #{dup['number']}: \"{dup['title']}\" "
                f"({ratio:.0%} similar)"
            )
            close_as_duplicate(issue["number"], dup["number"], repo, dry_run)
            closed_count += 1

    return closed_count


def check_single(
    issue_number: int, issues: list[dict], threshold: float, repo: str | None, dry_run: bool
) -> bool:
    """Check a single issue against all older open issues. Returns True if duplicate found."""
    target = None
    for i in issues:
        if i["number"] == issue_number:
            target = i
            break

    if target is None:
        print(f"Issue #{issue_number} not found among open issues.")
        return False

    older = [i for i in issues if i["number"] < issue_number]
    dup = find_duplicate(target, older, threshold)
    if dup:
        ratio = difflib.SequenceMatcher(
            None,
            normalize_title(target["title"]),
            normalize_title(dup["title"]),
        ).ratio()
        print(
            f"#{target['number']}: \"{target['title']}\"\n"
            f"  -> duplicate of #{dup['number']}: \"{dup['title']}\" "
            f"({ratio:.0%} similar)"
        )
        close_as_duplicate(issue_number, dup["number"], repo, dry_run)
        return True

    print(f"#{issue_number}: no duplicate found above threshold {threshold}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and close duplicate GitHub issues")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scan", action="store_true", help="Scan all open issues")
    mode.add_argument("--issue-number", type=int, help="Check a single issue number")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold (0-1)")
    parser.add_argument("--close", action="store_true", help="Actually close duplicates (default is dry-run)")
    parser.add_argument("--repo", type=str, help="Repository (owner/repo). Auto-detected if omitted.")
    args = parser.parse_args()

    dry_run = not args.close

    if dry_run:
        print("=== DRY RUN MODE (pass --close to actually close issues) ===\n")

    print("Fetching open issues...")
    issues = fetch_open_issues(args.repo)
    print(f"Found {len(issues)} open issues.\n")

    if args.scan:
        count = scan_all(issues, args.threshold, args.repo, dry_run)
        print(f"\nTotal duplicates {'found' if dry_run else 'closed'}: {count}")
    else:
        found = check_single(args.issue_number, issues, args.threshold, args.repo, dry_run)
        sys.exit(0 if found else 0)  # Always exit 0; finding no dup is not an error


if __name__ == "__main__":
    main()
