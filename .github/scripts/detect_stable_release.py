#!/usr/bin/env python3
"""
Detect New Stable Upstream Release

This script checks if there's a new stable upstream release available
from BerriAI/litellm that hasn't been synced to CARTO's fork yet.

Stable releases are identified by the `-stable` suffix (e.g., v1.78.5-stable).
We skip nightlies, release candidates, and any other non-stable tags.

Usage:
    python detect_stable_release.py

Outputs:
    - If new stable found: Prints version and exits with code 0
    - If no new stable: Prints nothing and exits with code 1
    - On error: Prints error and exits with code 2

Environment Variables:
    GITHUB_TOKEN: GitHub API token for authenticated requests (optional but recommended)
"""

import os
import sys
import re
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from typing import Optional, List, Dict


def get_github_token() -> Optional[str]:
    """Get GitHub token from environment."""
    return os.environ.get("GITHUB_TOKEN")


def fetch_upstream_releases() -> List[Dict]:
    """
    Fetch all releases from upstream BerriAI/litellm repository.

    Returns:
        List of release dictionaries from GitHub API
    """
    url = "https://api.github.com/repos/BerriAI/litellm/releases?per_page=50"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "CARTO-LiteLLM-Sync-Bot"
    }

    token = get_github_token()
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        request = Request(url, headers=headers)
        with urlopen(request, timeout=30) as response:
            data = response.read()
            return json.loads(data)
    except HTTPError as e:
        print(f"Error fetching releases: HTTP {e.code} - {e.reason}", file=sys.stderr)
        sys.exit(2)
    except URLError as e:
        print(f"Error fetching releases: {e.reason}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Unexpected error fetching releases: {e}", file=sys.stderr)
        sys.exit(2)


def parse_version(tag: str) -> Optional[tuple]:
    """
    Parse version from tag string.

    Args:
        tag: Git tag (e.g., "v1.78.5-stable")

    Returns:
        Tuple of (major, minor, patch) or None if invalid
    """
    # Match vX.Y.Z-stable pattern
    match = re.match(r'^v(\d+)\.(\d+)\.(\d+)-stable$', tag)
    if match:
        return tuple(int(x) for x in match.groups())
    return None


def get_stable_releases(releases: List[Dict]) -> List[Dict]:
    """
    Filter releases to only include stable versions.

    Args:
        releases: List of release dictionaries from GitHub API

    Returns:
        List of stable release dictionaries, sorted by version (newest first)
    """
    stable_releases = []

    for release in releases:
        tag = release.get("tag_name", "")

        # Skip pre-releases, drafts, and non-stable tags
        if release.get("prerelease", False):
            continue
        if release.get("draft", False):
            continue
        if not tag.endswith("-stable"):
            continue

        # Parse and validate version
        version = parse_version(tag)
        if version:
            release["parsed_version"] = version
            stable_releases.append(release)

    # Sort by version (newest first)
    stable_releases.sort(key=lambda r: r["parsed_version"], reverse=True)

    return stable_releases


def get_current_version() -> Optional[tuple]:
    """
    Read current version from pyproject.toml.

    Returns:
        Tuple of (major, minor, patch) or None if not found
    """
    try:
        with open("pyproject.toml", "r") as f:
            for line in f:
                if line.startswith("version = "):
                    # Extract version string
                    match = re.search(r'version = "(\d+)\.(\d+)\.(\d+)"', line)
                    if match:
                        return tuple(int(x) for x in match.groups())
        return None
    except FileNotFoundError:
        print("Error: pyproject.toml not found", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error reading pyproject.toml: {e}", file=sys.stderr)
        sys.exit(2)


def check_tag_exists_locally(tag: str) -> bool:
    """
    Check if a git tag exists locally.

    Args:
        tag: Git tag to check

    Returns:
        True if tag exists locally, False otherwise
    """
    import subprocess
    try:
        result = subprocess.run(
            ["git", "tag", "-l", tag],
            capture_output=True,
            text=True,
            check=False
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def main():
    """Main entry point."""
    print("🔍 Checking for new stable upstream releases...", file=sys.stderr)

    # Get current version
    current_version = get_current_version()
    if not current_version:
        print("Error: Could not determine current version from pyproject.toml", file=sys.stderr)
        sys.exit(2)

    print(f"📦 Current version: {'.'.join(map(str, current_version))}", file=sys.stderr)

    # Fetch upstream releases
    releases = fetch_upstream_releases()
    print(f"📥 Fetched {len(releases)} total releases from upstream", file=sys.stderr)

    # Filter for stable releases
    stable_releases = get_stable_releases(releases)
    print(f"✅ Found {len(stable_releases)} stable releases", file=sys.stderr)

    if not stable_releases:
        print("⚠️  No stable releases found in upstream", file=sys.stderr)
        sys.exit(1)

    # Check for newer stable release
    latest_stable = stable_releases[0]
    latest_version = latest_stable["parsed_version"]
    latest_tag = latest_stable["tag_name"]

    print(f"🆕 Latest stable upstream: {latest_tag} ({'.'.join(map(str, latest_version))})", file=sys.stderr)

    # Compare versions
    if latest_version > current_version:
        print(f"🎉 New stable release available: {latest_tag}", file=sys.stderr)
        print(f"📊 Current: {'.'.join(map(str, current_version))} → New: {'.'.join(map(str, latest_version))}", file=sys.stderr)

        # Check if we've already synced this tag locally
        if check_tag_exists_locally(latest_tag):
            print(f"⚠️  Tag {latest_tag} already exists locally (may be merged in carto/main)", file=sys.stderr)
            print("Skipping to avoid duplicate PRs", file=sys.stderr)
            sys.exit(1)

        # Output for GitHub Actions
        print(latest_tag)  # This goes to stdout for capture
        sys.exit(0)
    else:
        print(f"✓ Already on latest stable: {'.'.join(map(str, current_version))}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
