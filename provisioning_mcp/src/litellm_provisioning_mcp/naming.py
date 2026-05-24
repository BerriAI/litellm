"""Helpers for deriving release names, label values, and image repositories."""

from __future__ import annotations

import re

_NON_DNS = re.compile(r"[^a-z0-9-]+")
_GITHUB = re.compile(
    r"(?:https?://|git@|ssh://git@)?github\.com[/:]([^/]+)/([^/]+?)(?:\.git)?/?$"
)

# Helm release names are used as the prefix of Kubernetes object names, which
# are RFC 1123 labels capped at 63 chars; helm itself caps the release name at
# 53 to leave room for the suffixes the chart appends.
RELEASE_MAX_LEN = 53
LABEL_MAX_LEN = 63


def sanitize_release_name(value: str) -> str:
    cleaned = _NON_DNS.sub("-", value.lower()).strip("-")
    cleaned = re.sub(r"-{2,}", "-", cleaned)[:RELEASE_MAX_LEN].strip("-")
    if not cleaned:
        raise ValueError(f"cannot derive a valid release name from {value!r}")
    return cleaned


def sanitize_label(value: str) -> str:
    """Coerce ``value`` into a valid Kubernetes label value (best effort)."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-_.")
    return cleaned[:LABEL_MAX_LEN].strip("-_.")


def registry_from_repo_url(repo_url: str) -> str | None:
    """Map a GitHub repo URL to its GHCR org, e.g. ``ghcr.io/<owner>``.

    Encodes the e2e convention that a fork's CI publishes the litellm component
    images to its own ``ghcr.io/<owner>`` namespace. Returns ``None`` for URLs
    that are not GitHub repositories so the caller can fall back to a default.
    """
    match = _GITHUB.search(repo_url.strip())
    if not match:
        return None
    owner = match.group(1).lower()
    return f"ghcr.io/{owner}"


def derive_image_repos(
    *, repo_url: str, registry_override: str | None, default_registry: str
) -> dict[str, str]:
    registry = (
        registry_override or registry_from_repo_url(repo_url) or default_registry
    ).rstrip("/")
    return {
        "gateway": f"{registry}/litellm-gateway",
        "backend": f"{registry}/litellm-backend",
        "ui": f"{registry}/litellm-ui",
        "migrations": f"{registry}/litellm-migrations",
    }
