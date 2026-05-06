"""
Scope-filtering helper for cloud-agent secrets (LIT-2891 validation #3).

A secret's `scope` field is either:
* the literal string `"all"` — applies to every session for the team, OR
* a list of repo full-names (e.g. `["BerriAI/litellm", "BerriAI/litellm-docs"]`)
  — applies only to sessions whose repo set intersects this list.

This module is the single source of truth for "should this secret be present in
the hydrate payload for this session?" Both the GET endpoints (for UI display)
and the session-create hydrate path (B2) call into here so the access-control
logic can never drift between UI and the wire.
"""

from typing import Any, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlparse

ScopeValue = Union[str, List[str]]


def _normalize_repo(repo: Any) -> Optional[str]:
    """Reduce any repo reference to its `owner/name` form (lowercase).

    Accepts:
    * a plain string (`"BerriAI/litellm"` or `"github.com/BerriAI/litellm"`)
    * a `https://github.com/BerriAI/litellm.git` URL
    * a dict with `full_name` or `url`

    Returns None for anything we can't parse — caller treats that as "no
    match" rather than crashing the hydrate path.
    """
    if repo is None:
        return None

    if isinstance(repo, dict):
        if isinstance(repo.get("full_name"), str):
            return _normalize_repo(repo["full_name"])
        if isinstance(repo.get("url"), str):
            return _normalize_repo(repo["url"])
        return None

    if not isinstance(repo, str):
        return None

    raw = repo.strip()
    if not raw:
        return None

    # URL form
    if "://" in raw:
        parsed = urlparse(raw)
        path = (parsed.path or "").strip("/")
    else:
        path = raw

    # Strip leading host fragments (`github.com/owner/name` → `owner/name`)
    while path.startswith(("github.com/", "gitlab.com/", "bitbucket.org/")):
        path = path.split("/", 1)[1]

    # Strip trailing `.git`
    if path.endswith(".git"):
        path = path[:-4]

    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        return None
    owner, name = parts[0], parts[1]
    return f"{owner.lower()}/{name.lower()}"


def normalize_repos(repos: Iterable[Any]) -> List[str]:
    """Public helper — used by hydrate to canonicalize a session's repo list."""
    out: List[str] = []
    seen = set()
    for r in repos or []:
        normalized = _normalize_repo(r)
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def secret_in_scope(scope: ScopeValue, session_repos: Iterable[Any]) -> bool:
    """Return True if a secret with this `scope` applies to a session.

    `scope == "all"` always matches. A list scope matches when any of its
    entries (normalized) is in the session's normalized repo set. Empty
    scope-list matches nothing — that's the safe default if a UI bug ever
    writes `scope=[]`.
    """
    if scope == "all":
        return True
    if not isinstance(scope, list):
        # Defensive: any unexpected shape is treated as no-match.
        return False
    if not scope:
        return False

    normalized_session = set(normalize_repos(session_repos))
    if not normalized_session:
        return False

    for entry in scope:
        normalized_entry = _normalize_repo(entry)
        if normalized_entry and normalized_entry in normalized_session:
            return True
    return False


def partition_secrets_for_session(
    secrets: Iterable[Tuple[str, ScopeValue]],
    session_repos: Iterable[Any],
) -> Tuple[List[str], List[str]]:
    """Split (name, scope) pairs into (in_scope_names, out_of_scope_names).

    Used by the hydrate-builder to log which secrets it skipped without
    leaking values. Order is preserved.
    """
    repos = list(session_repos or [])
    in_scope: List[str] = []
    out_of_scope: List[str] = []
    for name, scope in secrets:
        if secret_in_scope(scope, repos):
            in_scope.append(name)
        else:
            out_of_scope.append(name)
    return in_scope, out_of_scope
