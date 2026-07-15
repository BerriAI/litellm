"""Regression tests for ``_cli_key.build_compat_cli_key_body``.

Pin the scope-narrowing invariants that make the compat CLI key safe
to hand to an untrusted subprocess. A silent widening (allowed_routes
grows to include management routes, or the models allowlist becomes
empty) would restore the master-key admin surface these tests exist
to prevent.
"""

from __future__ import annotations

import pytest

from claude_code._cli_key import build_compat_cli_key_body


def test_key_body_restricts_routes_to_inference_only() -> None:
    """SECURITY-critical: the key must be scoped to the inference
    route group and nothing else. Adding management, spend, or key
    routes here would leak the surface a compromised CLI can pivot to."""
    body = build_compat_cli_key_body(["claude-haiku-4-5"])
    assert body.allowed_routes == ["llm_api_routes"]


def test_key_body_lists_only_the_supplied_deployment_names() -> None:
    """The models allowlist is what makes the scope non-empty on the
    proxy side. An empty list would fall back to "all models this key
    can access," which for a master-signed generate is wide-open."""
    names = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7-vertex"]
    body = build_compat_cli_key_body(names)
    assert body.models == names


def test_key_body_carries_a_stable_alias_for_teardown() -> None:
    """A predictable alias makes ad-hoc cleanup of a leaked session key
    possible via /key/info + /key/delete without needing to remember
    the token itself."""
    body = build_compat_cli_key_body(["claude-haiku-4-5"])
    assert body.key_alias == "compat-cli-session-key"


def test_key_body_consumes_arbitrary_iterables_not_just_lists() -> None:
    """The fixture builds the model names via a generator over the
    deployment list. Passing that generator must produce the same body
    as passing a materialized list - protects against a future refactor
    that swaps in an iterable that gets exhausted."""
    names = ("claude-haiku-4-5", "claude-sonnet-4-6")

    def gen():
        for n in names:
            yield n

    body = build_compat_cli_key_body(gen())
    assert body.models == list(names)


def test_key_body_never_includes_admin_route_groups() -> None:
    """Explicitly assert the groups that MUST NOT appear. If a future
    edit accidentally widens the scope, this test names the specific
    groups that would give the CLI admin capabilities."""
    body = build_compat_cli_key_body(["claude-haiku-4-5"])
    for forbidden in (
        "management_routes",
        "info_routes",
        "spend_tracking_routes",
        "self_managed_routes",
        "admin_only_routes",
        "sso_only_routes",
    ):
        assert forbidden not in (body.allowed_routes or []), (
            f"allowed_routes must not include {forbidden!r} - the CLI "
            "gets that route group's entire surface if it captures the "
            "key, and that's exactly the attack we're preventing"
        )
