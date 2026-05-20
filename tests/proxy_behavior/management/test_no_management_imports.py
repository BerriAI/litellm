"""Codify G3 (strict-import grep) as a test.

The behavior-pinning suite asserts at the HTTP boundary against a real proxy
app. Two forbidden patterns:

1. ``from litellm.proxy.management_endpoints`` — importing handler functions
   directly turns the suite into unit tests of the handler module rather than
   behavior tests of the API, and makes the suite brittle to refactors.

2. ``mock``/``patch`` on ``user_api_key_auth`` — mocking auth is the
   structural failure mode of today's mock-heavy suite. The whole point of the
   real-DB harness is that auth runs.

Running this as a pytest item means G3 is enforced by CI on every PR, not by
a checklist someone might forget.
"""

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
BEHAVIOR_DIR = REPO_ROOT / "tests" / "proxy_behavior"

FORBIDDEN_IMPORT = re.compile(r"^\s*from\s+litellm\.proxy\.management_endpoints\b")
FORBIDDEN_AUTH_MOCK = re.compile(
    r"(?:mock\.[A-Za-z_]+|patch[a-z_]*)\([^)]*user_api_key_auth"
)

# This very file is the one place where the forbidden patterns appear as
# regex source; exclude it from its own scan so the test is self-checkable.
SELF = pathlib.Path(__file__).resolve()


def _iter_py_files():
    for path in BEHAVIOR_DIR.rglob("*.py"):
        if path.resolve() == SELF:
            continue
        yield path


def test_no_management_endpoint_imports():
    violations = []
    for path in _iter_py_files():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if FORBIDDEN_IMPORT.search(line):
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}"
                )
    assert not violations, (
        "tests/proxy_behavior/ must not import from litellm.proxy.management_endpoints "
        "(G3 — assert at the HTTP boundary). Violations:\n  " + "\n  ".join(violations)
    )


def test_no_user_api_key_auth_mocking():
    violations = []
    for path in _iter_py_files():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if FORBIDDEN_AUTH_MOCK.search(line):
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}"
                )
    assert not violations, (
        "tests/proxy_behavior/ must not mock user_api_key_auth (G3 — auth runs for "
        "real). Violations:\n  " + "\n  ".join(violations)
    )
