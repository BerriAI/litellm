import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
BEHAVIOR_DIR = REPO_ROOT / "tests" / "proxy_behavior"

FORBIDDEN_IMPORT = re.compile(r"^\s*from\s+litellm\.proxy\.management_endpoints\b")
FORBIDDEN_AUTH_MOCK = re.compile(
    r"(?:mock\.[A-Za-z_]+|patch[a-z_]*)\([^)]*user_api_key_auth"
)
# This file is the only place the forbidden patterns appear as regex source;
# exclude it so it can describe what it forbids.
SELF = pathlib.Path(__file__).resolve()


def _iter_py_files():
    for path in BEHAVIOR_DIR.rglob("*.py"):
        if path.resolve() != SELF:
            yield path


def _scan(pattern):
    violations = []
    for path in _iter_py_files():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if pattern.search(line):
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}"
                )
    return violations


def test_no_management_endpoint_imports():
    violations = _scan(FORBIDDEN_IMPORT)
    assert not violations, (
        "tests/proxy_behavior/ must not import from litellm.proxy.management_endpoints. "
        "Violations:\n  " + "\n  ".join(violations)
    )


def test_no_user_api_key_auth_mocking():
    violations = _scan(FORBIDDEN_AUTH_MOCK)
    assert not violations, (
        "tests/proxy_behavior/ must not mock user_api_key_auth. "
        "Violations:\n  " + "\n  ".join(violations)
    )
