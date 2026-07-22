"""tests/e2e routes every HTTP call through the typed transport (e2e_http.py), so
raw HTTP client imports (requests, urllib.request, httpx, aiohttp, http.client) are
banned in suite code. Importing requests' exception types for catching is fine
anywhere; a small allowlist grandfathers the files that legitimately make raw calls
(the transport itself, the root conftest liveness probe, and the claude_code version
resolver's constant registry URL fetch). Referenced by tests/e2e/CLAUDE.md."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

E2E_DIR = Path(__file__).resolve().parents[1] / "e2e"

BANNED_MODULES = ("requests", "urllib.request", "http.client", "httpx", "aiohttp")

ALLOWED_RAW_CLIENT_FILES = {
    "e2e_http.py": ("requests",),
    "conftest.py": ("requests",),
    "claude_code/pr_gate_version_resolver.py": ("urllib.request",),
}

EXCEPTION_ONLY_NAMES = frozenset({"RequestException", "ConnectionError", "Timeout", "HTTPError"})


def _is_banned(module: str) -> bool:
    return any(module == banned or module.startswith(banned + ".") for banned in BANNED_MODULES)


def _banned_imports(tree: ast.Module) -> tuple[tuple[str, int], ...]:
    plain = tuple(
        (alias.name, node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if _is_banned(alias.name)
    )
    from_imports = tuple(
        (node.module, node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and _is_banned(node.module)
        and not all(alias.name in EXCEPTION_ONLY_NAMES for alias in node.names)
    )
    return plain + from_imports


def _violations_in(path: Path) -> tuple[str, ...]:
    relative = path.relative_to(E2E_DIR).as_posix()
    allowed = ALLOWED_RAW_CLIENT_FILES.get(relative, ())
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return tuple(
        f"tests/e2e/{relative}:{lineno}: raw HTTP client import '{module}'"
        for module, lineno in _banned_imports(tree)
        if module not in allowed
    )


def main() -> int:
    violations = tuple(
        violation
        for path in sorted(E2E_DIR.rglob("*.py"))
        for violation in _violations_in(path)
    )
    for violation in violations:
        print(violation)
    if violations:
        print(
            f"\n{len(violations)} raw HTTP client import(s) in tests/e2e. "
            "Route the call through tests/e2e/e2e_http.py (get_external for absolute "
            "third-party URLs) so it gets the typed Result handling."
        )
        return 1
    print("tests/e2e raw HTTP client check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
