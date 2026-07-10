"""Fail if tests/e2e references os.environ/KEY that stage gateway does not mount.

Stage mounts provider credentials via ExternalSecret litellm-provider-keys
(litellm-ops). E2e provisions deployments with os.environ/KEY so the gateway
must have KEY in its env. This check keeps PRs from adding a ref without
updating the allowlist (and, by convention, litellm-ops + ASM).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

OS_ENVIRON_REF = re.compile(r"os\.environ/([A-Z][A-Z0-9_]*)")
_ENV_REF_FIRST_ARG = re.compile(r"""_env_ref\(\s*["']([A-Z][A-Z0-9_]*)["']""")

HARNESS_ONLY_KEYS = frozenset(
    {
        "LITELLM_MASTER_KEY",
        "DATABASE_URL",
        "LITELLM_PROXY_URL",
        "LITELLM_CONTROL_PLANE_URL",
        "E2E_UI_USERNAME",
        "E2E_UI_PASSWORD",
        "E2E_POLL_TIMEOUT",
        "E2E_POLL_INTERVAL",
        "E2E_REQUEST_TIMEOUT",
    }
)

SKIP_DIR_NAMES = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        "fixtures",
        "litellm-regression-tests",
    }
)


def repo_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "tests" / "e2e").is_dir():
        return cwd
    return Path(__file__).resolve().parents[2]


def load_stage_keys(path: Path) -> frozenset[str]:
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        keys.add(stripped)
    return frozenset(keys)


def collect_refs(e2e_root: Path, root: Path) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(e2e_root.rglob("*")):
        if not path.is_file() or path.suffix != ".py":
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(root)
        for match in OS_ENVIRON_REF.finditer(text):
            key = match.group(1)
            found.setdefault(key, []).append(f"{rel}:{_line_no(text, match.start())}")
        for match in _ENV_REF_FIRST_ARG.finditer(text):
            key = match.group(1)
            found.setdefault(key, []).append(f"{rel}:{_line_no(text, match.start())}")
    return found


def _line_no(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def main() -> int:
    root = repo_root()
    allowlist_path = root / "tests" / "e2e" / "stage_gateway_env_keys.txt"
    e2e_root = root / "tests" / "e2e"
    if not allowlist_path.is_file():
        print(f"missing allowlist: {allowlist_path}", file=sys.stderr)
        return 1
    if not e2e_root.is_dir():
        print(f"missing e2e tree: {e2e_root}", file=sys.stderr)
        return 1

    allowed = load_stage_keys(allowlist_path) | HARNESS_ONLY_KEYS
    refs = collect_refs(e2e_root, root)
    missing = sorted(key for key in refs if key not in allowed)
    if not missing:
        print(
            f"ok: {len(refs)} os.environ refs under tests/e2e are covered by "
            f"{allowlist_path.name} (+ harness-only keys)"
        )
        return 0

    print(
        "e2e references os.environ/KEY that are not in "
        f"{allowlist_path.relative_to(root)} (stage gateway env contract):\n",
        file=sys.stderr,
    )
    for key in missing:
        print(f"  {key}", file=sys.stderr)
        for loc in refs[key][:5]:
            print(f"    {loc}", file=sys.stderr)
        if len(refs[key]) > 5:
            print(f"    ... +{len(refs[key]) - 5} more", file=sys.stderr)
    print(
        "\nAdd the key to tests/e2e/stage_gateway_env_keys.txt and mount it on "
        "stage via litellm-ops "
        "apps/overlays/berrie-litellm-stage/litellm-secrets.yaml "
        "(ExternalSecret litellm-provider-keys) plus the ASM JSON "
        "berrie-litellm-stage-provider-keys, then restart gateway/backend.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
