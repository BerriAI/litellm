"""Fail if tests/e2e uses os.environ/KEY missing from stage litellm-secrets.yaml.

Source of truth is litellm-ops ExternalSecret litellm-provider-keys in
apps/overlays/berrie-litellm-stage/litellm-secrets.yaml (mounted on the stage
gateway). E2e PRs that add a new os.environ ref without a matching secretKey
there must fail CI.
"""

from __future__ import annotations

import argparse
import os
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

DEFAULT_SECRETS_REL = Path(
    "apps/overlays/berrie-litellm-stage/litellm-secrets.yaml"
)


def repo_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "tests" / "e2e").is_dir():
        return cwd
    return Path(__file__).resolve().parents[2]


def resolve_secrets_yaml(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    env = os.environ.get("LITELLM_OPS_SECRETS_YAML")
    if env:
        return Path(env)
    root = repo_root()
    candidates = (
        root / "litellm-ops" / DEFAULT_SECRETS_REL,
        root.parent / "litellm-ops" / DEFAULT_SECRETS_REL,
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "could not find litellm-ops stage litellm-secrets.yaml; pass "
        "--secrets-yaml or set LITELLM_OPS_SECRETS_YAML. Expected path: "
        f"litellm-ops/{DEFAULT_SECRETS_REL}"
    )


def extract_provider_secret_keys(secrets_yaml: Path) -> frozenset[str]:
    text = secrets_yaml.read_text(encoding="utf-8")
    match = re.search(
        r"(?ms)^kind: ExternalSecret\nmetadata:\n  name: litellm-provider-keys\n.*?^(?:---|\Z)",
        text,
    )
    if not match:
        raise ValueError(
            f"ExternalSecret litellm-provider-keys not found in {secrets_yaml}"
        )
    keys = set(re.findall(r"(?m)^\s+- secretKey: ([A-Z][A-Z0-9_]+)\s*$", match.group(0)))
    if not keys:
        raise ValueError(f"no secretKey entries under litellm-provider-keys in {secrets_yaml}")
    return frozenset(keys)


def collect_refs(e2e_root: Path, root: Path) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(e2e_root.rglob("*.py")):
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--secrets-yaml",
        type=Path,
        default=None,
        help="Path to litellm-ops stage litellm-secrets.yaml",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    e2e_root = root / "tests" / "e2e"
    if not e2e_root.is_dir():
        print(f"missing e2e tree: {e2e_root}", file=sys.stderr)
        return 1

    try:
        secrets_yaml = resolve_secrets_yaml(args.secrets_yaml)
        mounted = extract_provider_secret_keys(secrets_yaml)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    allowed = mounted | HARNESS_ONLY_KEYS
    refs = collect_refs(e2e_root, root)
    missing = sorted(key for key in refs if key not in allowed)
    if not missing:
        print(
            f"ok: {len(refs)} os.environ refs under tests/e2e are keys on "
            f"ExternalSecret litellm-provider-keys in {secrets_yaml} "
            f"({len(mounted)} mounted keys + harness-only)"
        )
        return 0

    print(
        "e2e references os.environ/KEY that are not secretKey entries on "
        "ExternalSecret litellm-provider-keys in litellm-ops "
        f"({secrets_yaml}):\n",
        file=sys.stderr,
    )
    for key in missing:
        print(f"  {key}", file=sys.stderr)
        for loc in refs[key][:5]:
            print(f"    {loc}", file=sys.stderr)
        if len(refs[key]) > 5:
            print(f"    ... +{len(refs[key]) - 5} more", file=sys.stderr)
    print(
        "\nAdd each missing key as a secretKey under litellm-provider-keys in "
        "litellm-ops apps/overlays/berrie-litellm-stage/litellm-secrets.yaml, "
        "put the value in ASM berrie-litellm-stage-provider-keys, merge ops, "
        "wait for Argo/ESO, and restart gateway/backend. Then re-run this check.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
