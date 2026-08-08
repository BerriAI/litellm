"""
Static checks guarding the Prisma migration path in the runtime Docker stages.

`prisma migrate deploy` runs at container startup against the configured
database. Against a fresh/empty database it has to bootstrap the Node-based
Prisma CLI. prisma-python resolves Node/npm from PATH first (use_global_node
defaults to True); only when npm is missing does it fall back to nodeenv,
which downloads a Node build whose dynamic dependencies (e.g. libatomic.so.1)
are not present in the Wolfi runtime image. That fallback is what broke every
fresh deployment since v1.90.0 (issues #33650 and #24554): the download
crashes with `libatomic.so.1: cannot open shared object file`, the migration
silently fails, and every DB-backed endpoint then 500s with
`The table 'public.LiteLLM_TeamTable' does not exist`.

Keeping nodejs AND npm in the runtime stage forces prisma-python down the
global-Node path so nodeenv is never invoked at runtime.
"""

import os
import re

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

RUNTIME_DOCKERFILES = (
    os.path.join(_REPO_ROOT, "Dockerfile"),
    os.path.join(_REPO_ROOT, "docker", "Dockerfile.database"),
)


def _runtime_stage(dockerfile_text: str) -> str:
    """Return the text of the final `FROM ... AS runtime` stage."""
    match = re.search(
        r"^FROM\s+.*\bAS\s+runtime\b.*$", dockerfile_text, re.MULTILINE
    )
    assert match, "no `FROM ... AS runtime` stage found"
    start = match.start()
    next_from = re.search(
        r"^FROM\s+", dockerfile_text[match.end():], re.MULTILINE
    )
    end = match.end() + next_from.start() if next_from else len(dockerfile_text)
    return dockerfile_text[start:end]


def _runtime_apk_packages(runtime_stage: str) -> set[str]:
    """Collect every package named across `apk add` invocations in the stage.

    Handles the backslash-continued, retry-looped `apk add` forms used across
    the runtime stages by scanning each `apk add ...` run up to the next
    shell operator.
    """
    packages: set[str] = set()
    for add in re.finditer(r"apk add[^\n]*(?:\\\n[^\n]*)*", runtime_stage):
        segment = add.group(0).replace("\\\n", " ")
        segment = re.split(r"&&|\|\||;", segment, maxsplit=1)[0]
        tokens = segment.split()[2:]  # drop the literal `apk add`
        packages.update(t for t in tokens if not t.startswith("-"))
    return packages


@pytest.mark.parametrize("dockerfile_path", RUNTIME_DOCKERFILES)
def test_runtime_stage_installs_nodejs_and_npm(dockerfile_path: str):
    """Runtime must ship both nodejs and npm so prisma-python uses the global
    Node toolchain instead of the nodeenv fallback that crashes on Wolfi."""
    assert os.path.exists(dockerfile_path), f"missing {dockerfile_path}"

    with open(dockerfile_path, "r", encoding="utf-8") as f:
        packages = _runtime_apk_packages(_runtime_stage(f.read()))

    missing = {"nodejs", "npm"} - packages
    assert not missing, (
        f"{os.path.relpath(dockerfile_path, _REPO_ROOT)} runtime stage is "
        f"missing {sorted(missing)} from `apk add`. Without npm, `prisma "
        f"migrate deploy` on a fresh database falls back to nodeenv and "
        f"crashes on libatomic.so.1 (issues #33650, #24554)."
    )
