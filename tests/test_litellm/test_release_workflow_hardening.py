"""Enforces invariants on the release/publish workflows.

This test is the regression net for the hardening introduced in the
'bulletproof release pipeline' PR. Each assertion catches a specific
class of supply-chain regression. The test runs without secrets or
network access — it inspects the workflow YAML files in the repo and
asserts file-shape invariants only.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

RELEASE_WORKFLOWS = [
    "publish_to_pypi.yml",
    "release-docker.yml",
    "create-release.yml",
    "_publish-container.yml",
]

SHA_PIN_RE = re.compile(r"@[0-9a-f]{40}\b")
USES_LINE_RE = re.compile(r"^\s*-?\s*uses:\s*(\S+)")


def _read_workflow_text(name: str) -> str:
    path = WORKFLOWS_DIR / name
    assert path.exists(), f"Expected workflow {name} to exist at {path}"
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("workflow", RELEASE_WORKFLOWS)
def test_release_workflows_only_use_sha_pinned_actions(workflow: str) -> None:
    """Every `uses:` in a release workflow must reference a 40-hex SHA, not a tag."""
    text = _read_workflow_text(workflow)
    offenders: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        m = USES_LINE_RE.match(line)
        if not m:
            continue
        ref = m.group(1)
        # Local reusable workflows (./.github/workflows/...) have no @ref
        if ref.startswith("./"):
            continue
        if "@" not in ref:
            offenders.append((lineno, line.rstrip()))
            continue
        if not SHA_PIN_RE.search(ref):
            offenders.append((lineno, line.rstrip()))
    assert not offenders, (
        f"{workflow} contains non-SHA-pinned action references:\n"
        + "\n".join(f"  L{n}: {ln}" for n, ln in offenders)
    )


def test_publish_pypi_has_explicit_attestations_true() -> None:
    """The PyPI publish step must explicitly set attestations: true."""
    text = _read_workflow_text("publish_to_pypi.yml")
    assert re.search(
        r"attestations:\s*true", text
    ), "publish_to_pypi.yml must set 'attestations: true' explicitly"


def test_publish_pypi_does_not_pass_password_to_pypi_publish() -> None:
    """No `password:` input is passed to pypa/gh-action-pypi-publish (OIDC only)."""
    text = _read_workflow_text("publish_to_pypi.yml")
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if "pypa/gh-action-pypi-publish" in line:
            window = "\n".join(lines[idx : idx + 30])
            assert "password:" not in window, (
                "Found `password:` near pypa/gh-action-pypi-publish in publish_to_pypi.yml — "
                "static credentials must not be passed; OIDC is mandatory"
            )


def test_publish_container_uses_keyless_cosign() -> None:
    """The cosign sign step must NOT pass --key (asserts keyless via Fulcio)."""
    text = _read_workflow_text("_publish-container.yml")
    cosign_sign_matches = re.findall(r"cosign sign[^\n]*", text)
    assert cosign_sign_matches, "_publish-container.yml must contain at least one `cosign sign` invocation"
    for invocation in cosign_sign_matches:
        assert "--key" not in invocation, (
            f"cosign sign invocation contains --key: {invocation!r}. "
            "Keyless signing (Fulcio + OIDC) is mandatory."
        )


def test_publish_container_login_steps_have_no_password() -> None:
    """No docker login step in _publish-container.yml passes a static password."""
    text = _read_workflow_text("_publish-container.yml")
    # Only acceptable usage of password: is `secrets.GITHUB_TOKEN` for GHCR.
    forbidden = re.findall(
        r"password:\s*\$\{\{\s*secrets\.(?!GITHUB_TOKEN\b)[A-Z_][A-Z0-9_]*\s*\}\}",
        text,
    )
    assert not forbidden, (
        f"_publish-container.yml passes static secrets as docker login passwords: {forbidden}. "
        "Only ${{ secrets.GITHUB_TOKEN }} (for GHCR auth) is permitted; Docker Hub must use OIDC."
    )


@pytest.mark.parametrize("workflow", RELEASE_WORKFLOWS)
def test_release_workflows_do_not_use_slsa_github_generator(workflow: str) -> None:
    """slsa-github-generator's reusables are abandoned here (broken container
    aggregator / public-fork misdetection). Provenance must come from
    actions/attest-build-provenance instead."""
    text = _read_workflow_text(workflow)
    assert "slsa-framework/slsa-github-generator" not in text, (
        f"{workflow} references slsa-framework/slsa-github-generator. "
        "Use actions/attest-build-provenance for GitHub-native SLSA provenance."
    )


def test_container_provenance_uses_attest_build_provenance() -> None:
    """The container build must generate SLSA provenance via
    actions/attest-build-provenance with push-to-registry."""
    text = _read_workflow_text("_publish-container.yml")
    assert "actions/attest-build-provenance@" in text, (
        "_publish-container.yml must use actions/attest-build-provenance "
        "to generate SLSA build provenance for the pushed image"
    )
    assert re.search(r"push-to-registry:\s*true", text), (
        "_publish-container.yml must set 'push-to-registry: true' so the "
        "provenance attestation is attached as an OCI referrer"
    )


@pytest.mark.parametrize("workflow", RELEASE_WORKFLOWS)
def test_cosign_identity_regexps_are_tag_scoped(workflow: str) -> None:
    """Keyless cosign identity regexps must bind signatures to a tag ref only.
    Allowing refs/heads means a branch-dispatched run produces signatures that
    pass CI but fail the tag-scoped verify commands shipped to consumers."""
    text = _read_workflow_text(workflow)
    offenders = [
        line.strip()
        for line in text.splitlines()
        if "certificate-identity-regexp" in line
        and ("refs/heads" in line or "(heads|tags)" in line or "(tags|heads)" in line)
    ]
    assert not offenders, (
        f"{workflow} has a cosign identity regexp that accepts a non-tag ref:\n"
        + "\n".join(f"  {o}" for o in offenders)
        + "\nSignatures must bind to refs/tags only."
    )


def test_cosign_version_is_consistent_across_release_workflows() -> None:
    """All release workflows must install the same cosign version. Mixing
    cosign majors (e.g. 2.x sign vs 3.x verify) risks signature/bundle
    format drift between the producing and verifying steps."""
    version_re = re.compile(
        r"""(?:cosign-release:\s*['"]?|COSIGN_VERSION=)(v\d+\.\d+\.\d+)"""
    )
    found: dict[str, set[str]] = {}
    for workflow in RELEASE_WORKFLOWS:
        text = _read_workflow_text(workflow)
        versions = set(version_re.findall(text))
        if versions:
            found[workflow] = versions
    all_versions = {v for vs in found.values() for v in vs}
    assert len(all_versions) == 1, (
        f"Inconsistent cosign versions across release workflows: {found}. "
        "Pin a single cosign version everywhere."
    )


def test_readme_documents_keyless_verification_only() -> None:
    """README image-verification docs must not reference a static cosign key.
    Keyless signing has no checked-in public key; documenting `--key`/cosign.pub
    sends users a verification command that cannot succeed."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "cosign.pub" not in readme, (
        "README references cosign.pub — keyless signing uses no static key"
    )
    assert "--key http" not in readme and "cosign verify --key" not in readme, (
        "README documents keyful cosign verification; signing is keyless"
    )


def test_release_body_instructions_match_asbuilt_slsa() -> None:
    """create-release.yml injects per-release verify instructions. SLSA
    provenance is verified with `gh attestation verify` — the pipeline uses
    actions/attest-build-provenance, not slsa-github-generator, so a
    slsa-verifier instruction would fail for consumers."""
    text = _read_workflow_text("create-release.yml")
    assert "slsa-verifier" not in text, (
        "create-release.yml still instructs consumers to use slsa-verifier; "
        "as-built provenance is verified via `gh attestation verify`"
    )


def test_cosign_pub_is_absent_from_repo_root() -> None:
    """The static cosign.pub key must not be present at repo root."""
    assert not (REPO_ROOT / "cosign.pub").exists(), (
        "cosign.pub exists at repo root. Keyless signing does not use a "
        "static public key — delete cosign.pub."
    )
