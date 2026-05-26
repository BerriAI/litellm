"""
Static checks on docker/Dockerfile.non_root.

The non_root image is intended for deployment into hardened Kubernetes
clusters where `securityContext.runAsNonRoot: true` is enforced. The
kubelet validates non-root status by parsing the image's USER field as
an integer — a string name like "nobody" is rejected with
CreateContainerConfigError because the kubelet cannot resolve
/etc/passwd inside the image at admission time.
"""

import os
import re

import pytest

DOCKERFILE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "docker",
    "Dockerfile.non_root",
)
DOCKERFILE_DATABASE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "docker",
    "Dockerfile.database",
)


def _final_user_directive(dockerfile_text: str) -> str:
    """Return the value of the last `USER` directive in the file."""
    matches = re.findall(r"^USER\s+(\S+)\s*$", dockerfile_text, re.MULTILINE)
    assert matches, "Dockerfile.non_root has no USER directive"
    return matches[-1]


def _nodejs_apk_add_blocks(dockerfile_text: str) -> list[str]:
    """Return apk add command blocks that install nodejs."""
    return re.findall(
        r"RUN .*?apk add --no-cache .*?nodejs.*?(?=\n\n|^RUN |\Z)",
        dockerfile_text,
        re.MULTILINE | re.DOTALL,
    )


@pytest.mark.skipif(
    not os.path.exists(DOCKERFILE_PATH),
    reason="Dockerfile.non_root not present in this checkout",
)
def test_final_user_directive_is_numeric():
    """The runtime USER must be a numeric UID so kubelet's runAsNonRoot
    admission check (strconv.Atoi) succeeds."""
    with open(DOCKERFILE_PATH, "r", encoding="utf-8") as f:
        contents = f.read()

    final_user = _final_user_directive(contents)

    assert final_user.isdigit(), (
        f"Dockerfile.non_root final USER is {final_user!r}; must be a numeric UID "
        "so Kubernetes' runAsNonRoot admission check can verify non-root status. "
        "See https://kubernetes.io/docs/tasks/configure-pod-container/security-context/"
    )

    assert int(final_user) != 0, (
        f"Dockerfile.non_root final USER is {final_user} (root); the non_root image "
        "must run as a non-zero UID."
    )


@pytest.mark.parametrize(
    "dockerfile_path",
    [DOCKERFILE_PATH, DOCKERFILE_DATABASE_PATH],
)
def test_wolfi_nodejs_installs_libatomic(dockerfile_path: str):
    """Wolfi images that install nodejs for prisma must also install libatomic."""
    if not os.path.exists(dockerfile_path):
        pytest.skip(f"{dockerfile_path} not present in this checkout")

    with open(dockerfile_path, "r", encoding="utf-8") as f:
        contents = f.read()

    nodejs_blocks = _nodejs_apk_add_blocks(contents)
    assert nodejs_blocks, f"{dockerfile_path} has no apk add block that installs nodejs"

    for block in nodejs_blocks:
        assert "libatomic" in block, (
            f"{dockerfile_path} installs nodejs without libatomic in:\n{block}\n"
            "Node/Prisma on Wolfi can fail to start with missing libatomic.so.1."
        )
