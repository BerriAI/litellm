"""
Static checks on the shipped Dockerfile's runtime user.

The image is deployed into hardened Kubernetes clusters where
`securityContext.runAsNonRoot: true` is enforced. The kubelet validates
non-root status by parsing the image's USER field as an integer: a string
name like "nonroot" is rejected with CreateContainerConfigError because the
kubelet cannot resolve /etc/passwd inside the image at admission time.
"""

import os
import re

DOCKERFILE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "Dockerfile")


def _final_user_directive(dockerfile_text: str) -> str:
    """Return the uid of the last `USER` directive in the file (`USER uid` or `USER uid:gid`)."""
    matches = re.findall(r"^USER\s+(\S+)\s*$", dockerfile_text, re.MULTILINE)
    assert matches, "Dockerfile has no USER directive"
    return matches[-1].split(":", 1)[0]


def test_final_user_directive_is_numeric():
    """The runtime USER must be a numeric UID so kubelet's runAsNonRoot
    admission check (strconv.Atoi) succeeds."""
    with open(DOCKERFILE_PATH, "r", encoding="utf-8") as f:
        contents = f.read()

    final_user = _final_user_directive(contents)

    assert final_user.isdigit(), (
        f"Dockerfile final USER is {final_user!r}; must be a numeric UID "
        "so Kubernetes' runAsNonRoot admission check can verify non-root status. "
        "See https://kubernetes.io/docs/tasks/configure-pod-container/security-context/"
    )

    assert int(final_user) != 0, f"Dockerfile final USER is {final_user} (root); the image must run as a non-zero UID."
