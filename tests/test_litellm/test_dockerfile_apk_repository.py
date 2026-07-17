"""
Static checks on the root Dockerfile's apk repository configuration.

The base image (cgr.dev/chainguard/wolfi-base) only configures the
authenticated Chainguard apk repo (https://apk.cgr.dev/chainguard) in
/etc/apk/repositories, which requires a Chainguard enterprise subscription.
Anyone pulling the published litellm image and running `apk add` inside it
hits SSL/auth failures with no fallback repo configured, so nothing can be
installed. See https://github.com/BerriAI/litellm/issues/33518
"""

import os
import re

import pytest

DOCKERFILE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "Dockerfile",
)


def _runtime_stage(dockerfile_text: str) -> str:
    """Return the contents of the final `FROM ... AS runtime` build stage."""
    match = re.search(
        r"^FROM .*\bAS runtime\b(.*)\Z", dockerfile_text, re.MULTILINE | re.DOTALL
    )
    assert match, "Dockerfile has no `FROM ... AS runtime` stage"
    return match.group(1)


@pytest.mark.skipif(
    not os.path.exists(DOCKERFILE_PATH),
    reason="Dockerfile not present in this checkout",
)
def test_runtime_stage_adds_public_wolfi_repo():
    """The runtime stage must add the public Wolfi apk repo so `apk add`
    works for users without a Chainguard enterprise subscription."""
    with open(DOCKERFILE_PATH, "r", encoding="utf-8") as f:
        contents = f.read()

    runtime_stage = _runtime_stage(contents)

    assert re.search(
        r'echo\s+"https://packages\.wolfi\.dev/os"\s*>>\s*/etc/apk/repositories',
        runtime_stage,
    ), (
        "Dockerfile's runtime stage doesn't add the public Wolfi apk repo "
        "(https://packages.wolfi.dev/os) to /etc/apk/repositories. Without it, "
        "`apk add` inside the published image only has the authenticated "
        "Chainguard repo available, which fails for anyone without an "
        "enterprise subscription."
    )
