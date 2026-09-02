"""Image-level check that the built proxy image can import the Bedrock realtime SDK.

Bedrock Nova Sonic (`/v1/realtime`) imports `aws_sdk_bedrock_runtime` lazily on the
first session, so an image whose `uv sync` stages skip the `bedrock-realtime` extra
boots, passes health checks, and then fails every Nova Sonic session with
"Missing aws_sdk_bedrock_runtime". Importing inside the built image is what catches
that class of regression (missing extra, lockfile drift, a stage that syncs a
different set of extras), which a static Dockerfile check cannot.

Gated on LITELLM_IMAGE like the other image checks in this directory; exercised
where an image has been built (the image-scan workflow). Requires a working docker CLI.
"""

import os
import shutil
import subprocess
from typing import Final

import pytest

IMAGE: Final = os.getenv("LITELLM_IMAGE")
NON_ROOT_UID: Final = "12345:0"
IMPORT_PROBE: Final = "import aws_sdk_bedrock_runtime, smithy_aws_core; print('bedrock-realtime ok')"

pytestmark = [
    pytest.mark.skipif(IMAGE is None, reason="requires a built image (set LITELLM_IMAGE)"),
    pytest.mark.skipif(shutil.which("docker") is None, reason="requires the docker CLI"),
]


def test_image_imports_bedrock_realtime_sdk():
    assert IMAGE is not None

    probe: Final = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            NON_ROOT_UID,
            "--entrypoint",
            "python",
            IMAGE,
            "-c",
            IMPORT_PROBE,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert probe.returncode == 0 and "bedrock-realtime ok" in probe.stdout, (
        f"{IMAGE} cannot import aws_sdk_bedrock_runtime as uid {NON_ROOT_UID}, so Bedrock Nova Sonic "
        "/v1/realtime sessions fail with 'Missing aws_sdk_bedrock_runtime'. Is `--extra bedrock-realtime` "
        f"passed to every `uv sync` in its Dockerfile?\nstdout:\n{probe.stdout}\nstderr:\n{probe.stderr}"
    )
