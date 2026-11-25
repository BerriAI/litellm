import hashlib
import shutil
import subprocess

import pytest

from litellm import sandbox

DOCKER_IMAGE = "alpine:3.20"


def _docker_usable() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=20
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _docker_usable(), reason="docker daemon not available")
async def test_real_docker_roundtrip():
    box = sandbox.docker(DOCKER_IMAGE, workdir="/workspace")
    try:
        result = await box.run(["echo", "hello"], timeout=120)
        assert result.stdout.strip() == "hello"
        assert result.exit_code == 0

        await box.write("seed.txt", b"seed")
        await box.write("sub/out.txt", b"from host")
        assert await box.read("sub/out.txt") == b"from host"
        await box.write("node_modules/skip.js", b"x")

        assert await box.which("sh") is not None
        assert await box.which("definitely-not-a-binary-xyz") is None
        tmp = await box.tempdir()
        assert tmp.startswith("/")

        snap = await box.snapshot()
        assert snap == {
            "seed.txt": hashlib.sha256(b"seed").hexdigest(),
            "sub/out.txt": hashlib.sha256(b"from host").hexdigest(),
        }
    finally:
        await box.close()
