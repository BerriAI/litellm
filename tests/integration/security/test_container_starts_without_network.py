from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import eventually


def test_container_starts_without_network(tmp_path: Path) -> None:
    docker: Final = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker is not installed")

    image: Final = "litellm/litellm"
    image_check: Final = subprocess.run(
        [docker, "image", "inspect", image],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if image_check.returncode != 0:
        pytest.skip(f"Docker image {image} is not available")

    config: Final = tmp_path / "litellm-test-config.yaml"
    config.write_text(
        "model_list:\n"
        "  - model_name: fake-model\n"
        "    litellm_params:\n"
        "      model: fake/fake-model\n"
        "general_settings:\n"
        "  master_key: test-master-key\n"
        "  database_url: null\n"
        "environment_variables: {}\n"
    )
    name: Final = f"litellm-network-none-{uuid.uuid4().hex}"

    try:
        started: Final = subprocess.run(
            [
                docker,
                "run",
                "--name",
                name,
                "--network=none",
                "-v",
                f"{config}:/app/config.yaml:ro",
                "-e",
                "LITELLM_MASTER_KEY=test-master-key",
                "-e",
                "STORE_MODEL_IN_DB=false",
                "-e",
                "LITELLM_LOCAL_MODEL_COST_MAP=True",
                "-d",
                image,
                "--config",
                "/app/config.yaml",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert started.returncode == 0, started.stderr

        network_mode: Final = subprocess.run(
            [docker, "inspect", "--format={{.HostConfig.NetworkMode}}", name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert network_mode.returncode == 0, network_mode.stderr
        assert network_mode.stdout.strip() == "none"

        def readiness() -> str:
            result: Final = subprocess.run(
                [
                    docker,
                    "exec",
                    name,
                    "python",
                    "-c",
                    "import urllib.request; "
                    "print(urllib.request.urlopen('http://127.0.0.1:4000/health/readiness', timeout=2).status)",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else result.stderr

        status: Final = eventually(readiness, lambda result: result == "200", seconds=30)
        assert status == "200", f"Container did not become ready without network: {status}"
    finally:
        subprocess.run(
            [docker, "rm", "-f", name],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
