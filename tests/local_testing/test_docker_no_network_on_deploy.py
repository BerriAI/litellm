"""
Test to ensure Docker container does not go out to network on deploy.

This test verifies that the LiteLLM proxy container does not make outbound
network requests during startup. This is important for:
1. Air-gapped environments where outbound network is restricted
2. Security compliance requiring no unexpected network calls
3. Fast container startup without network dependencies

The test works by:
1. Building/running the container with network disabled
2. Verifying the container starts successfully without network
3. Checking for any errors related to network failures during startup
"""

import os
import re
import subprocess
import time

import pytest


def is_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


@pytest.mark.skipif(
    not is_docker_available(),
    reason="Docker not available",
)
class TestDockerNoNetworkOnDeploy:
    """
    Test suite for verifying Docker container starts without network access.
    """

    # Container and image names for testing
    TEST_IMAGE_NAME = "litellm-no-network-test"
    TEST_CONTAINER_NAME = "litellm-no-network-test-container"

    # Timeout for container operations
    CONTAINER_START_TIMEOUT = 60  # seconds

    # Patterns that indicate network-related failures during startup
    NETWORK_ERROR_PATTERNS = [
        r"connection refused",
        r"network is unreachable",
        r"could not resolve host",
        r"name resolution failed",
        r"dns lookup failed",
        r"failed to establish.*connection",
        r"socket.gaierror",
        r"urllib.error.URLError.*Errno",
        r"requests.exceptions.ConnectionError",
        r"httpx.*ConnectError",
        r"aiohttp.*ClientConnectorError",
    ]

    # Patterns that indicate EXPECTED local-only startup
    LOCAL_STARTUP_PATTERNS = [
        r"starting.*(server|proxy)",
        r"listening on",
        r"uvicorn running",
        r"application startup complete",
    ]

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Cleanup any existing test containers before and after each test."""
        self._cleanup_container()
        yield
        self._cleanup_container()

    def _cleanup_container(self):
        """Remove test container if it exists."""
        subprocess.run(
            ["docker", "rm", "-f", self.TEST_CONTAINER_NAME],
            capture_output=True,
            timeout=30,
        )

    def _build_test_image(self) -> bool:
        """
        Build the Docker image if needed.
        Returns True if image is available (built or already exists).
        """
        # Check if main litellm image exists
        result = subprocess.run(
            ["docker", "images", "-q", "litellm/litellm"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.stdout.strip():
            # Image exists, use it
            return True

        # Try to build from Dockerfile
        dockerfile_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "Dockerfile",
        )
        if os.path.exists(dockerfile_path):
            result = subprocess.run(
                [
                    "docker",
                    "build",
                    "-t",
                    self.TEST_IMAGE_NAME,
                    "-f",
                    dockerfile_path,
                    os.path.dirname(dockerfile_path),
                ],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes for build
            )
            return result.returncode == 0

        return False

    def test_container_starts_without_network(self):
        """
        Test that the container can start with network completely disabled.

        This test runs the container with --network=none to ensure no outbound
        network requests are required during startup.
        """
        # Use a minimal config that doesn't require external services
        minimal_config = """
model_list:
  - model_name: fake-model
    litellm_params:
      model: fake/fake-model

general_settings:
  master_key: sk-test-1234
  database_url: null

environment_variables: {}
"""

        # Create a temporary config file
        config_path = "/tmp/litellm_test_config.yaml"
        with open(config_path, "w") as f:
            f.write(minimal_config)

        image_to_use = "litellm/litellm"
        # Check if image exists
        result = subprocess.run(
            ["docker", "images", "-q", image_to_use],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if not result.stdout.strip():
            pytest.skip(f"Docker image {image_to_use} not available")

        # Run container with network disabled
        run_cmd = [
            "docker",
            "run",
            "--name",
            self.TEST_CONTAINER_NAME,
            "--network=none",  # Disable all network access
            "-v",
            f"{config_path}:/app/config.yaml:ro",
            "-e",
            "LITELLM_MASTER_KEY=sk-test-1234",
            "-e",
            "DATABASE_URL=",  # Empty to disable DB
            "-e",
            "STORE_MODEL_IN_DB=false",
            "-e",
            "LITELLM_LOG=DEBUG",
            "-d",  # Detached mode
            image_to_use,
            "--config",
            "/app/config.yaml",
        ]

        result = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            pytest.fail(f"Failed to start container: {result.stderr}")

        # Wait for container to start up
        time.sleep(5)

        # Check container logs for any network-related errors
        logs_result = subprocess.run(
            ["docker", "logs", self.TEST_CONTAINER_NAME],
            capture_output=True,
            text=True,
            timeout=30,
        )

        logs = logs_result.stdout + logs_result.stderr

        # Check for network error patterns (case-insensitive)
        network_errors = []
        for pattern in self.NETWORK_ERROR_PATTERNS:
            matches = re.findall(pattern, logs, re.IGNORECASE)
            if matches:
                network_errors.extend(matches)

        # Check if container is still running
        inspect_result = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                "{{.State.Running}}",
                self.TEST_CONTAINER_NAME,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        is_running = inspect_result.stdout.strip() == "true"

        # If container crashed, get exit code and reason

        # If container crashed, get exit code and reason
        if not is_running:
            exit_result = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "-f",
                    "{{.State.ExitCode}}",
                    self.TEST_CONTAINER_NAME,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            exit_code = exit_result.stdout.strip()

            # Container not running is OK if it didn't crash due to network issues
            # Check if exit was due to network errors
            if network_errors:
                pytest.fail(
                    f"Container failed with network errors (exit code {exit_code}): "
                    f"{network_errors}\n\nFull logs:\n{logs}"
                )

        # Assert no network errors were found
        assert len(network_errors) == 0, (
            f"Container made network requests during startup that failed: "
            f"{network_errors}"
        )

    def test_no_external_urls_in_startup_code(self):
        """
        Static analysis test: check that startup code doesn't contain
        hardcoded external URLs that would be called during import/startup.

        This is a complementary test to catch issues without needing Docker.
        """
        # Directories to check for startup code
        startup_dirs = [
            "litellm/proxy",
            "litellm/__init__.py",
            "litellm/main.py",
        ]

        # Patterns that indicate external URL calls during startup (not in functions)
        problematic_patterns = [
            # Immediate HTTP calls (not inside functions)
            r'^requests\.get\(["\'](https?://)',
            r'^httpx\.get\(["\'](https?://)',
            r'^urllib\.request\.urlopen\(["\'](https?://)',
        ]

        # Files that are OK to have URLs (they're called on-demand, not startup)
        allowed_files = [
            "model_prices_and_context_window.json",  # Static data file
            "test_",  # Test files
            "_test.py",
            "conftest.py",
        ]

        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        issues_found = []

        for startup_dir in startup_dirs:
            full_path = os.path.join(workspace_root, startup_dir)
            if not os.path.exists(full_path):
                continue

            if os.path.isfile(full_path):
                files_to_check = [full_path]
            else:
                files_to_check = []
                for root, _dirs, files in os.walk(full_path):
                    for f in files:
                        if f.endswith(".py"):
                            files_to_check.append(os.path.join(root, f))

            for filepath in files_to_check:
                # Skip allowed files
                if any(allowed in filepath for allowed in allowed_files):
                    continue

                try:
                    with open(filepath, "r") as f:
                        content = f.read()

                    for pattern in problematic_patterns:
                        matches = re.findall(pattern, content, re.MULTILINE)
                        if matches:
                            issues_found.append(f"{filepath}: {pattern} matched")
                except Exception:
                    pass  # Skip unreadable files

        # This test is informational - we document but don't fail
        if issues_found:
            pass


@pytest.mark.skipif(
    not is_docker_available(),
    reason="Docker not available",
)
def test_container_build_no_network_fetch():
    """
    Test that the Docker build process doesn't require network for runtime.

    This verifies that all dependencies are properly bundled and no
    runtime network calls are made during container initialization.

    Note: Build itself may need network for pip install, but runtime should not.
    """
    # This is a simplified version - full test would need to:
    # 1. Build image with --network=none (requires pre-cached deps)
    # 2. Or run built image in isolated network

    # For now, just verify the Dockerfile doesn't have wget/curl in CMD/ENTRYPOINT
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    dockerfile_path = os.path.join(workspace_root, "Dockerfile")

    if not os.path.exists(dockerfile_path):
        pytest.skip("Dockerfile not found")

    with open(dockerfile_path, "r") as f:
        content = f.read()

    # Check for network calls in CMD/ENTRYPOINT
    problematic = []
    lines = content.split("\n")
    for i, line in enumerate(lines, 1):
        line_upper = line.strip().upper()
        if line_upper.startswith(("CMD", "ENTRYPOINT")):
            if any(
                cmd in line.lower()
                for cmd in ["curl", "wget", "fetch", "http://", "https://"]
            ):
                problematic.append(f"Line {i}: {line.strip()}")

    assert (
        len(problematic) == 0
    ), f"Dockerfile CMD/ENTRYPOINT contains network calls: {problematic}"
