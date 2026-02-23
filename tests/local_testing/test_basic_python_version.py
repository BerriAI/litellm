import asyncio
import os
import subprocess
import sys
import time
import traceback

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


def test_using_litellm():
    try:
        import litellm

        print("litellm imported successfully")
    except Exception as e:
        pytest.fail(
            f"Error occurred: {e}. Installing litellm on python3.8 failed please retry"
        )


def test_litellm_proxy_server():
    # Install the local litellm[proxy] package in development mode
    subprocess.run(["pip", "install", "-e", ".[proxy]"])

    # Import the proxy_server module
    try:
        import litellm.proxy.proxy_server
    except ImportError:
        pytest.fail("Failed to import litellm.proxy_server")

    # Assertion to satisfy the test, you can add other checks as needed
    assert True


def test_package_dependencies():
    """
    Test that all optional dependencies are correctly specified in extras.
    """
    try:
        import pathlib
        import litellm
        
        # Try to import tomllib (Python 3.11+) or tomli (older versions)
        try:
            import tomllib as tomli
        except ImportError:
            try:
                import tomli
            except ImportError:
                pytest.skip("tomli/tomllib not available - skipping dependency check")

        # Get the litellm package root path
        litellm_path = pathlib.Path(litellm.__file__).parent.parent
        pyproject_path = litellm_path / "pyproject.toml"

        # Read and parse pyproject.toml
        with open(pyproject_path, "rb") as f:
            pyproject = tomli.load(f)

        # Get all optional dependencies from poetry.dependencies
        poetry_deps = pyproject["tool"]["poetry"]["dependencies"]
        optional_deps = {
            name.lower()
            for name, value in poetry_deps.items()
            if isinstance(value, dict) and value.get("optional", False)
        }
        print(optional_deps)
        # Get all packages listed in extras
        extras = pyproject["tool"]["poetry"]["extras"]
        all_extra_deps = set()
        for extra_group in extras.values():
            all_extra_deps.update(dep.lower() for dep in extra_group)
        print(all_extra_deps)
        # Check that all optional dependencies are in some extras group
        missing_from_extras = optional_deps - all_extra_deps
        assert (
            not missing_from_extras
        ), f"Optional dependencies missing from extras: {missing_from_extras}"

        print(
            f"All {len(optional_deps)} optional dependencies are correctly specified in extras"
        )

    except Exception as e:
        pytest.fail(
            f"Error occurred while checking dependencies: {str(e)}\n"
            + traceback.format_exc()
        )


import os
import subprocess
import time

import pytest
import requests


def test_litellm_proxy_server_config_no_general_settings():
    # Install the local litellm packages in development mode
    server_process = None
    try:
        subprocess.run(["pip", "install", "-e", ".[proxy]"])
        subprocess.run(["pip", "install", "-e", ".[extra_proxy]"])
        
        # Ensure Prisma client is generated
        try:
            # Get the project root directory (where schema.prisma is located)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            print(f"Running prisma generate from: {project_root}")
            
            result = subprocess.run(
                ["prisma", "generate"], 
                capture_output=True, 
                text=True, 
                check=True,
                cwd=project_root
            )
            print(f"Prisma generate stdout: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Prisma generate failed: {e}")
            print(f"Prisma generate stderr: {e.stderr}")
            raise
        filepath = os.path.dirname(os.path.abspath(__file__))
        config_fp = f"{filepath}/test_configs/test_config_no_auth.yaml"
        server_process = subprocess.Popen(
            [
                "python",
                "-m",
                "litellm.proxy.proxy_cli",
                "--config",
                config_fp,
            ]
        )

        # Allow some time for the server to start
        time.sleep(60)  # Adjust the sleep time if necessary

        # Send a request to the /health/liveliness endpoint
        response = requests.get("http://localhost:4000/health/liveliness")

        # Check if the response is successful
        assert response.status_code == 200
        assert response.json() == "I'm alive!"

        # Test /chat/completions
        response = requests.post(
            "http://localhost:4000/chat/completions",
            headers={"Authorization": "Bearer 1234567890"},
            json={
                "model": "test_openai_models",
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
            },
        )

        assert response.status_code == 200

    except ImportError:
        pytest.fail("Failed to import litellm.proxy_server")
    except requests.ConnectionError:
        pytest.fail("Failed to connect to the server")
    finally:
        # Shut down the server
        if server_process:
            server_process.terminate()
            server_process.wait()

    # Additional assertions can be added here
    assert True
