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
        pytest.fail(f"Error occurred: {e}. Installing litellm failed please retry")


def test_litellm_proxy_server():
    # Sync the local litellm[proxy] dependencies into the project environment
    subprocess.run(["uv", "sync", "--frozen", "--extra", "proxy"], check=True)

    # Import the proxy_server module
    try:
        import litellm.proxy.proxy_server
    except ImportError:
        pytest.fail("Failed to import litellm.proxy_server")

    # Assertion to satisfy the test, you can add other checks as needed
    assert True


def test_package_dependencies():
    """
    Test that all optional dependency entries are exposed via project optional-dependencies.
    """
    try:
        import pathlib
        import litellm
        from packaging.requirements import Requirement
        
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

        optional_deps = pyproject["project"]["optional-dependencies"]
        assert optional_deps, "Expected project.optional-dependencies to be defined"

        parsed_requirements = set()
        for extra_name, requirements in optional_deps.items():
            assert requirements, f"Optional dependency group '{extra_name}' is empty"
            for requirement in requirements:
                assert isinstance(
                    requirement, str
                ), f"Expected string requirement in extra '{extra_name}'"
                parsed = Requirement(requirement)
                parsed_requirements.add(parsed.name.lower())

        print(parsed_requirements)
        print(
            f"Validated {len(parsed_requirements)} optional dependencies across {len(optional_deps)} extras groups"
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
    # Sync the local litellm packages into the project environment
    server_process = None
    try:
        subprocess.run(
            ["uv", "sync", "--frozen", "--group", "proxy-dev", "--extra", "proxy", "--extra", "extra_proxy"],
            check=True,
        )
        
        # Ensure Prisma client is generated
        try:
            # Get the project root directory (where schema.prisma is located)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            print(f"Running prisma generate from: {project_root}")
            
            result = subprocess.run(
                ["uv", "run", "--no-sync", "prisma", "generate"], 
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

        # Allow some time for the server to start (increased for CI environments)
        time.sleep(90)  # Increased from 60s for slower CI runners

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
