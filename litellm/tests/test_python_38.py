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
    # Install the litellm[proxy] package
    subprocess.run(["pip", "install", "litellm[proxy]"])

    # Import the proxy_server module
    try:
        import litellm.proxy.proxy_server
    except ImportError:
        pytest.fail("Failed to import litellm.proxy_server")

    # Assertion to satisfy the test, you can add other checks as needed
    assert True


import os
import subprocess
import time

import pytest
import requests


def test_litellm_proxy_server_config_no_general_settings():
    # Install the litellm[proxy] package
    # Start the server
    try:
        subprocess.run(["pip", "install", "litellm[proxy]"])
        subprocess.run(["pip", "install", "litellm[extra_proxy]"])
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
    except ImportError:
        pytest.fail("Failed to import litellm.proxy_server")
    except requests.ConnectionError:
        pytest.fail("Failed to connect to the server")
    finally:
        # Shut down the server
        server_process.terminate()
        server_process.wait()

    # Additional assertions can be added here
    assert True
