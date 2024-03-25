import sys, os, time
import traceback, asyncio
import pytest
import subprocess

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
