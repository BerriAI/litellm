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

import litellm

import os
import subprocess
import time

import pytest
import requests

TEST_MASTER_KEY = "sk-1234"


def test_litellm_proxy_server_config_no_general_settings():
    # Install the litellm[proxy] package
    # Start the server
    try:
        subprocess.run(["pip", "install", "litellm[proxy]"])
        filepath = os.path.dirname(os.path.abspath(__file__))
        config_fp = f"{filepath}/test_configs/test_config.yaml"

        # Set DATABASE_URL environment variable
        os.environ["DATABASE_URL"] = os.getenv("TOXI_PROXY_DATABASE_URL")
        os.environ["LITELLM_MASTER_KEY"] = TEST_MASTER_KEY

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
        server_process.terminate()
        server_process.wait()
