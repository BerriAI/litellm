import sys, os, time
import traceback
from dotenv import load_dotenv

load_dotenv()
import os, io

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest, logging, requests
import litellm
from litellm import embedding, completion, completion_cost, Timeout
from litellm import RateLimitError


def test_add_new_key():
    try:
        # Your test data
        test_data = {
            "models": ["gpt-3.5-turbo", "gpt-4", "claude-2", "azure-model"],
            "aliases": {"mistral-7b": "gpt-3.5-turbo"},
            "duration": "20m",
        }
        print("testing proxy server")
        # Your bearer token
        token = os.getenv("PROXY_MASTER_KEY")

        headers = {"Authorization": f"Bearer {token}"}

        staging_endpoint = "https://litellm-litellm-pr-1366.up.railway.app"

        # Your bearer token
        token = os.getenv("PROXY_MASTER_KEY")

        headers = {"Authorization": f"Bearer {token}"}

        # Make a request to the staging endpoint
        response = requests.post(
            staging_endpoint + "/key/generate", json=test_data, headers=headers
        )

        print(f"response: {response.text}")
        assert response.status_code == 200
        result = response.json()
    except Exception as e:
        print(traceback.format_exc())
        pytest.fail(f"An error occurred {e}")


# test_add_new_key()
