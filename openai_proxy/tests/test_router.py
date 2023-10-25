#### What this tests ####
#    This tests calling batch_completions by running 100 messages together

import sys, os
import traceback, asyncio
import pytest
from fastapi.testclient import TestClient
from fastapi import Request
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from openai_proxy import app


def test_router_completion(): 
    client = TestClient(app)
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hey, how's it going?"}],
        "model_list": [{ # list of model deployments 
                "model_name": "gpt-3.5-turbo", # openai model name 
                "litellm_params": { # params for litellm completion/embedding call 
                    "model": "azure/chatgpt-v-2", 
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE")
                },
                "tpm": 240000,
                "rpm": 1800
            }, {
                "model_name": "gpt-3.5-turbo", # openai model name 
                "litellm_params": { # params for litellm completion/embedding call 
                    "model": "azure/chatgpt-functioncalling", 
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE")
                },
                "tpm": 240000,
                "rpm": 1800
            }, {
                "model_name": "gpt-3.5-turbo", # openai model name 
                "litellm_params": { # params for litellm completion/embedding call 
                    "model": "gpt-3.5-turbo", 
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "tpm": 1000000,
                "rpm": 9000
        }]
    }

    response = client.post("/router/completions", json=data)
    print(f"response: {response.text}")
    assert response.status_code == 200

    response_data = response.json()
    # Perform assertions on the response data
    assert isinstance(response_data['choices'][0]['message']['content'], str)

test_router_completion()
