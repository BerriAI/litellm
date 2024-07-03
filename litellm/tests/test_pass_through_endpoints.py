import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds-the parent directory to the system path

import asyncio

import httpx

from litellm.proxy.proxy_server import app, initialize_pass_through_endpoints


# Mock the async_client used in the pass_through_request function
async def mock_request(*args, **kwargs):
    return httpx.Response(200, json={"message": "Mocked response"})


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_pass_through_endpoint(client, monkeypatch):
    # Mock the httpx.AsyncClient.request method
    monkeypatch.setattr("httpx.AsyncClient.request", mock_request)

    # Define a pass-through endpoint
    pass_through_endpoints = [
        {
            "path": "/test-endpoint",
            "target": "https://api.example.com/v1/chat/completions",
            "headers": {"Authorization": "Bearer test-token"},
        }
    ]

    # Initialize the pass-through endpoint
    await initialize_pass_through_endpoints(pass_through_endpoints)

    # Make a request to the pass-through endpoint
    response = client.post("/test-endpoint", json={"prompt": "Hello, world!"})

    # Assert the response
    assert response.status_code == 200
    assert response.json() == {"message": "Mocked response"}


@pytest.mark.asyncio
async def test_pass_through_endpoint_rerank(client):
    _cohere_api_key = os.environ.get("COHERE_API_KEY")

    # Define a pass-through endpoint
    pass_through_endpoints = [
        {
            "path": "/v1/rerank",
            "target": "https://api.cohere.com/v1/rerank",
            "headers": {"Authorization": f"bearer {_cohere_api_key}"},
        }
    ]

    # Initialize the pass-through endpoint
    await initialize_pass_through_endpoints(pass_through_endpoints)

    _json_data = {
        "model": "rerank-english-v3.0",
        "query": "What is the capital of the United States?",
        "top_n": 3,
        "documents": [
            "Carson City is the capital city of the American state of Nevada."
        ],
    }

    # Make a request to the pass-through endpoint
    response = client.post("/v1/rerank", json=_json_data)

    print("JSON response: ", _json_data)

    # Assert the response
    assert response.status_code == 200
