"""
Tests for Key based logging callbacks

"""

import httpx
import pytest


@pytest.mark.asyncio()
async def test_key_logging_callbacks():
    """
    Create virtual key with a logging callback set on the key
    Call /key/health for the key -> it should be unhealthy
    """
    # Generate a key with logging callback
    generate_url = "http://0.0.0.0:4000/key/generate"
    generate_headers = {
        "Authorization": "Bearer sk-1234",
        "Content-Type": "application/json",
    }
    generate_payload = {
        "metadata": {
            "logging": [
                {
                    "callback_name": "gcs_bucket",
                    "callback_type": "success_and_failure",
                    "callback_vars": {
                        "gcs_bucket_name": "key-logging-project1",
                        "gcs_path_service_account": "bad-service-account",
                    },
                }
            ]
        }
    }

    async with httpx.AsyncClient() as client:
        generate_response = await client.post(
            generate_url, headers=generate_headers, json=generate_payload
        )

    assert generate_response.status_code == 200
    generate_data = generate_response.json()
    assert "key" in generate_data

    _key = generate_data["key"]

    # Check key health
    health_url = "http://localhost:4000/key/health"
    health_headers = {
        "Authorization": f"Bearer {_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        health_response = await client.post(health_url, headers=health_headers, json={})

    assert health_response.status_code == 200
    health_data = health_response.json()
    print("key_health_data", health_data)
    # Check the response format and content
    assert "key" in health_data
    assert "logging_callbacks" in health_data
    assert health_data["logging_callbacks"]["callbacks"] == ["gcs_bucket"]
    assert health_data["logging_callbacks"]["status"] == "unhealthy"
    assert (
        "Failed to load vertex credentials"
        in health_data["logging_callbacks"]["details"]
    )
