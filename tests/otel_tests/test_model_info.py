"""
/model/info test
"""

import httpx
import pytest


@pytest.mark.asyncio()
async def test_custom_model_supports_vision():
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:4000/model/info",
            headers={"Authorization": "Bearer sk-1234"},
        )
        assert response.status_code == 200

        data = response.json()["data"]

        print("response from /model/info", data)
        llava_model = next(
            (model for model in data if model["model_name"] == "llava-hf"), None
        )

        assert llava_model is not None, "llava-hf model not found in response"
        assert (
            llava_model["model_info"]["supports_vision"] == True
        ), "llava-hf model should support vision"
