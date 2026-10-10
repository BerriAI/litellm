import os
# What this tests?
## Tests /spend endpoints.

import pytest
import aiohttp


async def get_predict_spend_logs(session):
    url = "http://0.0.0.0:4000/global/predict/spend/logs"
    headers = {"Authorization": f"Bearer {os.environ['LITELLM_MASTER_KEY']}", "Content-Type": "application/json"}
    data = {
        "data": [
            {
                "date": "2024-03-09",
                "spend": 200000,
                "api_key": "sk-test-mock-api-key-456",
            }
        ]
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


@pytest.mark.skip(reason="datetime in ci/cd gets set weirdly")
@pytest.mark.asyncio
async def test_get_predicted_spend_logs():
    """
    - Create key
    - Make call (makes sure it's in spend logs)
    - Get request id from logs
    """
    async with aiohttp.ClientSession() as session:
        result = await get_predict_spend_logs(session=session)
        print(result)

        assert "response" in result
        assert len(result["response"]) > 0
