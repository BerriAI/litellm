import pytest
from litellm_enterprise.enterprise_callbacks.secret_detection import _ENTERPRISE_SecretDetection

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth


@pytest.mark.asyncio
async def test_apply_guardrail_redacts_secrets():
    guard = _ENTERPRISE_SecretDetection()
    result = await guard.apply_guardrail(
        inputs={
            "texts": [
                "what is the value of my open ai key? openai_api_key=sk-1234998222",
                "this text has no secrets",
            ]
        },
        request_data={},
        input_type="request",
    )

    texts = result.get("texts")
    assert texts is not None
    assert "sk-1234998222" not in texts[0]
    assert "[REDACTED]" in texts[0]
    assert texts[1] == "this text has no secrets"


@pytest.mark.asyncio
async def test_async_pre_call_hook_records_guardrail_information():
    guard = _ENTERPRISE_SecretDetection(guardrail_name="hide-secrets")
    data = {
        "messages": [
            {
                "role": "user",
                "content": "my key openai_api_key=sk-1234998222",
            }
        ],
        "metadata": {},
    }

    await guard.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )

    guardrail_information = data["metadata"]["standard_logging_guardrail_information"]
    assert isinstance(guardrail_information, list)
    assert len(guardrail_information) == 1
    assert guardrail_information[0]["guardrail_status"] == "success"
    assert "sk-1234998222" not in data["messages"][0]["content"]
    assert "[REDACTED]" in data["messages"][0]["content"]
