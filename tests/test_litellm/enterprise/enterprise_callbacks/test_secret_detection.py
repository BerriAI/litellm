import pytest
from litellm_enterprise.enterprise_callbacks.secret_detection import _ENTERPRISE_SecretDetection


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
