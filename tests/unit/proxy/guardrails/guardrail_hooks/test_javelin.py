from unittest.mock import Mock, patch

import pytest

from litellm.proxy.guardrails.guardrail_hooks.javelin.javelin import JavelinGuardrail
from litellm.proxy.guardrails.guardrail_registry import InMemoryGuardrailHandler
from litellm.types.guardrails import GuardrailEventHooks


@pytest.mark.asyncio
async def test_config_without_api_version_calls_javelin_v1():
    handler = InMemoryGuardrailHandler()
    registered = handler.initialize_guardrail(
        guardrail={
            "guardrail_name": "javelin-no-api-version",
            "litellm_params": {
                "guardrail": "javelin",
                "mode": "pre_call",
                "api_key": "javelin_api_key",
                "api_base": "https://javelin.example",
                "guard_name": "trustsafety",
            },
        }
    )
    assert registered is not None
    guardrail = handler.guardrail_id_to_custom_guardrail[registered["guardrail_id"]]
    assert isinstance(guardrail, JavelinGuardrail)
    assessments = [{"trustsafety": {"request_reject": False}}]
    response = Mock()
    response.json.return_value = {"assessments": assessments}

    with patch.object(guardrail.async_handler, "post", return_value=response) as mock_post:
        result = await guardrail.call_javelin_guard(
            request={"input": {"text": "hello"}, "config": None, "metadata": None},
            event_type=GuardrailEventHooks.pre_call,
        )

    assert result == {"assessments": assessments}
    assert mock_post.call_args.kwargs["url"] == "https://javelin.example/v1/guardrail/trustsafety/apply"
