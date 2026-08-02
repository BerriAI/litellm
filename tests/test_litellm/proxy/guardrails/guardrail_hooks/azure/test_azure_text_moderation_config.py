from litellm.proxy.guardrails.guardrail_hooks.azure import initialize_guardrail
from litellm.proxy.guardrails.guardrail_hooks.azure.text_moderation import (
    AzureContentSafetyTextModerationGuardrail,
)
from litellm.types.guardrails import LitellmParams


def test_azure_text_moderation_initialize_forwards_provider_params():
    litellm_params = LitellmParams(
        guardrail="azure/text_moderations",
        mode=["pre_call", "post_call"],
        api_key="azure_text_moderation_api_key",
        api_base="azure_text_moderation_api_base",
        severity_threshold=2,
        categories=["Hate", "Sexual"],
        haltOnBlocklistHit=True,
        outputType="EightSeverityLevels",
    )

    guardrail = initialize_guardrail(
        litellm_params=litellm_params,
        guardrail={"guardrail_name": "azure_text_moderation"},
    )

    assert isinstance(guardrail, AzureContentSafetyTextModerationGuardrail)
    assert guardrail.severity_threshold == 2
    assert guardrail.optional_params_request_body["categories"] == ["Hate", "Sexual"]
    assert guardrail.optional_params_request_body["haltOnBlocklistHit"] is True
    assert guardrail.optional_params_request_body["outputType"] == "EightSeverityLevels"
