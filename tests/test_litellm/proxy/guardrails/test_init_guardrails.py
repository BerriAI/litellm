import json
from unittest.mock import MagicMock, patch

import pytest


from litellm.proxy.guardrails.guardrail_registry import InMemoryGuardrailHandler
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2
from litellm.types.guardrails import SupportedGuardrailIntegrations


def test_initialize_presidio_guardrail():
    """
    Test that initialize_guardrail correctly uses registered initializers
    for presidio guardrail
    """
    # Setup test data for a non-custom guardrail (using Presidio as an example)
    test_guardrail = {
        "guardrail_name": "test_presidio_guardrail",
        "litellm_params": {
            "guardrail": SupportedGuardrailIntegrations.PRESIDIO.value,
            "mode": "pre_call",
            "presidio_analyzer_api_base": "https://fakelink.com/v1/presidio/analyze",
            "presidio_anonymizer_api_base": "https://fakelink.com/v1/presidio/anonymize",
        },
    }

    # Call the initialize_guardrail method
    guardrail_handler = InMemoryGuardrailHandler()
    result = guardrail_handler.initialize_guardrail(
        guardrail=test_guardrail,
    )

    assert result["guardrail_name"] == "test_presidio_guardrail"
    assert result["litellm_params"].guardrail == SupportedGuardrailIntegrations.PRESIDIO.value
    assert result["litellm_params"].mode == "pre_call"


def test_initialize_bedrock_forwards_chunk_budget_chars():
    """Regression: `chunk_budget_chars` set in config.yaml must reach the guardrail.

    The field lives on BedrockGuardrailConfigModel, so LitellmParams parsed it and the
    Admin UI rendered it, but initialize_bedrock enumerates its kwargs explicitly and
    dropped it. The setting validated and then silently did nothing. Asserting through
    initialize_guardrail rather than the constructor is the point: constructing
    BedrockGuardrail directly bypasses the only path a user can actually reach.
    """
    import litellm
    from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrail

    test_guardrail = {
        "guardrail_name": "test_bedrock_chunk_budget",
        "litellm_params": {
            "guardrail": SupportedGuardrailIntegrations.BEDROCK.value,
            "mode": "pre_call",
            "guardrailIdentifier": "test-guardrail",
            "guardrailVersion": "DRAFT",
            "chunk_budget_chars": 60_000,
        },
    }

    guardrail_handler = InMemoryGuardrailHandler()
    guardrail_handler.initialize_guardrail(guardrail=test_guardrail)

    initialized = [
        callback
        for callback in litellm.callbacks
        if isinstance(callback, BedrockGuardrail) and callback.guardrail_name == "test_bedrock_chunk_budget"
    ]
    assert initialized, "bedrock guardrail was not registered as a callback"
    assert initialized[-1].chunk_budget_chars == 60_000


def test_initialize_guardrail_preserves_guardrail_info():
    """
    Regression (LIT-2529): initialize_guardrail must carry guardrail_info into the
    stored in-memory Guardrail. Dropping it left the Guardrail Monitor's usage
    endpoints unable to render type/description for YAML-defined guardrails.
    """
    test_guardrail = {
        "guardrail_name": "test_presidio_with_info",
        "litellm_params": {
            "guardrail": SupportedGuardrailIntegrations.PRESIDIO.value,
            "mode": "pre_call",
            "presidio_analyzer_api_base": "https://fakelink.com/v1/presidio/analyze",
            "presidio_anonymizer_api_base": "https://fakelink.com/v1/presidio/anonymize",
        },
        "guardrail_info": {"type": "PII", "description": "masks PII"},
    }

    guardrail_handler = InMemoryGuardrailHandler()
    result = guardrail_handler.initialize_guardrail(guardrail=test_guardrail)

    assert result is not None
    assert result["guardrail_info"] == {"type": "PII", "description": "masks PII"}
    stored = guardrail_handler.IN_MEMORY_GUARDRAILS[result["guardrail_id"]]
    assert stored["guardrail_info"] == {"type": "PII", "description": "masks PII"}


@pytest.mark.parametrize(
    "config_value, expected",
    [(True, True), (False, False), (None, False)],
)
def test_initialize_guardrail_sets_run_in_parallel(config_value, expected):
    """run_in_parallel from litellm_params must reach the built guardrail instance."""
    litellm_params = {
        "guardrail": SupportedGuardrailIntegrations.PRESIDIO.value,
        "mode": "pre_call",
        "presidio_analyzer_api_base": "https://fakelink.com/v1/presidio/analyze",
        "presidio_anonymizer_api_base": "https://fakelink.com/v1/presidio/anonymize",
    }
    if config_value is not None:
        litellm_params["run_in_parallel"] = config_value

    guardrail_handler = InMemoryGuardrailHandler()
    result = guardrail_handler.initialize_guardrail(
        guardrail={"guardrail_name": "test_parallel_flag", "litellm_params": litellm_params},
    )

    custom_guardrail = guardrail_handler.guardrail_id_to_custom_guardrail[result["guardrail_id"]]
    assert custom_guardrail.run_in_parallel is expected


def test_initialize_presidio_forwards_analyze_chunk_size_bytes():
    """Regression (LIT-4785): `presidio_analyze_chunk_size_bytes` set in
    config.yaml must reach the guardrail instance. The field lives on
    PresidioConfigModel, so LitellmParams parses it, but initialize_presidio
    enumerates its constructor kwargs explicitly and would silently drop it.
    """
    import litellm
    from litellm.proxy.guardrails.guardrail_hooks.presidio import (
        _OPTIONAL_PresidioPIIMasking,
    )

    test_guardrail = {
        "guardrail_name": "test_presidio_chunk_size",
        "litellm_params": {
            "guardrail": SupportedGuardrailIntegrations.PRESIDIO.value,
            "mode": "pre_call",
            "presidio_analyzer_api_base": "https://fakelink.com/v1/presidio/analyze",
            "presidio_anonymizer_api_base": "https://fakelink.com/v1/presidio/anonymize",
            "presidio_analyze_chunk_size_bytes": 250_000,
        },
    }

    guardrail_handler = InMemoryGuardrailHandler()
    guardrail_handler.initialize_guardrail(guardrail=test_guardrail)

    initialized = [
        callback
        for callback in litellm.callbacks
        if isinstance(callback, _OPTIONAL_PresidioPIIMasking)
        and callback.guardrail_name == "test_presidio_chunk_size"
    ]
    assert initialized, "presidio guardrail was not registered as a callback"
    assert initialized[-1].presidio_analyze_chunk_size_bytes == 250_000


def test_init_guardrails_v2_skips_invalid_guardrail_instead_of_crashing_boot():
    """
    Regression: one guardrail with an invalid litellm_params combination (Lakera's
    on_flagged="inject_system_message" with mode="during_call", which LakeraAIGuardrail's
    __init__ rejects with ValueError) must not take down the entire proxy at startup.
    init_guardrails_v2 previously had no try/except around initialize_guardrail, so this
    ValueError propagated all the way through proxy_server.py's load_config and crashed
    the whole process, including every other, correctly-configured guardrail in the list.
    """
    from litellm.proxy.guardrails.guardrail_registry import IN_MEMORY_GUARDRAIL_HANDLER

    IN_MEMORY_GUARDRAIL_HANDLER.IN_MEMORY_GUARDRAILS.clear()
    IN_MEMORY_GUARDRAIL_HANDLER.guardrail_id_to_custom_guardrail.clear()

    all_guardrails = [
        {
            "guardrail_name": "broken_lakera_advisory",
            "litellm_params": {
                "guardrail": SupportedGuardrailIntegrations.LAKERA_V2.value,
                "mode": "during_call",
                "on_flagged": "inject_system_message",
                "api_key": "fake-key",
            },
        },
        {
            "guardrail_name": "healthy_presidio",
            "litellm_params": {
                "guardrail": SupportedGuardrailIntegrations.PRESIDIO.value,
                "mode": "pre_call",
                "presidio_analyzer_api_base": "https://fakelink.com/v1/presidio/analyze",
                "presidio_anonymizer_api_base": "https://fakelink.com/v1/presidio/anonymize",
            },
        },
    ]

    init_guardrails_v2(all_guardrails=all_guardrails)

    guardrail_names = {
        guardrail["guardrail_name"] for guardrail in IN_MEMORY_GUARDRAIL_HANDLER.IN_MEMORY_GUARDRAILS.values()
    }
    assert "broken_lakera_advisory" not in guardrail_names
    assert "healthy_presidio" in guardrail_names


def test_init_guardrails_v2_skips_guardrail_with_malformed_advisory_template():
    """
    Regression: a malformed advisory_system_message (missing the {reason} placeholder
    LakeraAIGuardrail's __init__ requires) is a second, independent trigger for the same
    uncaught-ValueError-crashes-boot root cause as the during_call+inject_system_message
    case above. Both must be caught by init_guardrails_v2, not just one.
    """
    from litellm.proxy.guardrails.guardrail_registry import IN_MEMORY_GUARDRAIL_HANDLER

    IN_MEMORY_GUARDRAIL_HANDLER.IN_MEMORY_GUARDRAILS.clear()
    IN_MEMORY_GUARDRAIL_HANDLER.guardrail_id_to_custom_guardrail.clear()

    all_guardrails = [
        {
            "guardrail_name": "broken_lakera_template",
            "litellm_params": {
                "guardrail": SupportedGuardrailIntegrations.LAKERA_V2.value,
                "mode": "pre_call",
                "on_flagged": "inject_system_message",
                "advisory_system_message": "This request was flagged, no placeholder here",
                "api_key": "fake-key",
            },
        },
        {
            "guardrail_name": "healthy_presidio",
            "litellm_params": {
                "guardrail": SupportedGuardrailIntegrations.PRESIDIO.value,
                "mode": "pre_call",
                "presidio_analyzer_api_base": "https://fakelink.com/v1/presidio/analyze",
                "presidio_anonymizer_api_base": "https://fakelink.com/v1/presidio/anonymize",
            },
        },
    ]

    init_guardrails_v2(all_guardrails=all_guardrails)

    guardrail_names = {
        guardrail["guardrail_name"] for guardrail in IN_MEMORY_GUARDRAIL_HANDLER.IN_MEMORY_GUARDRAILS.values()
    }
    assert "broken_lakera_template" not in guardrail_names
    assert "healthy_presidio" in guardrail_names
