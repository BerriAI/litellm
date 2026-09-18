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


def test_initialize_bedrock_forwards_contextual_grounding_from_messages():
    """`contextual_grounding_from_messages: true` in config.yaml must make the post-call
    payload carry the plain system prompt and user turn as grounding_source and query."""
    import litellm
    from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrail
    from litellm.types.utils import Choices, Message, ModelResponse

    test_guardrail = {
        "guardrail_name": "test_bedrock_grounding_from_messages",
        "litellm_params": {
            "guardrail": SupportedGuardrailIntegrations.BEDROCK.value,
            "mode": "post_call",
            "guardrailIdentifier": "test-guardrail",
            "guardrailVersion": "DRAFT",
            "contextual_grounding_from_messages": True,
        },
    }
    messages = [
        {"role": "system", "content": "Returns are accepted for 30 days."},
        {"role": "user", "content": "How long is the return window?"},
    ]
    response = ModelResponse(
        choices=[Choices(index=0, message=Message(role="assistant", content="30 days."), finish_reason="stop")]
    )
    expected_request = {
        "source": "OUTPUT",
        "content": [
            {"text": {"text": "Returns are accepted for 30 days.", "qualifiers": ["grounding_source"]}},
            {"text": {"text": "How long is the return window?", "qualifiers": ["query"]}},
            {"text": {"text": "30 days.", "qualifiers": ["guard_content"]}},
        ],
    }

    guardrail_handler = InMemoryGuardrailHandler()
    guardrail_handler.initialize_guardrail(guardrail=test_guardrail)

    initialized = [
        callback
        for callback in litellm.callbacks
        if isinstance(callback, BedrockGuardrail) and callback.guardrail_name == "test_bedrock_grounding_from_messages"
    ]
    assert initialized, "bedrock guardrail was not registered as a callback"
    actual_request = initialized[-1].convert_to_bedrock_format(source="OUTPUT", response=response, messages=messages)
    assert json.loads(json.dumps(actual_request)) == expected_request


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, filter_scope, expect_output_scanned",
    [
        ("pre_mcp_call", None, False),
        (["pre_mcp_call", "post_mcp_call"], None, False),
        ({"tags": {"team:mcp": "pre_mcp_call"}, "default": ["pre_mcp_call", "post_mcp_call"]}, None, False),
        ({"tags": {"team:mcp": ["pre_mcp_call"]}, "default": "pre_call"}, None, True),
        ({"tags": {}}, None, True),
        ("pre_mcp_call", "both", True),
        ("pre_mcp_call", "output", True),
        ("pre_call", None, True),
    ],
)
async def test_initialize_presidio_mcp_only_mode_skips_post_call_output_scan(mode, filter_scope, expect_output_scanned):
    """Regression: an MCP-only Presidio guardrail used to also scan the LLM
    response on post_call, so a blocked MCP tool call that the model repeated in
    its answer turned the whole request into an HTTP 400 instead of a 200."""
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.guardrails import GuardrailEventHooks
    from litellm.types.utils import Choices, Message, ModelResponse

    llm_answer = "Call me at 415-555-2671"
    litellm_params = {
        "guardrail": SupportedGuardrailIntegrations.PRESIDIO.value,
        "mode": mode,
        "presidio_analyzer_api_base": "https://fakelink.com/v1/presidio/analyze",
        "presidio_anonymizer_api_base": "https://fakelink.com/v1/presidio/anonymize",
        "mock_redacted_text": {"text": "Call me at <PHONE_NUMBER>", "items": []},
        "default_on": True,
    }
    if filter_scope is not None:
        litellm_params["presidio_filter_scope"] = filter_scope

    guardrail_handler = InMemoryGuardrailHandler()
    result = guardrail_handler.initialize_guardrail(
        guardrail={"guardrail_name": "test_presidio_mcp_scope", "litellm_params": litellm_params}
    )
    guardrail_id = result["guardrail_id"]
    callbacks = [
        guardrail_handler.guardrail_id_to_custom_guardrail[guardrail_id],
        *guardrail_handler.guardrail_id_to_sibling_callbacks[guardrail_id],
    ]

    request_data = {"metadata": {}}
    response = ModelResponse(
        choices=[Choices(message=Message(role="assistant", content=llm_answer), index=0, finish_reason="stop")]
    )
    for callback in callbacks:
        if callback.should_run_guardrail(data=request_data, event_type=GuardrailEventHooks.post_call):
            await callback.async_post_call_success_hook(
                data=request_data, user_api_key_dict=UserAPIKeyAuth(), response=response
            )

    assert (response.choices[0].message.content != llm_answer) is expect_output_scanned


@pytest.mark.parametrize(
    "config_value, expected",
    [(True, True), (False, False), (None, False)],
)
def test_initialize_guardrail_sets_scan_raw_request(config_value, expected):
    """scan_raw_request from litellm_params must reach the built guardrail instance,
    same wiring as run_in_parallel."""
    litellm_params = {
        "guardrail": SupportedGuardrailIntegrations.PRESIDIO.value,
        "mode": "pre_call",
        "presidio_analyzer_api_base": "https://fakelink.com/v1/presidio/analyze",
        "presidio_anonymizer_api_base": "https://fakelink.com/v1/presidio/anonymize",
    }
    if config_value is not None:
        litellm_params["scan_raw_request"] = config_value

    guardrail_handler = InMemoryGuardrailHandler()
    result = guardrail_handler.initialize_guardrail(
        guardrail={"guardrail_name": "test_scan_raw_request_flag", "litellm_params": litellm_params},
    )

    custom_guardrail = guardrail_handler.guardrail_id_to_custom_guardrail[result["guardrail_id"]]
    assert custom_guardrail.scan_raw_request is expected


def test_init_guardrails_v2_skips_invalid_guardrail_instead_of_crashing_boot():
    """
    Regression: one guardrail with an invalid litellm_params combination (Lakera's
    on_flagged="inject_system_message" with payload=False, which LakeraAIGuardrail's
    __init__ rejects with ValueError since masking can't happen without payload data)
    must not take down the entire proxy at startup. init_guardrails_v2 previously had
    no try/except around initialize_guardrail, so this ValueError propagated all the
    way through proxy_server.py's load_config and crashed the whole process, including
    every other, correctly-configured guardrail in the list.

    mode="during_call" + on_flagged="inject_system_message" is deliberately NOT used
    here anymore (maintainer finding on BerriAI/litellm#34940): that combination is
    now accepted at construction time, since async_moderation_hook already degrades
    it gracefully at runtime instead of needing a config-time rejection.
    """
    from litellm.proxy.guardrails.guardrail_registry import IN_MEMORY_GUARDRAIL_HANDLER

    IN_MEMORY_GUARDRAIL_HANDLER.IN_MEMORY_GUARDRAILS.clear()
    IN_MEMORY_GUARDRAIL_HANDLER.guardrail_id_to_custom_guardrail.clear()

    all_guardrails = [
        {
            "guardrail_name": "broken_lakera_advisory",
            "litellm_params": {
                "guardrail": SupportedGuardrailIntegrations.LAKERA_V2.value,
                "mode": "pre_call",
                "on_flagged": "inject_system_message",
                "payload": False,
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


def test_init_guardrails_v2_accepts_during_call_advisory_mode():
    """
    Maintainer finding on BerriAI/litellm#34940: on_flagged='inject_system_message'
    with mode='during_call' must construct successfully now -- async_moderation_hook
    already masks whatever's maskable and falls back to a log-only warning when the
    advisory itself can't be delivered, so rejecting this combination at config time
    disabled a guardrail that runtime already handles safely.
    """
    from litellm.proxy.guardrails.guardrail_registry import IN_MEMORY_GUARDRAIL_HANDLER

    IN_MEMORY_GUARDRAIL_HANDLER.IN_MEMORY_GUARDRAILS.clear()
    IN_MEMORY_GUARDRAIL_HANDLER.guardrail_id_to_custom_guardrail.clear()

    all_guardrails = [
        {
            "guardrail_name": "during_call_advisory",
            "litellm_params": {
                "guardrail": SupportedGuardrailIntegrations.LAKERA_V2.value,
                "mode": "during_call",
                "on_flagged": "inject_system_message",
                "api_key": "fake-key",
            },
        },
    ]

    init_guardrails_v2(all_guardrails=all_guardrails)

    guardrail_names = {
        guardrail["guardrail_name"] for guardrail in IN_MEMORY_GUARDRAIL_HANDLER.IN_MEMORY_GUARDRAILS.values()
    }
    assert "during_call_advisory" in guardrail_names


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
