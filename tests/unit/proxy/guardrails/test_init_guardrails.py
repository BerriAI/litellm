import json
from typing import Final, Literal
from unittest.mock import MagicMock, patch

import pytest


from litellm.proxy.guardrails.guardrail_hooks.custom_code.custom_code_guardrail import CustomCodeCompilationError
from litellm.proxy.guardrails.guardrail_registry import InMemoryGuardrailHandler
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2
from litellm.types.guardrails import Mode, SupportedGuardrailIntegrations


def test_init_guardrails_v2_registers_panw_mcp_output_scanner(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm
    from litellm.proxy.guardrails import guardrail_registry
    from litellm.proxy.guardrails.guardrail_hooks.panw_prisma_airs import PanwPrismaAirsHandler
    from litellm.types.guardrails import GuardrailEventHooks

    monkeypatch.setenv("LITELLM_STRICT_GUARDRAIL_MODES", "true")
    monkeypatch.setattr(guardrail_registry, "IN_MEMORY_GUARDRAIL_HANDLER", InMemoryGuardrailHandler())
    init_guardrails_v2(
        all_guardrails=[
            {
                "guardrail_name": "panw-mcp-output",
                "litellm_params": {
                    "guardrail": "panw_prisma_airs",
                    "mode": "post_mcp_call",
                    "default_on": True,
                    "api_key": "test-panw-key",
                    "profile_name": "test-profile",
                },
            }
        ]
    )
    scanners: Final = tuple(
        callback
        for callback in litellm.callbacks
        if isinstance(callback, PanwPrismaAirsHandler) and callback.guardrail_name == "panw-mcp-output"
    )
    assert len(scanners) == 1, "PANW MCP output scanning must be registered at startup"
    assert scanners[0].should_run_guardrail({}, GuardrailEventHooks.post_mcp_call) is True
    assert scanners[0].should_run_guardrail({}, GuardrailEventHooks.post_call) is False


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
        OPTIONAL_PresidioPIIMasking,
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
        if isinstance(callback, OPTIONAL_PresidioPIIMasking) and callback.guardrail_name == "test_presidio_chunk_size"
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
        ({"tags": {}}, None, False),
        ("pre_mcp_call", "both", True),
        ("pre_mcp_call", "output", True),
        ("pre_call", None, True),
    ],
)
async def test_initialize_presidio_mcp_only_mode_skips_post_call_output_scan(
    mode, filter_scope, expect_output_scanned, monkeypatch
):
    """Regression: an MCP-only Presidio guardrail used to also scan the LLM
    response on post_call, so a blocked MCP tool call that the model repeated in
    its answer turned the whole request into an HTTP 400 instead of a 200."""
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.guardrails import GuardrailEventHooks
    from litellm.types.utils import Choices, Message, ModelResponse

    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
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


def test_init_guardrails_v2_stops_boot_when_a_default_on_guardrail_cannot_be_initialized():
    """
    BerriAI/litellm#45028: a `default_on` guardrail whose type this release does not know
    (a config written for a newer LiteLLM, for example) used to be skipped with one error
    line, and the proxy started without it. For a guardrail that is meant to run on every
    request that is fail-open: every request reaches the provider carrying whatever the
    guardrail was configured to withhold. Initialization of a default_on guardrail is fatal.
    """
    from litellm.proxy.guardrails.guardrail_registry import IN_MEMORY_GUARDRAIL_HANDLER

    IN_MEMORY_GUARDRAIL_HANDLER.IN_MEMORY_GUARDRAILS.clear()
    IN_MEMORY_GUARDRAIL_HANDLER.guardrail_id_to_custom_guardrail.clear()

    all_guardrails = [
        {
            "guardrail_name": "pii-redaction",
            "litellm_params": {
                "guardrail": "a_guardrail_type_this_release_does_not_know",
                "mode": "pre_call",
                "default_on": True,
            },
        },
    ]

    with pytest.raises(ValueError, match="'pii-redaction' is default_on and could not be initialized"):
        init_guardrails_v2(all_guardrails=all_guardrails)


@pytest.mark.parametrize("default_on", [True, "true", "True", 1, "1", "yes"])
def test_default_on_is_read_the_way_litellm_params_would_parse_it(default_on):
    """The config value reaches this check raw, before LitellmParams parses it. Every spelling
    pydantic accepts as true must stop the boot too; `default_on: "true"` from a YAML string
    is as required as `default_on: true`."""
    from litellm.proxy.guardrails.guardrail_registry import IN_MEMORY_GUARDRAIL_HANDLER

    IN_MEMORY_GUARDRAIL_HANDLER.IN_MEMORY_GUARDRAILS.clear()
    IN_MEMORY_GUARDRAIL_HANDLER.guardrail_id_to_custom_guardrail.clear()

    all_guardrails = [
        {
            "guardrail_name": "pii-redaction",
            "litellm_params": {
                "guardrail": "a_guardrail_type_this_release_does_not_know",
                "mode": "pre_call",
                "default_on": default_on,
            },
        },
    ]

    with pytest.raises(ValueError, match="is default_on and could not be initialized"):
        init_guardrails_v2(all_guardrails=all_guardrails)


def test_default_on_is_read_from_a_parsed_litellm_params_object_too():
    """A config entry can arrive with `litellm_params` already parsed into a model, not a dict;
    the attribute is read the same way. `None` and an unknown spelling count as not default_on."""
    from litellm.proxy.guardrails.init_guardrails import _is_default_on
    from litellm.types.guardrails import LitellmParams

    parsed = LitellmParams(guardrail="presidio", mode="pre_call", default_on=True)
    assert _is_default_on({"litellm_params": parsed}) is True
    assert _is_default_on({"litellm_params": LitellmParams(guardrail="presidio", mode="pre_call")}) is False
    assert _is_default_on({"litellm_params": {"default_on": None}}) is False
    assert _is_default_on({"litellm_params": {"default_on": "sometimes"}}) is False
    assert _is_default_on({}) is False


def test_init_guardrails_v2_still_skips_an_optional_guardrail_it_cannot_initialize():
    """
    The skip (#34940) stays for guardrails that are not default_on: nothing runs them
    unless a request names them, so starting without one withholds nothing by default.
    The healthy guardrail beside it is still registered.
    """
    from litellm.proxy.guardrails.guardrail_registry import IN_MEMORY_GUARDRAIL_HANDLER

    IN_MEMORY_GUARDRAIL_HANDLER.IN_MEMORY_GUARDRAILS.clear()
    IN_MEMORY_GUARDRAIL_HANDLER.guardrail_id_to_custom_guardrail.clear()

    all_guardrails = [
        {
            "guardrail_name": "optional-unknown",
            "litellm_params": {
                "guardrail": "a_guardrail_type_this_release_does_not_know",
                "mode": "pre_call",
                "default_on": False,
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
    assert "optional-unknown" not in guardrail_names
    assert "healthy_presidio" in guardrail_names


def test_init_guardrails_v2_stops_boot_when_a_custom_code_guardrail_does_not_compile():
    from litellm.proxy.guardrails.guardrail_registry import IN_MEMORY_GUARDRAIL_HANDLER

    IN_MEMORY_GUARDRAIL_HANDLER.IN_MEMORY_GUARDRAILS.clear()
    IN_MEMORY_GUARDRAIL_HANDLER.guardrail_id_to_custom_guardrail.clear()

    all_guardrails = [
        {
            "guardrail_name": "custom-code-without-apply-guardrail",
            "litellm_params": {
                "guardrail": SupportedGuardrailIntegrations.CUSTOM_CODE.value,
                "mode": "pre_call",
                "custom_code": "x = 1\n",
            },
        },
    ]

    with pytest.raises(CustomCodeCompilationError, match="apply_guardrail"):
        init_guardrails_v2(all_guardrails=all_guardrails)


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,tags,restore,scope,tokens,expected,expected_calls",
    [
        ({"tags": {"team:mcp": "pre_mcp_call"}, "default": "pre_call"}, ["team:mcp"], False, None, {}, "raw", 0),
        ({"tags": {"team:mcp": "pre_mcp_call"}, "default": "pre_call"}, ["other"], False, None, {}, "masked", 1),
        ({"tags": {"team:mcp": "pre_mcp_call"}, "default": "pre_call"}, [], False, None, {}, "masked", 1),
        ({"tags": {"team:mcp": "pre_mcp_call"}}, [], False, None, {}, "raw", 0),
        ("pre_mcp_call", [], True, None, {}, "raw", 1),
        ("pre_mcp_call", [], True, None, {"restored": "twice", "raw": "restored"}, "restored", 1),
        ("pre_mcp_call", [], False, "output", {"raw": "restored"}, "masked", 1),
        ({"tags": {"team:mcp": "pre_mcp_call"}}, ["team:mcp"], False, "output", {}, "masked", 1),
        ({"tags": {"team:mcp": "pre_mcp_call"}}, [], False, "output", {}, "raw", 0),
    ],
)
async def test_presidio_initialized_output_dispatch(
    mode: str | list[str] | Mode,
    tags: list[str],
    restore: bool,
    scope: Literal["input", "output", "both"] | None,
    tokens: dict[str, str],
    expected: str,
    expected_calls: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typing import Final

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import UnifiedLLMGuardrails
    from litellm.proxy.guardrails.guardrail_initializers import initialize_presidio
    from litellm.types.guardrails import GuardrailEventHooks, LitellmParams
    from litellm.types.utils import Choices, Message, ModelResponse

    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    params: Final = LitellmParams(
        guardrail="presidio",
        mode=mode,
        default_on=True,
        output_parse_pii=restore,
        presidio_filter_scope=scope,
        presidio_analyzer_api_base="https://example.invalid/analyze",
        presidio_anonymizer_api_base="https://example.invalid/anonymize",
        mock_redacted_text={"text": "masked", "items": []},
    )
    callbacks: Final = initialize_presidio(params, {"guardrail_name": "output_dispatch"})
    data: Final = {"metadata": {"tags": tags, "pii_tokens": tokens}}
    response: Final = ModelResponse(choices=[Choices(message=Message(role="assistant", content="raw"), index=0)])
    selected: Final = tuple(
        callback for callback in callbacks if callback.should_run_guardrail(data, GuardrailEventHooks.post_call)
    )
    for callback in selected:
        data["guardrail_to_apply"] = callback
        await UnifiedLLMGuardrails().async_post_call_success_hook(
            data, UserAPIKeyAuth(request_route="/v1/chat/completions"), response
        )
    assert response.choices[0].message.content == expected
    assert len(selected) == expected_calls


def test_init_guardrails_v2_publishes_initialized_guardrail_to_the_proxy_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm
    from litellm.proxy import proxy_server
    from litellm.proxy.guardrails import guardrail_registry
    from litellm.proxy.guardrails.guardrail_hooks.aporia_ai import AporiaGuardrail

    router: Final = litellm.Router(model_list=[])
    monkeypatch.setattr(guardrail_registry, "IN_MEMORY_GUARDRAIL_HANDLER", InMemoryGuardrailHandler())
    monkeypatch.setattr(proxy_server, "llm_router", router)

    init_guardrails_v2(
        all_guardrails=[
            {
                "guardrail_name": "aporia-guard",
                "guardrail_id": "aporia-1",
                "litellm_params": {
                    "guardrail": "aporia",
                    "mode": "during_call",
                    "default_on": True,
                    "api_key": "aporia-key",
                    "api_base": "https://aporia.example.test/project-1",
                },
            }
        ]
    )

    (registered,) = (callback for callback in litellm.callbacks if isinstance(callback, AporiaGuardrail))
    assert router.get_available_guardrail("aporia-guard") == {
        "guardrail_name": "aporia-guard",
        "litellm_params": {
            "guardrail": "aporia",
            "mode": "during_call",
            "api_key": "aporia-key",
            "api_base": "https://aporia.example.test/project-1",
        },
        "callback": registered,
        "id": "aporia-1",
    }
