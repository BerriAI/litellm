from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.guardrails.guardrail_registry import (
    get_guardrail_initializer_from_hooks,
    InMemoryGuardrailHandler,
)
from litellm.types.guardrails import GuardrailEventHooks, Guardrail, LitellmParams


def test_get_guardrail_initializer_from_hooks():
    initializers = get_guardrail_initializer_from_hooks()
    assert "aim" in initializers


def test_guardrail_class_registry():
    from litellm.proxy.guardrails.guardrail_registry import guardrail_class_registry

    assert "aim" in guardrail_class_registry
    assert "aporia" in guardrail_class_registry


def test_noma_registry_resolution():
    from litellm.proxy.guardrails.guardrail_hooks.noma.noma import NomaGuardrail
    from litellm.proxy.guardrails.guardrail_hooks.noma.noma_v2 import NomaV2Guardrail
    from litellm.proxy.guardrails.guardrail_registry import (
        guardrail_class_registry,
        guardrail_initializer_registry,
    )

    assert guardrail_class_registry["noma"] is NomaGuardrail
    assert guardrail_class_registry["noma_v2"] is NomaV2Guardrail
    assert "noma" in guardrail_initializer_registry
    assert "noma_v2" in guardrail_initializer_registry


def test_update_in_memory_guardrail():
    handler = InMemoryGuardrailHandler()
    handler.guardrail_id_to_custom_guardrail["123"] = CustomGuardrail(
        guardrail_name="test-guardrail",
        default_on=False,
        event_hook=GuardrailEventHooks.pre_call,
    )

    handler.update_in_memory_guardrail(
        "123",
        Guardrail(
            guardrail_name="test-guardrail",
            litellm_params=LitellmParams(
                guardrail="test-guardrail", mode="pre_call", default_on=True
            ),
        ),
    )

    assert (
        handler.guardrail_id_to_custom_guardrail["123"].should_run_guardrail(
            data={}, event_type=GuardrailEventHooks.pre_call
        )
        is True
    )
    assert (
        handler.guardrail_id_to_custom_guardrail["123"].event_hook
        is GuardrailEventHooks.pre_call
    )
