from litellm.proxy.guardrails.guardrail_registry import (
    get_guardrail_initializer_from_hooks,
)


def test_get_guardrail_initializer_from_hooks():
    initializers = get_guardrail_initializer_from_hooks()
    print(f"initializers: {initializers}")
    assert "aim" in initializers


def test_guardrail_class_registry():
    from litellm.proxy.guardrails.guardrail_registry import guardrail_class_registry

    print(f"guardrail_class_registry: {guardrail_class_registry}")
    assert "aim" in guardrail_class_registry
    assert "aporia" in guardrail_class_registry
