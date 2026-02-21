from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.guardrails.guardrail_registry import (
    get_guardrail_initializer_from_hooks,
    InMemoryGuardrailHandler,
    GuardrailRegistry,
)
from litellm.types.guardrails import GuardrailEventHooks, Guardrail, LitellmParams


def test_get_guardrail_initializer_from_hooks():
    initializers = get_guardrail_initializer_from_hooks()
    print(f"initializers: {initializers}")
    assert "aim" in initializers


def test_guardrail_class_registry():
    from litellm.proxy.guardrails.guardrail_registry import guardrail_class_registry

    print(f"guardrail_class_registry: {guardrail_class_registry}")
    assert "aim" in guardrail_class_registry
    assert "aporia" in guardrail_class_registry


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


@pytest.mark.asyncio
async def test_get_guardrail_by_name_allows_duplicates():
    registry = GuardrailRegistry()
    table = SimpleNamespace(
        find_first=AsyncMock(
            return_value={
                "guardrail_id": "id-1",
                "guardrail_name": "duplicate-name",
                "litellm_params": {},
                "guardrail_info": {"source": "test"},
            }
        )
    )
    prisma_client = SimpleNamespace(db=SimpleNamespace(litellm_guardrailstable=table))

    guardrail = await registry.get_guardrail_by_name_from_db(
        "duplicate-name", prisma_client
    )

    table.find_first.assert_awaited_once_with(where={"guardrail_name": "duplicate-name"})
    assert guardrail["guardrail_id"] == "id-1"
    assert guardrail["guardrail_name"] == "duplicate-name"
