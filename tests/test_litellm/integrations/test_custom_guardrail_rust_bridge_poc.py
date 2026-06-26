from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks


@dataclass
class RustStyleGuardrailContext:
    call_type: str
    selected_guardrails: List[str]
    user_api_key_dict: Any = None
    cache: Any = None


@dataclass
class RustStyleGuardrailRequest:
    data: Dict[str, Any]


class PyO3StyleCustomGuardrailAdapter:
    """POC shape for a PyO3 adapter over Python CustomGuardrail.

    This mirrors the PR #31368 Rust trait names:
    CustomGuardrail::async_pre_call_hook(context, request) delegates to
    Python CustomGuardrail.async_pre_call_hook(user_api_key_dict, cache, data, call_type).
    """

    def __init__(self, guardrail: CustomGuardrail):
        self.guardrail = guardrail

    async def async_pre_call_hook(
        self,
        context: RustStyleGuardrailContext,
        request: RustStyleGuardrailRequest,
    ) -> RustStyleGuardrailRequest:
        if (
            self.guardrail.should_run_guardrail(
                data=request.data,
                event_type=GuardrailEventHooks.pre_call,
            )
            is not True
        ):
            return request

        result = await self.guardrail.async_pre_call_hook(
            user_api_key_dict=context.user_api_key_dict,
            cache=context.cache,
            data=request.data,
            call_type=context.call_type,
        )
        if isinstance(result, dict):
            return RustStyleGuardrailRequest(data=result)
        return request


class RustStyleGuardrailLifecycleStub:
    def __init__(self, adapters: List[PyO3StyleCustomGuardrailAdapter]):
        self.adapters = adapters

    async def run_pre_call(
        self,
        context: RustStyleGuardrailContext,
        request: RustStyleGuardrailRequest,
    ) -> RustStyleGuardrailRequest:
        for adapter in self.adapters:
            request = await adapter.async_pre_call_hook(context, request)
        return request


class RecordingPythonCustomGuardrail(CustomGuardrail):
    def __init__(self) -> None:
        super().__init__(
            guardrail_name="python-rail",
            supported_event_hooks=[GuardrailEventHooks.pre_call],
            event_hook=GuardrailEventHooks.pre_call,
        )
        self.calls: List[Dict[str, Any]] = []

    async def async_pre_call_hook(
        self,
        user_api_key_dict,
        cache,
        data: dict,
        call_type: str,
        **kwargs,
    ):
        self.calls.append(
            {
                "call_type": call_type,
                "messages": data["messages"],
                "selected_guardrails": data["guardrails"],
                "user_api_key_dict": user_api_key_dict,
                "cache": cache,
            }
        )
        return {
            **data,
            "messages": [{"role": "user", "content": "masked by python guardrail"}],
        }


def collect_python_custom_guardrail_adapters_from_callback_manager():
    guardrails = litellm.logging_callback_manager.get_custom_loggers_for_type(
        CustomGuardrail
    )
    return [PyO3StyleCustomGuardrailAdapter(guardrail) for guardrail in guardrails]


@pytest.mark.asyncio
async def test_should_collect_registered_python_custom_guardrail_and_invoke_from_rust_lifecycle_stub():
    manager = litellm.logging_callback_manager
    manager._reset_all_callbacks()
    guardrail = RecordingPythonCustomGuardrail()

    try:
        manager.add_litellm_callback(guardrail)
        manager.add_litellm_success_callback(guardrail)

        adapters = collect_python_custom_guardrail_adapters_from_callback_manager()
        lifecycle = RustStyleGuardrailLifecycleStub(adapters)
        result = await lifecycle.run_pre_call(
            RustStyleGuardrailContext(
                call_type="ocr",
                selected_guardrails=["python-rail"],
                user_api_key_dict={"user_id": "user-1"},
                cache="cache-sentinel",
            ),
            RustStyleGuardrailRequest(
                data={
                    "model": "mistral-ocr-latest",
                    "custom_llm_provider": "mistral",
                    "guardrails": ["python-rail"],
                    "messages": [{"role": "user", "content": "raw input"}],
                }
            ),
        )

        assert len(adapters) == 1
        assert guardrail.calls == [
            {
                "call_type": "ocr",
                "messages": [{"role": "user", "content": "raw input"}],
                "selected_guardrails": ["python-rail"],
                "user_api_key_dict": {"user_id": "user-1"},
                "cache": "cache-sentinel",
            }
        ]
        assert result.data["messages"] == [
            {"role": "user", "content": "masked by python guardrail"}
        ]
    finally:
        manager._reset_all_callbacks()
