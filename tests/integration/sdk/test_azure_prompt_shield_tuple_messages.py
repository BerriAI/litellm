import asyncio
import json
from collections.abc import Iterator
from contextlib import ExitStack
from typing import Final

import litellm
import pytest
from fastapi import HTTPException
from integration._support.client import object_value
from integration._support.wire import Reply, Request, Wire, wire_server
from litellm import Router
from litellm.proxy.guardrails.guardrail_hooks.azure.prompt_shield import (
    AzureContentSafetyPromptShieldGuardrail,
)

_ATTACK_MARKER: Final = "synthetic-sdk-attack-marker"
_ATTACK_PROMPT: Final = f"synthetic sdk prompt {_ATTACK_MARKER}"
_SHIELD_TARGET_PREFIX: Final = "/contentsafety/text:shieldPrompt?api-version="
_PROVIDER_KEY: Final = "synthetic-provider-key"
_AZURE_KEY: Final = "synthetic-azure-key"
_GUARDRAIL_NAME: Final = "sdk-azure-shield"


def _azure(request: Request) -> Reply:
    assert request.method == "POST", request.method
    assert request.target.startswith(_SHIELD_TARGET_PREFIX), request.target
    body: Final = object_value(json.loads(request.body))
    prompt: Final = body["userPrompt"]
    assert isinstance(prompt, str), body
    return Reply(
        body=json.dumps(
            {
                "userPromptAnalysis": {"attackDetected": _ATTACK_MARKER in prompt},
                "documentsAnalysis": [],
            }
        ).encode()
    )


def _provider(request: Request) -> Reply:
    assert request.method == "POST", request.method
    assert request.target == "/v1/chat/completions", request.target
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-sdk-azure-guardrail",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "permitted response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
    )


@pytest.fixture
def sdk_rig(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[AzureContentSafetyPromptShieldGuardrail, Wire, Wire]]:
    with ExitStack() as stack:
        azure: Final = stack.enter_context(wire_server(_azure))
        provider: Final = stack.enter_context(wire_server(_provider))
        guardrail: Final = AzureContentSafetyPromptShieldGuardrail(
            guardrail_name=_GUARDRAIL_NAME,
            api_key=_AZURE_KEY,
            api_base=azure.url,
            event_hook="pre_call",
            default_on=False,
        )
        monkeypatch.setattr(litellm, "callbacks", [guardrail])
        try:
            yield guardrail, azure, provider
        finally:
            litellm.logging_callback_manager.remove_callback_from_all_lists(guardrail)


def _azure_prompts(azure: Wire) -> tuple[str, ...]:
    return tuple(_azure_prompt(request) for request in azure.drain())


def _azure_prompt(request: Request) -> str:
    body: Final = object_value(json.loads(request.body))
    prompt: Final = body["userPrompt"]
    assert isinstance(prompt, str), body
    return prompt


def test_acompletion_scans_tuple_messages(sdk_rig: tuple[AzureContentSafetyPromptShieldGuardrail, Wire, Wire]) -> None:
    _, azure, provider = sdk_rig
    with pytest.raises((HTTPException, litellm.BadRequestError), match="Violated Azure Prompt Shield"):
        asyncio.run(
            litellm.acompletion(
                model="openai/gpt-4o-mini",
                api_base=provider.url + "/v1",
                api_key=_PROVIDER_KEY,
                messages=({"role": "user", "content": _ATTACK_PROMPT},),
                guardrails=[_GUARDRAIL_NAME],
                max_tokens=8,
            )
        )
    assert _azure_prompts(azure) == (_ATTACK_PROMPT,)
    assert provider.drain() == ()


def test_acompletion_list_messages_control(sdk_rig: tuple[AzureContentSafetyPromptShieldGuardrail, Wire, Wire]) -> None:
    _, azure, provider = sdk_rig
    with pytest.raises((HTTPException, litellm.BadRequestError), match="Violated Azure Prompt Shield"):
        asyncio.run(
            litellm.acompletion(
                model="openai/gpt-4o-mini",
                api_base=provider.url + "/v1",
                api_key=_PROVIDER_KEY,
                messages=[{"role": "user", "content": _ATTACK_PROMPT}],
                guardrails=[_GUARDRAIL_NAME],
                max_tokens=8,
            )
        )
    assert _azure_prompts(azure) == (_ATTACK_PROMPT,)
    assert provider.drain() == ()


def _router(provider: Wire) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "sdk-guardrail-model",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": provider.url + "/v1",
                    "api_key": _PROVIDER_KEY,
                },
            }
        ]
    )


def test_router_acompletion_scans_tuple_messages(
    sdk_rig: tuple[AzureContentSafetyPromptShieldGuardrail, Wire, Wire],
) -> None:
    _, azure, provider = sdk_rig
    router: Final = _router(provider)
    try:
        with pytest.raises((HTTPException, litellm.BadRequestError), match="Violated Azure Prompt Shield"):
            asyncio.run(
                router.acompletion(
                    model="sdk-guardrail-model",
                    messages=({"role": "user", "content": _ATTACK_PROMPT},),
                    guardrails=[_GUARDRAIL_NAME],
                    max_tokens=8,
                )
            )
    finally:
        router.reset()
    assert _azure_prompts(azure) == (_ATTACK_PROMPT,)
    assert provider.drain() == ()


def test_router_acompletion_list_messages_control(
    sdk_rig: tuple[AzureContentSafetyPromptShieldGuardrail, Wire, Wire],
) -> None:
    _, azure, provider = sdk_rig
    router: Final = _router(provider)
    try:
        with pytest.raises((HTTPException, litellm.BadRequestError), match="Violated Azure Prompt Shield"):
            asyncio.run(
                router.acompletion(
                    model="sdk-guardrail-model",
                    messages=[{"role": "user", "content": _ATTACK_PROMPT}],
                    guardrails=[_GUARDRAIL_NAME],
                    max_tokens=8,
                )
            )
    finally:
        router.reset()
    assert _azure_prompts(azure) == (_ATTACK_PROMPT,)
    assert provider.drain() == ()
