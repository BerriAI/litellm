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
    if b'"stream":true' in request.body.replace(b" ", b""):
        chunk: Final = {
            "id": "chatcmpl-sdk-azure-guardrail",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": {"content": "permitted response"}, "finish_reason": None}],
        }
        return Reply(
            content_type="text/event-stream",
            chunks=(b"data: " + json.dumps(chunk).encode() + b"\n\n", b"data: [DONE]\n\n"),
        )
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
        monkeypatch.setattr(litellm, "success_callback", list(litellm.success_callback))
        monkeypatch.setattr(litellm, "_async_success_callback", list(litellm._async_success_callback))
        monkeypatch.setattr(litellm, "failure_callback", list(litellm.failure_callback))
        monkeypatch.setattr(litellm, "_async_failure_callback", list(litellm._async_failure_callback))
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


def _provider_prompts(provider: Wire) -> tuple[str, ...]:
    return tuple(_provider_prompt(request) for request in provider.drain())


def _provider_prompt(request: Request) -> str:
    body: Final = object_value(json.loads(request.body))
    messages: Final = body["messages"]
    assert isinstance(messages, list), body
    prompt: Final = object_value(messages[-1])["content"]
    assert isinstance(prompt, str), body
    return prompt


def test_k1_acompletion_scans_tuple_messages(sdk_rig: tuple[AzureContentSafetyPromptShieldGuardrail, Wire, Wire]) -> None:
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


def test_k1_acompletion_scans_system_and_user_tuple_messages(
    sdk_rig: tuple[AzureContentSafetyPromptShieldGuardrail, Wire, Wire],
) -> None:
    _, azure, provider = sdk_rig
    messages: Final = (
        {"role": "system", "content": "system context"},
        {"role": "user", "content": _ATTACK_PROMPT},
    )
    with pytest.raises((HTTPException, litellm.BadRequestError), match="Violated Azure Prompt Shield"):
        asyncio.run(
            litellm.acompletion(
                model="openai/gpt-4o-mini",
                api_base=provider.url + "/v1",
                api_key=_PROVIDER_KEY,
                messages=messages,
                guardrails=[_GUARDRAIL_NAME],
                max_tokens=8,
            )
        )
    assert _azure_prompts(azure) == (_ATTACK_PROMPT,)
    assert provider.drain() == ()


def test_k1_acompletion_stream_scans_tuple_messages(
    sdk_rig: tuple[AzureContentSafetyPromptShieldGuardrail, Wire, Wire],
) -> None:
    _, azure, provider = sdk_rig

    async def consume() -> None:
        stream: Final = await litellm.acompletion(
            model="openai/gpt-4o-mini",
            api_base=provider.url + "/v1",
            api_key=_PROVIDER_KEY,
            messages=({"role": "user", "content": _ATTACK_PROMPT},),
            guardrails=[_GUARDRAIL_NAME],
            max_tokens=8,
            stream=True,
        )
        async for _chunk in stream:
            pass

    with pytest.raises((HTTPException, litellm.BadRequestError), match="Violated Azure Prompt Shield"):
        asyncio.run(consume())
    assert _azure_prompts(azure) == (_ATTACK_PROMPT,)
    assert provider.drain() == ()


def test_k1_acompletion_list_messages_control(
    sdk_rig: tuple[AzureContentSafetyPromptShieldGuardrail, Wire, Wire]
) -> None:
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


def test_k3_router_acompletion_guardrails_kwarg_scans_tuple_messages(
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


def test_k3_router_acompletion_guardrails_kwarg_list_control(
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


def test_k2_litellm_completion_tuple_behavior_pinned_from_base(
    sdk_rig: tuple[AzureContentSafetyPromptShieldGuardrail, Wire, Wire],
) -> None:
    _, azure, provider = sdk_rig
    expected_block: Final = False
    messages: Final = ({"role": "user", "content": _ATTACK_PROMPT},)
    if expected_block:
        with pytest.raises((HTTPException, litellm.BadRequestError), match="Violated Azure Prompt Shield"):
            litellm.completion(
                model="openai/gpt-4o-mini",
                api_base=provider.url + "/v1",
                api_key=_PROVIDER_KEY,
                messages=messages,
                guardrails=[_GUARDRAIL_NAME],
                max_tokens=8,
            )
    else:
        response: Final = litellm.completion(
            model="openai/gpt-4o-mini",
            api_base=provider.url + "/v1",
            api_key=_PROVIDER_KEY,
            messages=messages,
            guardrails=[_GUARDRAIL_NAME],
            max_tokens=8,
        )
        assert response.choices
    assert _azure_prompts(azure) == ((_ATTACK_PROMPT,) if expected_block else ())
    if expected_block:
        assert provider.drain() == ()
    else:
        assert _provider_prompts(provider) == (_ATTACK_PROMPT,)


def _router_with_guardrails(provider: Wire) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "sdk-guardrail-model",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": provider.url + "/v1",
                    "api_key": _PROVIDER_KEY,
                    "guardrails": [_GUARDRAIL_NAME],
                },
            }
        ]
    )


def test_k4_router_deployment_guardrails_scan_tuple_messages(
    sdk_rig: tuple[AzureContentSafetyPromptShieldGuardrail, Wire, Wire],
) -> None:
    _, azure, provider = sdk_rig
    router: Final = _router_with_guardrails(provider)
    try:
        with pytest.raises((HTTPException, litellm.BadRequestError), match="Violated Azure Prompt Shield"):
            asyncio.run(
                router.acompletion(
                    model="sdk-guardrail-model",
                    messages=({"role": "user", "content": _ATTACK_PROMPT},),
                    max_tokens=8,
                )
            )
    finally:
        router.reset()
    assert _azure_prompts(azure) == (_ATTACK_PROMPT,)
    assert provider.drain() == ()


def test_k4_router_deployment_guardrails_scan_list_messages(
    sdk_rig: tuple[AzureContentSafetyPromptShieldGuardrail, Wire, Wire],
) -> None:
    _, azure, provider = sdk_rig
    router: Final = _router_with_guardrails(provider)
    try:
        with pytest.raises((HTTPException, litellm.BadRequestError), match="Violated Azure Prompt Shield"):
            asyncio.run(
                router.acompletion(
                    model="sdk-guardrail-model",
                    messages=[{"role": "user", "content": _ATTACK_PROMPT}],
                    max_tokens=8,
                )
            )
    finally:
        router.reset()
    assert _azure_prompts(azure) == (_ATTACK_PROMPT,)
    assert provider.drain() == ()
