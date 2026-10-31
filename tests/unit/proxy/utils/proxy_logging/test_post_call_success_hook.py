"""Pin ``ProxyLogging.post_call_success_hook``."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks


@pytest.fixture(autouse=True)
def _clear_caps_cache():
    ProxyLogging._callback_capabilities_cache.clear()
    yield
    ProxyLogging._callback_capabilities_cache.clear()


def _make_guardrail(name="g", should_run=True, override=None):
    cb = MagicMock(spec=CustomGuardrail)
    cb.__class__ = CustomGuardrail
    cb.guardrail_name = name
    cb.event_hook = GuardrailEventHooks.post_call
    cb.should_run_guardrail = MagicMock(return_value=should_run)
    cb.async_post_call_success_hook = AsyncMock(return_value=override)
    cb.run_in_parallel = False
    return cb


@pytest.mark.asyncio
async def test_post_call_success_hook_returns_response_when_no_callbacks(proxy_logging, make_user_api_key_auth, mock_callbacks_disabled):
    response = {"original": True, "model": "m", "choices": []}
    out = await proxy_logging.post_call_success_hook(
        data={}, response=response, user_api_key_dict=make_user_api_key_auth()
    )
    assert out == {"original": True, "model": "m", "choices": []}


@pytest.mark.asyncio
async def test_post_call_success_hook_runs_other_callback_and_replaces_response(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    new_response = {"modified": True, "kept": "yes", "final": "v"}

    class _CL(CustomLogger):
        async def async_post_call_success_hook(self, **kwargs):  # type: ignore[override]
            return new_response

    monkeypatch.setattr(litellm, "callbacks", [_CL()])
    out = await proxy_logging.post_call_success_hook(
        data={}, response={"original": True}, user_api_key_dict=make_user_api_key_auth()
    )
    assert out == new_response


@pytest.mark.asyncio
async def test_post_call_success_hook_guardrail_should_not_run_skipped(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    g = _make_guardrail(should_run=False)
    monkeypatch.setattr(litellm, "callbacks", [g])
    response = MagicMock()
    out = await proxy_logging.post_call_success_hook(
        data={}, response=response, user_api_key_dict=make_user_api_key_auth()
    )
    g.async_post_call_success_hook.assert_not_called()
    assert out is response


@pytest.mark.asyncio
async def test_post_call_success_hook_guardrail_error_raises(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    g = _make_guardrail()
    g.async_post_call_success_hook = AsyncMock(side_effect=RuntimeError("blocked"))
    monkeypatch.setattr(litellm, "callbacks", [g])
    with pytest.raises(RuntimeError):
        await proxy_logging.post_call_success_hook(
            data={}, response=MagicMock(), user_api_key_dict=make_user_api_key_auth()
        )


@pytest.mark.asyncio
async def test_post_call_success_hook_guardrail_returns_modified_response(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    modified = {"a": 1, "b": 2, "c": 3}
    g = _make_guardrail(override=modified)
    monkeypatch.setattr(litellm, "callbacks", [g])
    out = await proxy_logging.post_call_success_hook(
        data={}, response={"orig": True}, user_api_key_dict=make_user_api_key_auth()
    )
    assert out == modified


@pytest.mark.asyncio
@pytest.mark.parametrize("run_in_parallel", [False, True], ids=["sequential", "parallel"])
async def test_post_call_block_names_the_blocking_guardrail_in_applied_guardrails(
    proxy_logging, make_user_api_key_auth, monkeypatch, run_in_parallel
):
    def _passer_that_records(data, user_api_key_dict, response):
        data["metadata"]["applied_guardrails"] = ["passer"]

    passer = _make_guardrail("passer")
    passer.async_post_call_success_hook = AsyncMock(side_effect=_passer_that_records)
    passer.run_in_parallel = run_in_parallel
    blocker = _make_guardrail("blocker")
    blocker.async_post_call_success_hook = AsyncMock(side_effect=HTTPException(status_code=400, detail="blocked"))
    blocker.run_in_parallel = run_in_parallel
    monkeypatch.setattr(litellm, "callbacks", [passer, blocker])
    data = {"model": "m", "metadata": {}}
    with pytest.raises(HTTPException):
        await proxy_logging.post_call_success_hook(
            data=data, response=MagicMock(), user_api_key_dict=make_user_api_key_auth()
        )
    assert data["metadata"]["applied_guardrails"] == ["passer", "blocker"]


def _vllm_chat_response(content: str) -> litellm.ModelResponse:
    from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
        convert_to_model_response_object,
    )

    return convert_to_model_response_object(
        response_object={
            "id": "chatcmpl-vllm",
            "object": "chat.completion",
            "created": 1,
            "model": "Qwen/Qwen3-0.6B",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "token_ids": [785, 6722, 315, 9625, 374, 12095, 13],
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        },
        model_response_object=litellm.ModelResponse(),
    )


def _vllm_text_completion_response(text: str) -> litellm.TextCompletionResponse:
    return litellm.TextCompletionResponse(
        id="cmpl-vllm",
        object="text_completion",
        created=1,
        model="Qwen/Qwen3-0.6B",
        choices=[{"index": 0, "text": text, "finish_reason": "length", "logprobs": None, "token_ids": [12095, 13]}],
        usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    )


def _chat_token_ids(response: litellm.ModelResponse) -> object:
    return response.choices[0].provider_specific_fields["token_ids"]


def _text_token_ids(response: litellm.TextCompletionResponse) -> object:
    return response.choices[0].token_ids


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "data", "token_ids_of", "kept"),
    [
        (
            _vllm_chat_response("The capital of France is Paris."),
            {"model": "m", "messages": [{"role": "user", "content": "capital of France?"}]},
            _chat_token_ids,
            None,
        ),
        (
            _vllm_chat_response("The capital of France is Lyon."),
            {"model": "m", "messages": [{"role": "user", "content": "capital of France?"}]},
            _chat_token_ids,
            [785, 6722, 315, 9625, 374, 12095, 13],
        ),
        (
            _vllm_text_completion_response(" Paris."),
            {"model": "m", "prompt": "The capital of France is"},
            _text_token_ids,
            None,
        ),
        (
            _vllm_text_completion_response(" Lyon."),
            {"model": "m", "prompt": "The capital of France is"},
            _text_token_ids,
            [12095, 13],
        ),
    ],
    ids=["chat-masked", "chat-untouched", "completion-masked", "completion-untouched"],
)
async def test_post_call_success_hook_drops_token_ids_only_from_choices_a_guardrail_rewrote(
    proxy_logging, make_user_api_key_auth, monkeypatch, response, data, token_ids_of, kept
):
    from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
        ContentFilterGuardrail,
    )
    from litellm.types.guardrails import BlockedWord, ContentFilterAction

    mask_paris = ContentFilterGuardrail(
        guardrail_name="mask-paris",
        blocked_words=[BlockedWord(keyword="Paris", action=ContentFilterAction.MASK)],
        event_hook=GuardrailEventHooks.post_call,
        default_on=True,
    )
    monkeypatch.setattr(litellm, "callbacks", [mask_paris])

    served = await proxy_logging.post_call_success_hook(
        data=data, response=response, user_api_key_dict=make_user_api_key_auth()
    )

    assert served is response
    assert token_ids_of(served) == kept
