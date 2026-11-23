"""Pin ``ProxyLogging.post_call_success_hook``."""

from __future__ import annotations

import json
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
                    "logprobs": {
                        "content": [{"token": content, "logprob": -0.1, "bytes": None, "top_logprobs": []}]
                    },
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
        choices=[
            {
                "index": 0,
                "text": text,
                "finish_reason": "length",
                "logprobs": {"tokens": [text], "token_logprobs": [-0.1], "top_logprobs": None, "text_offset": [0]},
                "token_ids": [12095, 13],
            }
        ],
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
async def test_post_call_success_hook_drops_token_data_only_from_choices_a_guardrail_rewrote(
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
    assert (getattr(served.choices[0], "logprobs", None) is None) is (kept is None)



def _masked(content: str) -> str:
    return content.replace("Paris", "[MASKED]")


class _DictMaskingLogger(CustomLogger):
    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        dumped = response.model_dump()
        choice = dumped["choices"][0]
        message = {**choice["message"], "content": _masked(choice["message"]["content"])}
        return {**dumped, "choices": [{**choice, "message": message}]}


class _InPlaceDictMaskingLogger(CustomLogger):
    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        message = response["choices"][0]["message"]
        message["content"] = _masked(message["content"])
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "kept"),
    [
        ("The capital of France is Paris.", None),
        ("The capital of France is Lyon.", [785, 6722, 315, 9625, 374, 12095, 13]),
    ],
    ids=["masked", "untouched"],
)
@pytest.mark.parametrize(
    ("callback", "as_dict"),
    [(_DictMaskingLogger(), False), (_InPlaceDictMaskingLogger(), True)],
    ids=["model-in-dict-out", "dict-rewritten-in-place"],
)
async def test_post_call_success_hook_drops_token_data_from_a_dict_choice_a_callback_rewrote(
    proxy_logging, make_user_api_key_auth, monkeypatch, content, kept, callback, as_dict
):
    monkeypatch.setattr(litellm, "callbacks", [callback])
    response = _vllm_chat_response(content)

    served = await proxy_logging.post_call_success_hook(
        data={"model": "m", "messages": [{"role": "user", "content": "capital of France?"}]},
        response=response.model_dump() if as_dict else response,
        user_api_key_dict=make_user_api_key_auth(),
    )

    choice = served["choices"][0]
    assert "Paris" not in choice["message"]["content"]
    assert choice["provider_specific_fields"]["token_ids"] == kept
    assert (choice["logprobs"] is None) is (kept is None)


class _InPlaceToolArgumentsMaskingLogger(CustomLogger):
    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        function = response["choices"][0]["message"]["tool_calls"][0]["function"]
        function["arguments"] = _masked(function["arguments"])
        return response


@pytest.mark.asyncio
async def test_post_call_success_hook_drops_token_data_when_a_callback_rewrites_dict_tool_arguments_in_place(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    monkeypatch.setattr(litellm, "callbacks", [_InPlaceToolArgumentsMaskingLogger()])
    response = _vllm_chat_response("").model_dump()
    response["choices"][0]["message"]["tool_calls"] = [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"city": "Paris"}'}}
    ]

    served = await proxy_logging.post_call_success_hook(
        data={"model": "m", "messages": [{"role": "user", "content": "capital of France?"}]},
        response=response,
        user_api_key_dict=make_user_api_key_auth(),
    )

    choice = served["choices"][0]
    assert choice["message"]["tool_calls"][0]["function"]["arguments"] == '{"city": "[MASKED]"}'
    assert choice["provider_specific_fields"]["token_ids"] is None
    assert choice["logprobs"] is None


class _InPlaceMessageFieldMaskingLogger(CustomLogger):
    def __init__(self, field: str) -> None:
        super().__init__()
        self.field = field

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        message = response["choices"][0]["message"]
        message[self.field] = json.loads(_masked(json.dumps(message[self.field])))
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reasoning_content", "The user wants Paris."),
        ("thinking_blocks", [{"type": "thinking", "thinking": "The user wants Paris.", "signature": "sig"}]),
        ("refusal", "I will not say Paris."),
        ("audio", {"id": "audio_1", "data": "", "expires_at": 1, "transcript": "Paris"}),
        ("function_call", {"name": "lookup", "arguments": '{"city": "Paris"}'}),
    ],
    ids=["reasoning_content", "thinking_blocks", "refusal", "audio", "function_call"],
)
async def test_post_call_success_hook_drops_token_data_when_a_callback_rewrites_any_answer_field(
    proxy_logging, make_user_api_key_auth, monkeypatch, field, value
):
    monkeypatch.setattr(litellm, "callbacks", [_InPlaceMessageFieldMaskingLogger(field)])
    response = _vllm_chat_response("The capital of France is Lyon.").model_dump()
    response["choices"][0]["message"][field] = value

    served = await proxy_logging.post_call_success_hook(
        data={"model": "m", "messages": [{"role": "user", "content": "capital of France?"}]},
        response=response,
        user_api_key_dict=make_user_api_key_auth(),
    )

    choice = served["choices"][0]
    assert "Paris" not in json.dumps(choice["message"][field])
    assert choice["provider_specific_fields"]["token_ids"] is None
    assert choice["logprobs"] is None


def _responses_output_text(text: str, logprobs_requested: bool):
    from openai.types.responses import ResponseOutputText
    from openai.types.responses.response_output_text import Logprob

    return ResponseOutputText(
        type="output_text",
        text=text,
        annotations=[],
        logprobs=[Logprob(token=text, logprob=-0.1, bytes=[], top_logprobs=[])] if logprobs_requested else [],
    )


class _ResponsesPartMaskingLogger(CustomLogger):
    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        message = response.output[0]
        parts = [part.model_copy(update={"text": _masked(part.text)}) for part in message.content]
        return response.model_copy(update={"output": [message.model_copy(update={"content": parts})]})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("logprobs_requested", "masked_logprobs", "untouched_tokens"),
    [(True, None, ["Berlin is in Germany."]), (False, [], [])],
    ids=["logprobs-requested", "logprobs-not-requested"],
)
async def test_post_call_success_hook_drops_logprobs_only_from_responses_output_text_a_callback_rewrote(
    proxy_logging, make_user_api_key_auth, monkeypatch, logprobs_requested, masked_logprobs, untouched_tokens
):
    from openai.types.responses import ResponseOutputMessage

    from litellm.types.llms.openai import ResponsesAPIResponse

    monkeypatch.setattr(litellm, "callbacks", [_ResponsesPartMaskingLogger()])
    response = ResponsesAPIResponse(
        id="resp_1",
        created_at=1,
        model="gpt-4.1-mini",
        object="response",
        status="completed",
        output=[
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    _responses_output_text("The capital of France is Paris.", logprobs_requested),
                    _responses_output_text("Berlin is in Germany.", logprobs_requested),
                ],
            )
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )

    served = await proxy_logging.post_call_success_hook(
        data={"model": "m", "input": "capital of France?"}, response=response, user_api_key_dict=make_user_api_key_auth()
    )

    masked_part, untouched_part = served.output[0].content
    assert masked_part.text == "The capital of France is [MASKED]."
    assert masked_part.logprobs == masked_logprobs
    assert [logprob.token for logprob in untouched_part.logprobs] == untouched_tokens
