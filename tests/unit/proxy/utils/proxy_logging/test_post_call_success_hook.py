"""Pin ``ProxyLogging.post_call_success_hook``."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from openai.types.responses import ResponseOutputMessage, ResponseOutputText
from openai.types.responses.response_output_text import Logprob

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_logger_v2 import CallContext, CustomLoggerV2, Field, Hook, Patch, on
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.llms.openai import ResponsesAPIResponse


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


def _vllm_chat_response(
    content: str,
    tool_calls: tuple[dict[str, object], ...] = (),
    token_ids: tuple[int, ...] = (785, 6722, 315, 9625, 374, 12095, 13),
    logprob: float = -0.1,
) -> litellm.ModelResponse:
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
                    "token_ids": list(token_ids),
                    "message": {"role": "assistant", "content": content, "tool_calls": list(tool_calls) or None},
                    "logprobs": {
                        "content": [{"token": content, "logprob": logprob, "bytes": None, "top_logprobs": []}]
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



_LOOKUP_TOOL_CALL: Final = {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}


def _masked(content: str) -> str:
    return content.replace("Paris", "[MASKED]")


class _DictMaskingLogger(CustomLogger):
    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> dict[str, object]:
        assert isinstance(response, litellm.ModelResponse)
        dumped = response.model_dump()
        choice = dumped["choices"][0]
        message = {**choice["message"], "content": _masked(choice["message"]["content"])}
        return {**dumped, "choices": [{**choice, "message": message}]}


class _InPlaceDictMaskingLogger(CustomLogger):
    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> dict[str, Any]:
        assert isinstance(response, dict)
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
    response = _vllm_chat_response(content, tool_calls=(_LOOKUP_TOOL_CALL,))

    served = await proxy_logging.post_call_success_hook(
        data={"model": "m", "messages": [{"role": "user", "content": "capital of France?"}]},
        response=response.model_dump() if as_dict else response,
        user_api_key_dict=make_user_api_key_auth(),
    )

    choice = served["choices"][0]
    assert "Paris" not in choice["message"]["content"]
    assert choice["message"]["tool_calls"][0]["function"] == _LOOKUP_TOOL_CALL["function"]
    assert choice["provider_specific_fields"]["token_ids"] == kept
    assert (choice["logprobs"] is None) is (kept is None)


class _TokenIdsMovingMaskingLogger(CustomLogger):
    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> dict[str, object]:
        assert isinstance(response, litellm.ModelResponse)
        dumped = response.model_dump()
        choice = dumped["choices"][0]
        message = {
            **choice["message"],
            "content": _masked(choice["message"]["content"]),
            "provider_specific_fields": {"token_ids": choice["provider_specific_fields"]["token_ids"]},
        }
        moved = {**choice, "token_ids": None, "provider_specific_fields": None, "message": message}
        return {**dumped, "choices": [moved]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "kept"),
    [
        ("The capital of France is Paris.", None),
        ("The capital of France is Lyon.", [785, 6722, 315, 9625, 374, 12095, 13]),
    ],
    ids=["masked", "untouched"],
)
async def test_post_call_success_hook_drops_token_ids_a_callback_moved_to_another_field_while_masking(
    proxy_logging, make_user_api_key_auth, monkeypatch, content, kept
):
    monkeypatch.setattr(litellm, "callbacks", [_TokenIdsMovingMaskingLogger()])

    served = await proxy_logging.post_call_success_hook(
        data={"model": "m", "messages": [{"role": "user", "content": "capital of France?"}]},
        response=_vllm_chat_response(content),
        user_api_key_dict=make_user_api_key_auth(),
    )

    choice = served["choices"][0]
    assert "Paris" not in choice["message"]["content"]
    assert choice["message"]["provider_specific_fields"]["token_ids"] == kept
    assert (choice["logprobs"] is None) is (kept is None)


class _NewIdsAddingMaskingLogger(CustomLogger):
    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> dict[str, object]:
        assert isinstance(response, litellm.ModelResponse)
        dumped = response.model_dump()
        choice = dumped["choices"][0]
        message = {**choice["message"], "content": _masked(choice["message"]["content"])}
        return {**dumped, "choices": [{**choice, "token_ids": [1, 2, 3], "message": message}]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "provider_token_ids", "kept_token_ids", "kept_provider_token_ids"),
    [
        ("The capital of France is Paris.", [785, 6722, 315, 9625, 374, 12095, 13], None, None),
        ("The capital of France is Paris.", [9, 9], None, None),
        ("The capital of France is Lyon.", [9, 9], [1, 2, 3], [9, 9]),
    ],
    ids=["masked", "masked-provider-ids-differ", "untouched"],
)
async def test_post_call_success_hook_drops_every_token_id_field_when_a_callback_masks_and_adds_new_ids(
    proxy_logging,
    make_user_api_key_auth,
    monkeypatch,
    content,
    provider_token_ids,
    kept_token_ids,
    kept_provider_token_ids,
):
    monkeypatch.setattr(litellm, "callbacks", [_NewIdsAddingMaskingLogger()])
    response = _vllm_chat_response(content)
    response.choices[0].provider_specific_fields = {"token_ids": provider_token_ids}

    served = await proxy_logging.post_call_success_hook(
        data={"model": "m", "messages": [{"role": "user", "content": "capital of France?"}]},
        response=response,
        user_api_key_dict=make_user_api_key_auth(),
    )

    choice = served["choices"][0]
    assert choice["token_ids"] == kept_token_ids
    assert choice["provider_specific_fields"]["token_ids"] == kept_provider_token_ids


class _InPlaceToolArgumentsMaskingLogger(CustomLogger):
    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> dict[str, Any]:
        assert isinstance(response, dict)
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

    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> dict[str, Any]:
        assert isinstance(response, dict)
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


class _InPlaceContentEditingLogger(CustomLogger):
    def __init__(self, edit: Callable[[str], str]) -> None:
        super().__init__()
        self.edit = edit

    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> dict[str, Any]:
        assert isinstance(response, dict)
        message = response["choices"][0]["message"]
        message["content"] = self.edit(message["content"])
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("edit", "keeps_token_data"),
    [
        (lambda content: f"{content}\n\nSource: atlas", True),
        (lambda content: f"Answer: {content}", False),
        (lambda content: content[:-1], False),
    ],
    ids=["appended-footer", "prepended-label", "truncated"],
)
async def test_post_call_success_hook_keeps_token_data_only_when_a_callback_appends_to_the_answer(
    proxy_logging, make_user_api_key_auth, monkeypatch, edit, keeps_token_data
):
    monkeypatch.setattr(litellm, "callbacks", [_InPlaceContentEditingLogger(edit)])
    content = "The capital of France is Lyon."

    served = await proxy_logging.post_call_success_hook(
        data={"model": "m", "messages": [{"role": "user", "content": "capital of France?"}]},
        response=_vllm_chat_response(content).model_dump(),
        user_api_key_dict=make_user_api_key_auth(),
    )

    choice = served["choices"][0]
    assert choice["message"]["content"] == edit(content)
    assert (choice["provider_specific_fields"]["token_ids"] == [785, 6722, 315, 9625, 374, 12095, 13]) is keeps_token_data
    assert (choice["logprobs"] is not None) is keeps_token_data


_ROME_TOKEN_IDS: Final = (785, 6722, 315, 15344, 374, 21718, 13)


class _ReaskingLogger(CustomLogger):
    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> litellm.ModelResponse:
        return _vllm_chat_response("The capital of Italy is Rome.", token_ids=_ROME_TOKEN_IDS, logprob=-0.3)


@pytest.mark.asyncio
async def test_post_call_success_hook_keeps_token_data_of_a_new_answer_a_callback_returned(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    monkeypatch.setattr(litellm, "callbacks", [_ReaskingLogger()])

    served = await proxy_logging.post_call_success_hook(
        data={"model": "m", "messages": [{"role": "user", "content": "capital of France?"}]},
        response=_vllm_chat_response("The capital of France is Paris."),
        user_api_key_dict=make_user_api_key_auth(),
    )

    choice = served.model_dump()["choices"][0]
    assert choice["message"]["content"] == "The capital of Italy is Rome."
    assert choice["provider_specific_fields"]["token_ids"] == list(_ROME_TOKEN_IDS)
    assert [entry["token"] for entry in choice["logprobs"]["content"]] == ["The capital of Italy is Rome."]


class _InPlaceRomeMaskingLogger(CustomLogger):
    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> None:
        assert isinstance(response, litellm.ModelResponse)
        choice = response.choices[0]
        assert isinstance(choice, litellm.Choices)
        choice.message.content = choice.message.content.replace("Rome", "[MASKED]")


class _ReaskingGuardrail(CustomGuardrail):
    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> litellm.ModelResponse:
        return _vllm_chat_response("The capital of Italy is Rome.", token_ids=_ROME_TOKEN_IDS, logprob=-0.3)


def _rome_mask(run_in_parallel: bool) -> CustomGuardrail:
    from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
        ContentFilterGuardrail,
    )
    from litellm.types.guardrails import BlockedWord, ContentFilterAction

    return ContentFilterGuardrail(
        guardrail_name="mask-rome",
        blocked_words=[BlockedWord(keyword="Rome", action=ContentFilterAction.MASK)],
        event_hook=GuardrailEventHooks.post_call,
        default_on=True,
        run_in_parallel=run_in_parallel,
    )


def _reasking_guardrail() -> CustomGuardrail:
    return _ReaskingGuardrail(guardrail_name="reask", event_hook=GuardrailEventHooks.post_call, default_on=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("callbacks", "served_answer"),
    [
        (lambda: [_ReaskingLogger(), _InPlaceRomeMaskingLogger()], "The capital of Italy is [MASKED]."),
        (lambda: [_reasking_guardrail(), _rome_mask(False)], "The capital of Italy is [KEYWORD_REDACTED]."),
        (lambda: [_reasking_guardrail(), _rome_mask(True)], "The capital of Italy is [KEYWORD_REDACTED]."),
    ],
    ids=["callback", "guardrail", "parallel-guardrail"],
)
async def test_post_call_success_hook_drops_token_data_when_a_later_hook_masks_a_new_answer(
    proxy_logging, make_user_api_key_auth, monkeypatch, callbacks, served_answer
):
    monkeypatch.setattr(litellm, "callbacks", callbacks())

    served = await proxy_logging.post_call_success_hook(
        data={"model": "m", "messages": [{"role": "user", "content": "capital of France?"}]},
        response=_vllm_chat_response("The capital of France is Paris."),
        user_api_key_dict=make_user_api_key_auth(),
    )

    choice = served.model_dump()["choices"][0]
    assert choice["message"]["content"] == served_answer
    assert choice["provider_specific_fields"]["token_ids"] is None
    assert choice["logprobs"] is None


class _AnswerCopyingLogger(CustomLogger):
    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> None:
        assert isinstance(response, dict)
        first, second = response["choices"]
        first["message"]["content"] = second["message"]["content"]


@pytest.mark.asyncio
async def test_post_call_success_hook_drops_token_data_of_a_choice_rewritten_to_a_sibling_with_equal_logprobs(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    monkeypatch.setattr(litellm, "callbacks", [_AnswerCopyingLogger()])
    hidden = _vllm_chat_response("The capital of France is Paris.").model_dump()
    sibling = _vllm_chat_response("Lyon is in France.", token_ids=(43, 44, 9625, 13)).model_dump()

    served = await proxy_logging.post_call_success_hook(
        data={"model": "m", "messages": [{"role": "user", "content": "a city in France?"}], "n": 2},
        response={**hidden, "choices": [*hidden["choices"], {**sibling["choices"][0], "index": 1}]},
        user_api_key_dict=make_user_api_key_auth(),
    )

    first, second = served["choices"]
    assert first["message"]["content"] == second["message"]["content"] == "Lyon is in France."
    assert first["provider_specific_fields"]["token_ids"] is None
    assert first["logprobs"] is None
    assert second["provider_specific_fields"]["token_ids"] == [43, 44, 9625, 13]
    assert second["logprobs"] is not None


@pytest.mark.asyncio
async def test_post_call_success_hook_drops_logprobs_of_a_choice_rewritten_to_a_sibling_with_equal_logprobs_only(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    monkeypatch.setattr(litellm, "callbacks", [_AnswerCopyingLogger()])
    hidden = _vllm_chat_response("The capital of France is Paris.").model_dump()["choices"][0]
    sibling = _vllm_chat_response("Lyon is in France.").model_dump()["choices"][0]
    logprobs_only = {"token_ids": None, "provider_specific_fields": None}

    served = await proxy_logging.post_call_success_hook(
        data={"model": "m", "messages": [{"role": "user", "content": "a city in France?"}], "n": 2, "logprobs": True},
        response={
            **_vllm_chat_response("").model_dump(),
            "choices": [{**hidden, **logprobs_only}, {**sibling, **logprobs_only, "index": 1}],
        },
        user_api_key_dict=make_user_api_key_auth(),
    )

    first, _ = served["choices"]
    assert first["message"]["content"] == "Lyon is in France."
    assert first["logprobs"] is None


class _FirstChoiceDroppingLogger(CustomLogger):
    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> dict[str, Any]:
        assert isinstance(response, dict)
        return {**response, "choices": response["choices"][1:]}


@pytest.mark.asyncio
async def test_post_call_success_hook_keeps_token_data_of_a_choice_that_moved_up_when_a_callback_dropped_another(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    monkeypatch.setattr(litellm, "callbacks", [_FirstChoiceDroppingLogger()])
    dropped = _vllm_chat_response("The capital of France is Paris.").model_dump()
    survivor = _vllm_chat_response("Lyon is in France.", token_ids=(43, 44, 9625, 13), logprob=-0.7).model_dump()

    served = await proxy_logging.post_call_success_hook(
        data={"model": "m", "messages": [{"role": "user", "content": "a city in France?"}], "n": 2},
        response={**dropped, "choices": [*dropped["choices"], {**survivor["choices"][0], "index": 1}]},
        user_api_key_dict=make_user_api_key_auth(),
    )

    (choice,) = served["choices"]
    assert choice["message"]["content"] == "Lyon is in France."
    assert choice["provider_specific_fields"]["token_ids"] == [43, 44, 9625, 13]
    assert choice["logprobs"] is not None


class _SerializedAnswerMaskingLogger(CustomLogger):
    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> dict[str, Any]:
        return json.loads(_masked(json.dumps(response)))


@pytest.mark.asyncio
async def test_post_call_success_hook_drops_logprobs_when_a_callback_masks_the_serialized_answer(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    monkeypatch.setattr(litellm, "callbacks", [_SerializedAnswerMaskingLogger()])
    response = _vllm_chat_response("The capital of France is Paris.").model_dump()
    logprobs_only = {**response["choices"][0], "token_ids": None, "provider_specific_fields": None}

    served = await proxy_logging.post_call_success_hook(
        data={"model": "m", "messages": [{"role": "user", "content": "capital of France?"}]},
        response={**response, "choices": [logprobs_only]},
        user_api_key_dict=make_user_api_key_auth(),
    )

    choice = served["choices"][0]
    assert choice["message"]["content"] == "The capital of France is [MASKED]."
    assert choice["logprobs"] is None


class _SerializedAnswerMaskingV2Callback(CustomLoggerV2):
    @on(Hook.post_call_success, reads={Field.response}, writes={Field.response})
    async def post_call_success(self, view: object, ctx: CallContext[object]) -> Patch:
        return Patch().set(Field.response, json.loads(_masked(json.dumps(getattr(view, "response")))))  # noqa: B009  # the view's declared field


@pytest.mark.asyncio
async def test_post_call_success_hook_drops_token_data_when_a_v2_callback_masks_the_answer(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    monkeypatch.setattr(litellm, "callbacks", [_SerializedAnswerMaskingV2Callback()])

    served = await proxy_logging.post_call_success_hook(
        data={"model": "m", "messages": [{"role": "user", "content": "capital of France?"}]},
        response=_vllm_chat_response("The capital of France is Paris.").model_dump(),
        user_api_key_dict=make_user_api_key_auth(),
    )

    choice = served["choices"][0]
    assert choice["message"]["content"] == "The capital of France is [MASKED]."
    assert choice["provider_specific_fields"]["token_ids"] is None
    assert choice["logprobs"] is None


class _ParisReaskingV2Callback(CustomLoggerV2):
    @on(Hook.post_call_success, reads={Field.response}, writes={Field.response})
    async def post_call_success(self, view: object, ctx: CallContext[object]) -> Patch:
        return Patch().set(Field.response, _vllm_chat_response("The capital of France is Paris.").model_dump())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("callbacks", "served_answer", "served_token_ids"),
    [
        (
            [_ParisReaskingV2Callback(), _SerializedAnswerMaskingV2Callback()],
            "The capital of France is [MASKED].",
            None,
        ),
        ([_ParisReaskingV2Callback()], "The capital of France is Paris.", [785, 6722, 315, 9625, 374, 12095, 13]),
    ],
    ids=["masked-by-a-later-v2-callback", "left-alone"],
)
async def test_post_call_success_hook_checks_token_data_of_a_new_answer_after_each_v2_callback(
    proxy_logging, make_user_api_key_auth, monkeypatch, callbacks, served_answer, served_token_ids
):
    monkeypatch.setattr(litellm, "callbacks", callbacks)

    served = await proxy_logging.post_call_success_hook(
        data={"model": "m", "messages": [{"role": "user", "content": "capital of Italy?"}]},
        response=_vllm_chat_response("The capital of Italy is Rome.", token_ids=_ROME_TOKEN_IDS, logprob=-0.3).model_dump(),
        user_api_key_dict=make_user_api_key_auth(),
    )

    choice = served["choices"][0]
    assert choice["message"]["content"] == served_answer
    assert choice["provider_specific_fields"]["token_ids"] == served_token_ids
    assert (choice["logprobs"] is None) is (served_token_ids is None)


def _responses_output_text(text: str, logprobs_requested: bool) -> ResponseOutputText:
    return ResponseOutputText(
        type="output_text",
        text=text,
        annotations=[],
        logprobs=[Logprob(token=text, logprob=-len(text) / 100, bytes=[], top_logprobs=[])] if logprobs_requested else [],
    )


class _ResponsesPartMaskingLogger(CustomLogger):
    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> ResponsesAPIResponse:
        assert isinstance(response, ResponsesAPIResponse)
        message = response.output[0]
        parts = [part.model_copy(update={"text": _masked(part.text)}) for part in message.content]
        return response.model_copy(update={"output": [message.model_copy(update={"content": parts})]})


class _ResponsesNoticeInsertingLogger(CustomLogger):
    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> ResponsesAPIResponse:
        masked = await _ResponsesPartMaskingLogger().async_post_call_success_hook(data, user_api_key_dict, response)
        notice = ResponseOutputMessage(
            id="msg_notice",
            type="message",
            role="assistant",
            status="completed",
            content=[_responses_output_text("Some words in this answer were masked.", logprobs_requested=False)],
        )
        return masked.model_copy(update={"output": [notice, *masked.output]})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("logprobs_requested", "masked_logprobs", "untouched_tokens"),
    [(True, None, ["Berlin is in Germany."]), (False, [], [])],
    ids=["logprobs-requested", "logprobs-not-requested"],
)
@pytest.mark.parametrize(
    "callback",
    [_ResponsesPartMaskingLogger(), _ResponsesNoticeInsertingLogger()],
    ids=["masked-in-place", "masked-after-an-inserted-notice"],
)
async def test_post_call_success_hook_drops_logprobs_only_from_responses_output_text_a_callback_rewrote(
    proxy_logging, make_user_api_key_auth, monkeypatch, logprobs_requested, masked_logprobs, untouched_tokens, callback
):
    monkeypatch.setattr(litellm, "callbacks", [callback])
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

    (message,) = (item for item in served.output if item.id == "msg_1")
    masked_part, untouched_part = message.content
    assert masked_part.text == "The capital of France is [MASKED]."
    assert masked_part.logprobs == masked_logprobs
    assert [logprob.token for logprob in untouched_part.logprobs] == untouched_tokens
