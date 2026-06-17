"""Tests for ``apply_guardrail`` BLOCK/MASK/DETECT/NO_ACTION + fail modes + helpers."""

import sys
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

# ----------------------------- BLOCK -----------------------------


@pytest.mark.asyncio
async def test_apply_guardrail_block_action(guardrail_and_client, make_request_data):
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "BLOCK"
    detection = Mock()
    detection.model_dump = Mock(return_value={"policy_name": "x", "confidence": 0.9})
    result_obj.detections = [detection]
    result_obj.correlation_id = "corr-1"
    client.evaluate_prompt.return_value = result_obj

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 400
    assert exc.value.detail["action"] == "BLOCK"
    assert exc.value.detail["wonderfence_correlation_id"] == "corr-1"
    assert exc.value.detail["error"] == (
        "Content violates our policies and has been blocked"
    )
    assert exc.value.detail["detections"][0]["policy_name"] == "x"


@pytest.mark.asyncio
async def test_apply_guardrail_block_uses_custom_block_message(
    make_guardrail, make_request_data
):
    guardrail, client = make_guardrail(block_message="custom blocked text")
    guardrail._client_cache["default-api-key"] = client
    result_obj = Mock()
    result_obj.action = "BLOCK"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_prompt.return_value = result_obj

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.detail["error"] == "custom blocked text"


@pytest.mark.asyncio
async def test_block_not_bypassed_by_fail_open(make_guardrail, make_request_data):
    guardrail, client = make_guardrail(fail_open=True)
    guardrail._client_cache["default-api-key"] = client
    result_obj = Mock()
    result_obj.action = "BLOCK"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_prompt.return_value = result_obj

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["bad"]},
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 400


# ----------------------------- MASK -----------------------------


@pytest.mark.asyncio
async def test_apply_guardrail_mask_replaces_scanned_text(
    guardrail_and_client, make_request_data
):
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "MASK"
    result_obj.action_text = "[REDACTED]"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_prompt.return_value = result_obj

    out = await guardrail.apply_guardrail(
        inputs={"texts": ["sensitive"]},
        request_data=make_request_data(),
        input_type="request",
    )
    assert out["texts"] == ["[REDACTED]"]


@pytest.mark.asyncio
async def test_apply_guardrail_mask_targets_only_the_flagged_slot(
    guardrail_and_client, make_request_data
):
    """MASK rewrites the ``texts`` entry of the flagged segment in place; the
    other scanned entries survive untouched. Confirms positional 1:1 mapping."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "MASK" if prompt == "sensitive content" else "NO_ACTION"
        r.action_text = "[REDACTED]"
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    out = await guardrail.apply_guardrail(
        inputs={"texts": ["first", "ack", "sensitive content"]},
        request_data=make_request_data(),
        input_type="request",
    )
    assert out["texts"] == ["first", "ack", "[REDACTED]"]


@pytest.mark.asyncio
async def test_apply_guardrail_scans_non_user_role_segments(
    guardrail_and_client, make_request_data
):
    """Bypass regression: blocked content in a system/assistant/tool message
    must still BLOCK. The translation layer already strips system/tool when the
    guardrail is configured to skip them, so whatever remains in ``texts`` is
    scanned regardless of role; the hook must not re-filter to user-only."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "BLOCK" if prompt == "disallowed system instruction" else "NO_ACTION"
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    inputs = {
        "structured_messages": [
            {"role": "system", "content": "disallowed system instruction"},
            {"role": "user", "content": "hello"},
        ],
        "texts": ["disallowed system instruction", "hello"],
    }
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 400
    assert exc.value.detail["action"] == "BLOCK"


def _tool_call(arguments, name="send_email"):
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


@pytest.mark.asyncio
async def test_apply_guardrail_blocks_on_tool_call_arguments(
    guardrail_and_client, make_request_data
):
    """Bypass regression: blocked content in tool_calls[].function.arguments must
    BLOCK. tool_calls reach the model but were never scanned (texts-only)."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "BLOCK" if "DISALLOWED" in prompt else "NO_ACTION"
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    inputs = {
        "texts": ["please run the tool"],
        "tool_calls": [_tool_call('{"body": "DISALLOWED payload"}')],
    }
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 400
    assert exc.value.detail["action"] == "BLOCK"


@pytest.mark.asyncio
async def test_apply_guardrail_masks_tool_call_arguments_in_place(
    guardrail_and_client, make_request_data
):
    """MASK on a tool-call argument string rewrites
    inputs['tool_calls'][i]['function']['arguments']."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "MASK" if "secret" in prompt else "NO_ACTION"
        r.action_text = '{"body": "[REDACTED]"}'
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    inputs = {
        "texts": ["benign"],
        "tool_calls": [_tool_call('{"body": "secret value"}')],
    }
    out = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data=make_request_data(),
        input_type="request",
    )
    assert out["tool_calls"][0]["function"]["arguments"] == '{"body": "[REDACTED]"}'
    assert out["texts"] == ["benign"]


@pytest.mark.asyncio
async def test_apply_guardrail_detect_on_tool_call_args_passes_through(
    guardrail_and_client, make_request_data
):
    """A DETECT verdict on a tool-call argument logs but does not block or mutate
    the arguments (symmetric with the text-side DETECT behavior)."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "DETECT" if "watch me" in prompt else "NO_ACTION"
        r.action_text = None
        r.detections = []
        r.correlation_id = "corr-detect"
        return r

    client.evaluate_prompt.side_effect = evaluate

    inputs = {"texts": ["benign"], "tool_calls": [_tool_call('{"x": "watch me"}')]}
    out = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data=make_request_data(),
        input_type="request",
    )
    assert out["tool_calls"][0]["function"]["arguments"] == '{"x": "watch me"}'
    assert out["texts"] == ["benign"]


@pytest.mark.asyncio
async def test_apply_guardrail_scans_tool_calls_when_no_texts(
    guardrail_and_client, make_request_data
):
    """An assistant message can carry tool_calls with no text content, so texts
    is empty; the hook must still scan the tool-call arguments (the old
    empty-texts early return skipped them)."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "BLOCK" if "DISALLOWED" in prompt else "NO_ACTION"
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": [], "tool_calls": [_tool_call('{"x": "DISALLOWED"}')]},
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_apply_guardrail_blocks_on_response_tool_call_arguments(
    guardrail_and_client, make_request_data
):
    """Model-generated tool-call arguments on the response side are scanned too."""
    guardrail, client = guardrail_and_client

    def evaluate(response, **kwargs):
        r = Mock()
        r.action = "BLOCK" if "DISALLOWED" in response else "NO_ACTION"
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_response.side_effect = evaluate

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": [], "tool_calls": [_tool_call('{"x": "DISALLOWED"}')]},
            request_data=make_request_data(),
            input_type="response",
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_apply_guardrail_mask_replaces_scanned_text_response(
    guardrail_and_client, make_request_data
):
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "MASK"
    result_obj.action_text = "[REDACTED]"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_response.return_value = result_obj

    out = await guardrail.apply_guardrail(
        inputs={"texts": ["model output"]},
        request_data=make_request_data(),
        input_type="response",
    )
    assert out["texts"] == ["[REDACTED]"]


@pytest.mark.asyncio
async def test_apply_guardrail_mask_fallback_when_action_text_is_none(
    guardrail_and_client, make_request_data
):
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "MASK"
    result_obj.action_text = None
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_prompt.return_value = result_obj

    out = await guardrail.apply_guardrail(
        inputs={"texts": ["a"]},
        request_data=make_request_data(),
        input_type="request",
    )
    assert out["texts"] == ["[MASKED]"]


# ----------------------------- DETECT / NO_ACTION -----------------------------


@pytest.mark.asyncio
async def test_apply_guardrail_no_action_passthrough(
    guardrail_and_client, make_request_data
):
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "NO_ACTION"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_prompt.return_value = result_obj

    out = await guardrail.apply_guardrail(
        inputs={"texts": ["safe"]},
        request_data=make_request_data(),
        input_type="request",
    )
    assert out["texts"] == ["safe"]
    client.evaluate_prompt.assert_awaited_once()


@pytest.mark.asyncio
async def test_apply_guardrail_detect_action_passes_through(
    guardrail_and_client, make_request_data
):
    """DETECT action logs a warning but does not block or mutate inputs."""
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "DETECT"
    result_obj.detections = []
    result_obj.correlation_id = "corr-detect"
    client.evaluate_prompt.return_value = result_obj

    out = await guardrail.apply_guardrail(
        inputs={"texts": ["watch me"]},
        request_data=make_request_data(),
        input_type="request",
    )
    assert out["texts"] == ["watch me"]
    client.evaluate_prompt.assert_awaited_once()


# ----------------------------- core path / app_id passthrough -----------------------------


@pytest.mark.asyncio
async def test_apply_guardrail_passes_app_id_per_call(
    guardrail_and_client, make_request_data
):
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "NO_ACTION"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_prompt.return_value = result_obj

    await guardrail.apply_guardrail(
        inputs={"texts": ["hi"]},
        request_data=make_request_data(
            metadata={"user_api_key_metadata": {"alice_wonderfence_app_id": "tenant-A"}}
        ),
        input_type="request",
    )
    kwargs = client.evaluate_prompt.call_args.kwargs
    assert kwargs["app_id"] == "tenant-A"
    assert kwargs["prompt"] == "hi"
    assert kwargs["custom_fields"] is None


@pytest.mark.asyncio
async def test_apply_guardrail_response_path_passes_app_id(
    make_guardrail, make_request_data
):
    guardrail, client = make_guardrail()
    guardrail._client_cache["default-api-key"] = client
    result_obj = Mock()
    result_obj.action = "NO_ACTION"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_response.return_value = result_obj

    await guardrail.apply_guardrail(
        inputs={"texts": ["resp"]},
        request_data=make_request_data(
            metadata={"user_api_key_metadata": {"alice_wonderfence_app_id": "tenant-B"}}
        ),
        input_type="response",
    )
    kwargs = client.evaluate_response.call_args.kwargs
    assert kwargs["app_id"] == "tenant-B"
    assert kwargs["response"] == "resp"


@pytest.mark.asyncio
async def test_apply_guardrail_evaluates_every_text_without_structured_messages(
    guardrail_and_client, make_request_data
):
    """With no structured_messages to identify roles, every text entry is
    scanned (over-scan is safe); the old code scanned only the last."""
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "NO_ACTION"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_prompt.return_value = result_obj

    await guardrail.apply_guardrail(
        inputs={"texts": ["t1", "t2", "t3"]},
        request_data=make_request_data(),
        input_type="request",
    )
    assert client.evaluate_prompt.call_count == 3
    prompts = {c.kwargs["prompt"] for c in client.evaluate_prompt.call_args_list}
    assert prompts == {"t1", "t2", "t3"}


@pytest.mark.asyncio
async def test_apply_guardrail_blocks_on_earlier_user_turn(
    guardrail_and_client, make_request_data
):
    """Bypass regression: disallowed content in an earlier user turn followed by
    a benign final turn must still BLOCK. The old last-only path only saw the
    benign final message and let the request through."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "BLOCK" if prompt == "disallowed" else "NO_ACTION"
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    inputs = {
        "structured_messages": [
            {"role": "user", "content": "disallowed"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "hello"},
        ],
        "texts": ["disallowed", "ok", "hello"],
    }
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 400
    assert exc.value.detail["action"] == "BLOCK"


@pytest.mark.asyncio
async def test_apply_guardrail_blocks_when_oversized_message_trips_in_late_chunk(
    guardrail_and_client, make_request_data
):
    """A single user message over the prompt limit is chunked; a BLOCK in a
    non-first chunk still blocks the request."""
    from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence import (
        chunked_evaluation,
    )

    guardrail, client = guardrail_and_client
    long_prompt = ("safe " * 5000) + "TRIPWIRE"
    assert len(long_prompt) > chunked_evaluation.MAX_PROMPT_CHARS

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "BLOCK" if "TRIPWIRE" in prompt else "NO_ACTION"
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": [long_prompt]},
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 400
    assert client.evaluate_prompt.call_count > 1


@pytest.mark.asyncio
async def test_apply_guardrail_no_text_short_circuits(
    guardrail_and_client, make_request_data
):
    """Empty inputs must skip the SDK call and return inputs unchanged."""
    guardrail, client = guardrail_and_client
    out = await guardrail.apply_guardrail(
        inputs={"texts": []},
        request_data=make_request_data(),
        input_type="request",
    )
    assert out == {"texts": []}
    client.evaluate_prompt.assert_not_awaited()
    client.evaluate_response.assert_not_awaited()


# ----------------------------- fail modes -----------------------------


@pytest.mark.asyncio
async def test_apply_guardrail_missing_app_id_fail_closed_returns_500(
    guardrail_and_client, make_request_data
):
    """Missing app_id follows the fail_open pattern: fail_open=False → HTTP 500."""
    guardrail, _ = guardrail_and_client
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=make_request_data(metadata={}),
            input_type="request",
        )
    assert exc.value.status_code == 500
    assert "Error in Alice WonderFence Guardrail" in exc.value.detail["error"]
    assert "alice_wonderfence_app_id" in exc.value.detail["exception"]


@pytest.mark.asyncio
async def test_apply_guardrail_missing_api_key_fail_closed_returns_500(
    monkeypatch, make_guardrail, make_request_data
):
    """Missing api_key follows the fail_open pattern: fail_open=False → HTTP 500."""
    monkeypatch.delenv("ALICE_API_KEY", raising=False)
    guardrail, _ = make_guardrail(api_key=None)
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 500
    assert "Error in Alice WonderFence Guardrail" in exc.value.detail["error"]
    assert "alice_wonderfence_api_key" in exc.value.detail["exception"]


@pytest.mark.asyncio
async def test_apply_guardrail_missing_app_id_fail_open_returns_500(
    make_guardrail, make_request_data
):
    """Missing app_id is a config error: never fail-open, even with fail_open=True."""
    guardrail, _ = make_guardrail(fail_open=True)
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=make_request_data(metadata={}),
            input_type="request",
        )
    assert exc.value.status_code == 500
    assert "alice_wonderfence_app_id" in exc.value.detail["exception"]


@pytest.mark.asyncio
async def test_apply_guardrail_missing_api_key_fail_open_returns_500(
    monkeypatch, make_guardrail, make_request_data
):
    """Missing api_key is a config error: never fail-open, even with fail_open=True."""
    monkeypatch.delenv("ALICE_API_KEY", raising=False)
    guardrail, _ = make_guardrail(api_key=None, fail_open=True)
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 500
    assert "alice_wonderfence_api_key" in exc.value.detail["exception"]


@pytest.mark.asyncio
async def test_apply_guardrail_fail_open_swallows_transport_error(
    make_guardrail, make_request_data
):
    guardrail, client = make_guardrail(fail_open=True)
    guardrail._client_cache["default-api-key"] = client
    client.evaluate_prompt.side_effect = RuntimeError("network down")

    inputs = {"texts": ["original"]}
    out = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data=make_request_data(),
        input_type="request",
    )
    assert out["texts"] == ["original"]


@pytest.mark.asyncio
async def test_apply_guardrail_fail_closed_returns_500(
    guardrail_and_client, make_request_data
):
    guardrail, client = guardrail_and_client
    client.evaluate_prompt.side_effect = RuntimeError("network down")

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 500
    assert "Error in Alice WonderFence Guardrail" in exc.value.detail["error"]


# ----------------------------- helpers -----------------------------


def test_get_config_model(make_guardrail):
    from litellm.types.proxy.guardrails.guardrail_hooks.alice_wonderfence import (
        WonderFenceGuardrailConfigModel,
    )

    guardrail, _ = make_guardrail()
    assert guardrail.get_config_model() is WonderFenceGuardrailConfigModel


def test_build_analysis_context_falls_back_to_slash_split(monkeypatch, make_guardrail):
    """When ``litellm.get_llm_provider`` raises, fall back to ``provider/model`` split."""
    import litellm

    from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.processing import (
        build_analysis_context,
    )

    guardrail, _ = make_guardrail()

    def boom(model):
        raise ValueError("unknown provider")

    monkeypatch.setattr(litellm, "get_llm_provider", boom)
    build_analysis_context(
        {"model": "myorg/custom-llm"}, guardrail.platform, guardrail._AnalysisContext
    )

    AnalysisContext = sys.modules["wonderfence_sdk.models"].AnalysisContext
    kwargs = AnalysisContext.call_args.kwargs
    assert kwargs["provider"] == "myorg"
    assert kwargs["model_name"] == "custom-llm"


@pytest.mark.asyncio
async def test_malformed_override_does_not_fail_open(make_guardrail, make_request_data):
    """A non-string request-metadata app_id override must not slip through under
    fail_open: it resolves to a config error (500), not a swallowed exception
    that skips scanning. The SDK is never called with a malformed value."""
    guardrail, client = make_guardrail(
        fail_open=True, allow_request_metadata_override=True
    )
    guardrail._client_cache["default-api-key"] = client

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=make_request_data(
                metadata={"alice_wonderfence_app_id": ["not", "a", "string"]}
            ),
            input_type="request",
        )
    assert exc.value.status_code == 500
    assert "alice_wonderfence_app_id" in exc.value.detail["exception"]
    client.evaluate_prompt.assert_not_awaited()
