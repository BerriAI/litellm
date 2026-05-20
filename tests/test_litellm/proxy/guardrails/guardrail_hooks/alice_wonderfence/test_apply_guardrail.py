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
async def test_apply_guardrail_mask_replaces_last_text(
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
        inputs={"texts": ["a", "b", "c"]},
        request_data=make_request_data(),
        input_type="request",
    )
    assert out["texts"] == ["a", "b", "[REDACTED]"]


@pytest.mark.asyncio
async def test_apply_guardrail_mask_replaces_structured_messages(
    guardrail_and_client, make_request_data
):
    """MASK on the request path must rewrite structured_messages when that's
    the source of the extracted text. Otherwise the user's prompt reaches the
    LLM unredacted while the header still claims the guardrail applied."""
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "MASK"
    result_obj.action_text = "[REDACTED]"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_prompt.return_value = result_obj

    inputs = {
        "structured_messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "sensitive content"},
        ],
    }
    out = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data=make_request_data(),
        input_type="request",
    )
    last_user = [m for m in out["structured_messages"] if m.get("role") == "user"][-1]
    assert last_user["content"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_apply_guardrail_mask_rewrites_texts_when_both_slots_present(
    guardrail_and_client, make_request_data
):
    """OpenAI chat translation populates both ``structured_messages`` and ``texts``,
    then reads back only ``texts``. MASK must overwrite ``texts[-1]`` even when
    the analyzed text was extracted from ``structured_messages``, otherwise the
    unmasked ``texts`` slot wins downstream and the original prompt reaches the
    LLM while the response header still claims the guardrail applied."""
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "MASK"
    result_obj.action_text = "[REDACTED]"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_prompt.return_value = result_obj

    inputs = {
        "structured_messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "sensitive content"},
        ],
        "texts": ["first", "ack", "sensitive content"],
    }
    out = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data=make_request_data(),
        input_type="request",
    )
    assert out["texts"] == ["first", "ack", "[REDACTED]"]
    last_user = [m for m in out["structured_messages"] if m.get("role") == "user"][-1]
    assert last_user["content"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_apply_guardrail_mask_replaces_last_text_response(
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
        inputs={"texts": ["a", "b", "c"]},
        request_data=make_request_data(),
        input_type="response",
    )
    assert out["texts"] == ["a", "b", "[REDACTED]"]


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
        inputs={"texts": ["a", "b", "c"]},
        request_data=make_request_data(),
        input_type="request",
    )
    assert out["texts"] == ["a", "b", "[MASKED]"]


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
async def test_apply_guardrail_evaluates_only_last_text(
    guardrail_and_client, make_request_data
):
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
    assert client.evaluate_prompt.call_count == 1
    assert client.evaluate_prompt.call_args.kwargs["prompt"] == "t3"


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


def test_extract_relevant_text_uses_structured_messages():
    """Request path with structured_messages routes through get_last_user_message."""
    from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.processing import (
        extract_relevant_text,
    )

    inputs = {
        "structured_messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "latest user msg"},
        ],
        "texts": ["unused-fallback"],
    }
    text, source = extract_relevant_text(inputs, input_type="request")  # type: ignore[arg-type]
    assert text == "latest user msg"
    assert source == "structured_messages"
