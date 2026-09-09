from types import SimpleNamespace

import pytest

from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.spanda import (
    SpandaGuardrail,
    initialize_guardrail,
)
from litellm.types.guardrails import LitellmParams, SupportedGuardrailIntegrations
from litellm.types.proxy.guardrails.guardrail_hooks.spanda import (
    SpandaGuardrailConfigModel,
)


def test_enum_value():
    assert SupportedGuardrailIntegrations.SPANDA.value == "spanda"


def test_config_model_instantiation_and_defaults():
    assert SpandaGuardrailConfigModel.ui_friendly_name() == "Spanda (BRHMN Labs)"
    model = SpandaGuardrailConfigModel(uncertainty_threshold=0.40, block_mode=True)
    assert model.uncertainty_threshold == 0.40
    assert model.block_mode is True


def test_config_driven_initialization_with_defaults():
    lp = LitellmParams(
        guardrail="spanda",
        mode="post_call",
        uncertainty_threshold=None,
        block_mode=None,
    )
    cb = initialize_guardrail(lp, {"guardrail_name": "my-spanda-guard"})
    assert isinstance(cb, SpandaGuardrail)
    assert cb.uncertainty_threshold == 0.35
    assert cb.grounding_threshold == 0.15
    assert cb.block_mode is False
    assert cb.guardrail_name == "my-spanda-guard"


def test_evaluate_texts_consistent_pass():
    g = SpandaGuardrail(uncertainty_threshold=0.35)
    samples = ["The answer is 42.", "The answer is 42.", "The answer is 42."]
    result = g.evaluate_texts(samples)
    assert result["is_safe"] is True
    assert result["decision"] == "PASS"
    assert result["rsc"] == 0.0


def test_evaluate_texts_divergent_flag():
    g = SpandaGuardrail(uncertainty_threshold=0.35)
    samples = [
        "The capital of France is Paris.",
        "Berlin is the capital of Germany.",
        "Tokyo is in Japan.",
    ]
    result = g.evaluate_texts(samples)
    assert result["is_safe"] is False
    assert result["decision"] == "FLAG_HIGH_UNCERTAINTY"
    assert result["rsc"] > 0.35


def test_evaluate_texts_single_sample_unchecked():
    g = SpandaGuardrail()
    result = g.evaluate_texts(["Sole completion without context"])
    assert result["is_safe"] is True
    assert result["decision"] == "PASS_SINGLE_SAMPLE_UNCHECKED"


def test_evaluate_texts_grounding_flag():
    g = SpandaGuardrail(grounding_threshold=0.10)
    context = "Photosynthesis occurs in plants converting sunlight into sugars."
    response = "Quantum mechanics governs subatomic particle interactions and wavefunction collapses."
    result = g.evaluate_texts([response], context=context)
    assert result["is_safe"] is False
    assert result["decision"] == "FLAG_UNGROUNDED"
    assert result["grounding_residual"] > 0.10


def test_sample_clamping_and_length_truncation():
    g = SpandaGuardrail()
    long_samples = ["Repeated text string " * 300 for _ in range(15)]
    result = g.evaluate_texts(long_samples)
    assert result["is_safe"] is True
    assert result["rsc"] == 0.0


@pytest.mark.asyncio
async def test_apply_guardrail_blocking():
    g = SpandaGuardrail(uncertainty_threshold=0.35, block_mode=True)
    inputs = {
        "texts": [
            "The capital of France is Paris.",
            "Berlin is the capital of Germany.",
            "Tokyo is in Japan.",
        ]
    }
    request_data = {"messages": [{"role": "user", "content": "Tell me a capital"}]}
    with pytest.raises(GuardrailRaisedException) as exc_info:
        await g.apply_guardrail(inputs, request_data, input_type="response")

    assert "Spanda guardrail intervention" in str(exc_info.value)
    assert exc_info.value.guardrail_name == "spanda"


@pytest.mark.asyncio
async def test_apply_guardrail_non_blocking():
    g = SpandaGuardrail(uncertainty_threshold=0.35, block_mode=False)
    inputs = {
        "texts": [
            "The capital of France is Paris.",
            "Berlin is the capital of Germany.",
            "Tokyo is in Japan.",
        ]
    }
    request_data = {"messages": [{"role": "user", "content": "Tell me a capital"}]}
    result = await g.apply_guardrail(inputs, request_data, input_type="response")
    assert result == inputs
    assert "_spanda_receipt" in request_data
    assert request_data["_spanda_receipt"]["decision"] == "FLAG_HIGH_UNCERTAINTY"


@pytest.mark.asyncio
async def test_apply_guardrail_pass():
    g = SpandaGuardrail(uncertainty_threshold=0.35, block_mode=True)
    inputs = {"texts": ["The answer is 42.", "The answer is 42."]}
    request_data = {"messages": [{"role": "user", "content": "What is the answer?"}]}
    result = await g.apply_guardrail(inputs, request_data, input_type="response")
    assert result == inputs
    assert request_data["_spanda_receipt"]["decision"] == "PASS"


@pytest.mark.asyncio
async def test_apply_guardrail_ignores_request_type():
    g = SpandaGuardrail()
    inputs = {"texts": ["User message"]}
    request_data = {}
    result = await g.apply_guardrail(inputs, request_data, input_type="request")
    assert result == inputs
    assert "_spanda_receipt" not in request_data


@pytest.mark.asyncio
async def test_apply_guardrail_empty_texts():
    g = SpandaGuardrail()
    inputs = {"texts": []}
    request_data = {}
    result = await g.apply_guardrail(inputs, request_data, input_type="response")
    assert result == inputs


@pytest.mark.asyncio
async def test_async_post_call_success_hook_dict_response():
    g = SpandaGuardrail(uncertainty_threshold=0.35, block_mode=False)
    data = {"messages": [{"role": "user", "content": "Hello"}]}
    response = {
        "choices": [
            {"message": {"content": "Sample A"}},
            {"message": {"content": "Sample B"}},
        ]
    }
    await g.async_post_call_success_hook(data, None, response)
    assert "_spanda_receipt" in response
    assert response["_spanda_receipt"]["samples_analyzed"] == 2


@pytest.mark.asyncio
async def test_async_post_call_success_hook_object_response():
    g = SpandaGuardrail(uncertainty_threshold=0.35, block_mode=False)
    data = {"messages": [{"role": "system", "content": "You are helpful."}]}
    c1 = SimpleNamespace(message=SimpleNamespace(content="Identical answer"))
    c2 = SimpleNamespace(message=SimpleNamespace(content="Identical answer"))
    response = SimpleNamespace(choices=[c1, c2], model_extra={})
    await g.async_post_call_success_hook(data, None, response)
    assert hasattr(response, "_spanda_receipt")
    assert response._spanda_receipt["decision"] == "PASS"
    assert response.model_extra["_spanda_receipt"]["decision"] == "PASS"


@pytest.mark.asyncio
async def test_async_post_call_success_hook_blocking():
    g = SpandaGuardrail(uncertainty_threshold=0.35, block_mode=True)
    data = {"messages": [{"role": "user", "content": "Prompt"}]}
    c1 = SimpleNamespace(message=SimpleNamespace(content="Answer 1"))
    c2 = SimpleNamespace(message=SimpleNamespace(content="Completely different 2"))
    response = SimpleNamespace(choices=[c1, c2])
    with pytest.raises(GuardrailRaisedException):
        await g.async_post_call_success_hook(data, None, response)


def test_zero_threshold_preservation():
    lp = LitellmParams(
        guardrail="spanda",
        mode="post_call",
        uncertainty_threshold=0.0,
        grounding_threshold=0.0,
        block_mode=True,
    )
    cb = initialize_guardrail(lp, {"guardrail_name": "zero-tolerance-spanda"})
    assert cb.uncertainty_threshold == 0.0
    assert cb.grounding_threshold == 0.0


def test_use_native_lifecycle_hooks_flag():
    assert SpandaGuardrail.use_native_lifecycle_hooks is True
