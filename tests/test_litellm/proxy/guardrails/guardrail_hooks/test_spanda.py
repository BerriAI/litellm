import pytest

from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.spanda import (
    SpandaGuardrail,
    guardrail_class_registry,
    guardrail_initializer_registry,
    initialize_guardrail,
)
from litellm.types.guardrails import LitellmParams, SupportedGuardrailIntegrations
from litellm.types.proxy.guardrails.guardrail_hooks.spanda import (
    SpandaGuardrailConfigModel,
)


def test_enum_value():
    assert SupportedGuardrailIntegrations.SPANDA.value == "spanda"


def test_config_model_ui_name_and_instantiation():
    assert SpandaGuardrailConfigModel.ui_friendly_name() == "Spanda (BRHMN Labs)"
    model = SpandaGuardrailConfigModel(uncertainty_threshold=0.40, block_mode=True)
    assert model.uncertainty_threshold == 0.40
    assert model.block_mode is True


def test_get_config_model_returns_config_model():
    g = SpandaGuardrail()
    assert g.get_config_model() is SpandaGuardrailConfigModel


def test_registries_expose_initializer_and_class():
    assert "spanda" in guardrail_initializer_registry
    assert guardrail_class_registry["spanda"] is SpandaGuardrail


def test_litellm_params_includes_config_model():
    assert SpandaGuardrailConfigModel in LitellmParams.__mro__


def test_config_driven_initialization_creates_callback():
    lp = LitellmParams(
        guardrail="spanda",
        mode="post_call",
        uncertainty_threshold=0.30,
        block_mode=True,
    )
    cb = initialize_guardrail(lp, {"guardrail_name": "my-spanda-guard"})
    assert isinstance(cb, SpandaGuardrail)
    assert cb.uncertainty_threshold == 0.30
    assert cb.block_mode is True
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
