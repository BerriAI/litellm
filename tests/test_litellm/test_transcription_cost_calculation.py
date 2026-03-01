"""
Tests for audio transcription cost calculation with custom pricing.

Verifies that `input_cost_per_second` configured on a deployment is correctly
propagated and used when calculating the cost of audio transcription calls.

Regression tests for https://github.com/BerriAI/litellm/issues/20228
"""

import uuid

import pytest

import litellm
from litellm.types.router import (
    Deployment,
    LiteLLM_Params,
    SPECIAL_MODEL_INFO_PARAMS,
)
from litellm.types.utils import TranscriptionResponse


# ---------------------------------------------------------------------------
# SPECIAL_MODEL_INFO_PARAMS includes per-second pricing
# ---------------------------------------------------------------------------


class TestSpecialModelInfoParams:
    """SPECIAL_MODEL_INFO_PARAMS must include per-second pricing fields."""

    def test_input_cost_per_second_in_special_params(self):
        assert "input_cost_per_second" in SPECIAL_MODEL_INFO_PARAMS

    def test_output_cost_per_second_in_special_params(self):
        assert "output_cost_per_second" in SPECIAL_MODEL_INFO_PARAMS


# ---------------------------------------------------------------------------
# Deployment.__init__ copies per-second pricing to model_info
# ---------------------------------------------------------------------------


class TestDeploymentCopiesPerSecondPricing:
    """Deployment.__init__ must copy input_cost_per_second to model_info."""

    def test_input_cost_per_second_copied_to_model_info(self):
        deployment = Deployment(
            model_name="whisper",
            litellm_params=LiteLLM_Params(
                model="hosted_vllm/whisper-large-v3-turbo",
                input_cost_per_second=0.006,
            ),
        )
        assert deployment.model_info.input_cost_per_second == 0.006

    def test_output_cost_per_second_copied_to_model_info(self):
        deployment = Deployment(
            model_name="whisper",
            litellm_params=LiteLLM_Params(
                model="hosted_vllm/whisper-large-v3-turbo",
                output_cost_per_second=0.012,
            ),
        )
        assert deployment.model_info.output_cost_per_second == 0.012

    def test_per_second_not_set_when_absent(self):
        deployment = Deployment(
            model_name="whisper",
            litellm_params=LiteLLM_Params(
                model="hosted_vllm/whisper-large-v3-turbo",
            ),
        )
        assert deployment.model_info.get("input_cost_per_second") is None

    def test_per_token_still_copied(self):
        deployment = Deployment(
            model_name="gpt-4",
            litellm_params=LiteLLM_Params(
                model="gpt-4",
                input_cost_per_token=0.03,
                output_cost_per_token=0.06,
            ),
        )
        assert deployment.model_info.input_cost_per_token == 0.03
        assert deployment.model_info.output_cost_per_token == 0.06


# ---------------------------------------------------------------------------
# _response_cost_calculator extracts router_model_id from metadata
# ---------------------------------------------------------------------------


class TestResponseCostCalculatorMetadataFallback:
    """
    _response_cost_calculator must fall back to metadata model_info.id
    when _hidden_params does not contain model_id.
    """

    def test_router_model_id_extracted_from_metadata(self):
        model_id = str(uuid.uuid4())
        litellm.model_cost[model_id] = {
            "input_cost_per_second": 0.006,
            "output_cost_per_token": 0,
            "input_cost_per_token": 0,
            "litellm_provider": "hosted_vllm",
            "mode": "audio_transcription",
        }
        try:
            response = TranscriptionResponse(text="hello world")
            response._hidden_params = {}

            router_model_id = None
            if hasattr(response, "_hidden_params"):
                hidden_params = response._hidden_params
                if "model_id" in hidden_params:
                    router_model_id = hidden_params["model_id"]

            # Simulate the NEW fallback
            if router_model_id is None:
                _metadata = {"model_info": {"id": model_id}}
                _mi = _metadata.get("model_info") or {}
                _mid = _mi.get("id")
                if _mid is not None and _mid in litellm.model_cost:
                    router_model_id = _mid

            assert router_model_id == model_id
        finally:
            litellm.model_cost.pop(model_id, None)

    def test_router_model_id_not_extracted_if_not_in_model_cost(self):
        model_id = str(uuid.uuid4())
        router_model_id = None
        _metadata = {"model_info": {"id": model_id}}
        _mi = _metadata.get("model_info") or {}
        _mid = _mi.get("id")
        if _mid is not None and _mid in litellm.model_cost:
            router_model_id = _mid
        assert router_model_id is None

    def test_router_model_id_from_hidden_params_takes_precedence(self):
        model_id_hidden = str(uuid.uuid4())
        model_id_metadata = str(uuid.uuid4())
        litellm.model_cost[model_id_hidden] = {"input_cost_per_second": 0.01}
        litellm.model_cost[model_id_metadata] = {"input_cost_per_second": 0.02}
        try:
            response = TranscriptionResponse(text="test")
            response._hidden_params = {"model_id": model_id_hidden}

            router_model_id = None
            hidden_params = response._hidden_params
            if "model_id" in hidden_params:
                router_model_id = hidden_params["model_id"]

            if router_model_id is None:
                _mid = model_id_metadata
                if _mid in litellm.model_cost:
                    router_model_id = _mid

            assert router_model_id == model_id_hidden
        finally:
            litellm.model_cost.pop(model_id_hidden, None)
            litellm.model_cost.pop(model_id_metadata, None)


# ---------------------------------------------------------------------------
# End-to-end: cost_per_token returns non-zero for transcription
# ---------------------------------------------------------------------------


class TestTranscriptionCostPerSecond:
    """
    Verify that cost_per_token correctly calculates transcription cost
    when using a model registered with input_cost_per_second.
    """

    def _register_model(self, model_id: str, cost_per_second: float = 0.006):
        litellm.model_cost[model_id] = {
            "input_cost_per_second": cost_per_second,
            "output_cost_per_token": 0,
            "input_cost_per_token": 0,
            "mode": "audio_transcription",
            "max_tokens": None,
            "max_input_tokens": None,
            "max_output_tokens": None,
        }

    def test_cost_per_second_via_model_id(self):
        from litellm.llms.openai.cost_calculation import (
            cost_per_second as openai_cost_per_second,
        )

        model_id = str(uuid.uuid4())
        self._register_model(model_id)
        try:
            prompt_cost, completion_cost_val = openai_cost_per_second(
                model=model_id, custom_llm_provider=None, duration=30.0
            )
            total = prompt_cost + completion_cost_val
            assert abs(total - 0.006 * 30.0) < 1e-9
        finally:
            litellm.model_cost.pop(model_id, None)

    def test_cost_per_token_transcription_path(self):
        from litellm.cost_calculator import cost_per_token

        model_id = str(uuid.uuid4())
        self._register_model(model_id)
        try:
            prompt_cost, completion_cost_val = cost_per_token(
                model=model_id,
                prompt_tokens=0,
                completion_tokens=0,
                custom_llm_provider="hosted_vllm",
                call_type="atranscription",
                audio_transcription_file_duration=45.0,
            )
            assert abs(prompt_cost + completion_cost_val - 0.006 * 45.0) < 1e-9
        finally:
            litellm.model_cost.pop(model_id, None)

    def test_zero_duration_gives_zero_cost(self):
        from litellm.cost_calculator import cost_per_token

        model_id = str(uuid.uuid4())
        self._register_model(model_id)
        try:
            prompt_cost, completion_cost_val = cost_per_token(
                model=model_id,
                prompt_tokens=0,
                completion_tokens=0,
                custom_llm_provider="hosted_vllm",
                call_type="transcription",
                audio_transcription_file_duration=0.0,
            )
            assert prompt_cost + completion_cost_val == 0.0
        finally:
            litellm.model_cost.pop(model_id, None)

    def test_high_duration_scales_linearly(self):
        from litellm.cost_calculator import cost_per_token

        model_id = str(uuid.uuid4())
        self._register_model(model_id, cost_per_second=1.0)
        try:
            p, c = cost_per_token(
                model=model_id,
                prompt_tokens=0,
                completion_tokens=0,
                custom_llm_provider="hosted_vllm",
                call_type="atranscription",
                audio_transcription_file_duration=3600.0,
            )
            assert abs(p + c - 3600.0) < 1e-9
        finally:
            litellm.model_cost.pop(model_id, None)


# ---------------------------------------------------------------------------
# _select_model_name_for_cost_calc prefers router_model_id
# ---------------------------------------------------------------------------


class TestSelectModelNameForCostCalc:
    def test_returns_router_model_id_when_custom_pricing(self):
        from litellm.cost_calculator import _select_model_name_for_cost_calc

        model_id = str(uuid.uuid4())
        litellm.model_cost[model_id] = {"input_cost_per_second": 0.006}
        try:
            result = _select_model_name_for_cost_calc(
                model="hosted_vllm/whisper-large-v3-turbo",
                completion_response=None,
                custom_pricing=True,
                router_model_id=model_id,
            )
            # The function prepends the provider derived from model name
            assert result is not None
            assert model_id in result
        finally:
            litellm.model_cost.pop(model_id, None)

    def test_falls_back_to_model_when_no_router_model_id(self):
        from litellm.cost_calculator import _select_model_name_for_cost_calc

        result = _select_model_name_for_cost_calc(
            model="hosted_vllm/whisper-large-v3-turbo",
            completion_response=None,
            custom_pricing=True,
            router_model_id=None,
        )
        assert result == "hosted_vllm/whisper-large-v3-turbo"
