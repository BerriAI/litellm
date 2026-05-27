"""
Unit tests for SageMaker chat transformation (`sagemaker_chat`).

Regression coverage for LIT-2853: `sagemaker_chat` should mirror the legacy
`sagemaker` completion handler and translate the `model_id` optional param
into an `X-Amzn-SageMaker-Inference-Component` request header.
"""

import pytest

from litellm.llms.sagemaker.chat.transformation import SagemakerChatConfig


class TestSagemakerChatValidateEnvironmentModelId:
    def setup_method(self):
        self.config = SagemakerChatConfig()

    def _call(self, headers, optional_params):
        return self.config.validate_environment(
            headers=headers,
            model="dummy-endpoint",
            messages=[{"role": "user", "content": "hi"}],
            optional_params=optional_params,
            litellm_params={},
            api_key=None,
            api_base=None,
        )

    def test_model_id_added_as_inference_component_header(self):
        """When `model_id` is set on optional_params, the SageMaker
        InferenceComponent header should be injected."""
        headers: dict = {}
        optional_params: dict = {"model_id": "my-component-1"}
        out = self._call(headers, optional_params)
        assert out["X-Amzn-SageMaker-Inference-Component"] == "my-component-1"
        # And it should be popped so it does not leak into the request body
        # via OpenAIGPTConfig.transform_request's **optional_params spread.
        assert "model_id" not in optional_params

    def test_no_model_id_means_no_inference_component_header(self):
        """When `model_id` is not set, the SageMaker InferenceComponent header
        must not be present."""
        headers: dict = {}
        optional_params: dict = {}
        out = self._call(headers, optional_params)
        assert "X-Amzn-SageMaker-Inference-Component" not in out

    def test_preserves_existing_headers(self):
        """Existing headers should not be dropped when injecting the
        InferenceComponent header."""
        headers = {"X-Custom-Header": "foo"}
        optional_params = {"model_id": "comp-42"}
        out = self._call(headers, optional_params)
        assert out["X-Custom-Header"] == "foo"
        assert out["X-Amzn-SageMaker-Inference-Component"] == "comp-42"

    def test_empty_string_model_id_is_treated_as_unset(self):
        """An empty `model_id` is falsy and should not produce a blank
        component header — SageMaker rejects empty component names."""
        headers: dict = {}
        optional_params: dict = {"model_id": ""}
        out = self._call(headers, optional_params)
        assert "X-Amzn-SageMaker-Inference-Component" not in out
        # The empty string is still consumed so it does not leak to the body.
        assert "model_id" not in optional_params

    def test_model_id_takes_precedence_over_explicit_header(self):
        """If a caller passes both an explicit
        `X-Amzn-SageMaker-Inference-Component` header (e.g. via
        `extra_headers`) and `model_id`, the `model_id`-derived value wins
        because it is the more recently-specified explicit `model_id` kwarg.

        This mirrors the behavior of the legacy `sagemaker` completion handler
        (`prepared_request.headers.update(...)`), which also overwrites any
        existing entry.
        """
        headers = {"X-Amzn-SageMaker-Inference-Component": "from-extra-headers"}
        optional_params = {"model_id": "from-model-id"}
        out = self._call(headers, optional_params)
        assert out["X-Amzn-SageMaker-Inference-Component"] == "from-model-id"

    def test_whitespace_only_model_id_is_treated_as_unset(self):
        """A whitespace-only `model_id` (e.g. `" "`) must not produce a
        non-empty component header — SageMaker rejects whitespace-only
        InferenceComponent names."""
        headers: dict = {}
        optional_params: dict = {"model_id": "   "}
        out = self._call(headers, optional_params)
        assert "X-Amzn-SageMaker-Inference-Component" not in out
        assert "model_id" not in optional_params

