"""Tests for the built-in Sensitive Data Routing guardrail."""

import pytest
from pydantic import ValidationError

from litellm.exceptions import SensitiveDataRouteException
from litellm.proxy.guardrails.guardrail_hooks.sensitive_data_routing import (
    SensitiveDataRoutingGuardrail,
    initialize_guardrail,
)
from litellm.proxy.guardrails.guardrail_registry import (
    IN_MEMORY_GUARDRAIL_HANDLER,
    guardrail_class_registry,
    guardrail_initializer_registry,
)
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams
from litellm.types.proxy.guardrails.guardrail_hooks.sensitive_data_routing import (
    SensitiveDataRoutingGuardrailConfigModel,
)

DOCUMENTED_LITELLM_PARAMS = {
    "guardrail": "sensitive_data_routing",
    "mode": "pre_call",
    "default_on": True,
    "on_premise_model": "on-prem-model",
    "prebuilt_patterns": ["us_ssn", "credit_card", "email"],
    "regex_patterns": [r"project\s+titan"],
    "keywords": ["confidential", "internal only"],
    "sticky_session": True,
    "session_ttl_seconds": 14400,
}


def make_guardrail(**overrides) -> SensitiveDataRoutingGuardrail:
    litellm_params = LitellmParams(**{**DOCUMENTED_LITELLM_PARAMS, **overrides})
    return initialize_guardrail(litellm_params, {"guardrail_name": "sensitive-data-routing"})


class TestSensitiveDataRoutingGuardrailRegistration:
    def test_guardrail_type_is_registered(self):
        """Regression: `guardrail: sensitive_data_routing` used to raise "Unsupported guardrail"."""
        assert "sensitive_data_routing" in guardrail_initializer_registry
        assert guardrail_class_registry["sensitive_data_routing"] is SensitiveDataRoutingGuardrail

    def test_documented_config_initializes_through_the_registry(self):
        """The config.yaml from the docs initializes instead of failing proxy startup."""
        guardrail = IN_MEMORY_GUARDRAIL_HANDLER.initialize_guardrail(
            {
                "guardrail_name": "docs-sensitive-data-routing",
                "litellm_params": dict(DOCUMENTED_LITELLM_PARAMS),
            }
        )

        assert guardrail is not None
        callback = IN_MEMORY_GUARDRAIL_HANDLER.guardrail_id_to_custom_guardrail[guardrail["guardrail_id"]]
        assert isinstance(callback, SensitiveDataRoutingGuardrail)
        assert callback.on_premise_model == "on-prem-model"
        assert callback.event_hook == GuardrailEventHooks.pre_call

    def test_on_premise_model_is_required(self):
        with pytest.raises(ValueError, match="on_premise_model"):
            make_guardrail(on_premise_model=None)

    def test_at_least_one_detector_is_required(self):
        with pytest.raises(ValueError, match="prebuilt_patterns"):
            make_guardrail(prebuilt_patterns=None, regex_patterns=None, keywords=None)

    def test_unknown_prebuilt_pattern_fails_fast(self):
        with pytest.raises(ValueError, match="Unknown pattern name"):
            make_guardrail(prebuilt_patterns=["not_a_real_pattern"])

    @pytest.mark.parametrize("ttl", [0, -1])
    def test_non_positive_session_ttl_is_rejected(self, ttl):
        """A non-positive TTL expires the pin immediately, so reject it at config time
        instead of silently never pinning the session."""
        with pytest.raises(ValidationError):
            SensitiveDataRoutingGuardrailConfigModel(
                on_premise_model="on-prem-model",
                keywords=["confidential"],
                session_ttl_seconds=ttl,
            )


class TestSensitiveDataRoutingGuardrailDetection:
    @pytest.mark.asyncio
    async def test_clean_request_is_untouched(self):
        guardrail = make_guardrail()
        inputs = {"texts": ["What is the capital of France?"]}

        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"metadata": {"session_id": "abc-123"}},
            input_type="request",
        )

        assert result["texts"] == ["What is the capital of France?"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "text, detection_type, rule",
        [
            ("My SSN is 123-45-6789, summarize my record", "prebuilt_pattern", "us_ssn"),
            ("notes on project  titan", "regex_pattern", r"project\s+titan"),
            ("This is INTERNAL ONLY", "keyword", "internal only"),
        ],
    )
    async def test_detection_reroutes_to_the_on_premise_model(self, text, detection_type, rule):
        guardrail = make_guardrail()

        with pytest.raises(SensitiveDataRouteException) as exc_info:
            await guardrail.apply_guardrail(
                inputs={"texts": [text]},
                request_data={"model": "cloud-model", "metadata": {"session_id": "abc-123"}},
                input_type="request",
            )

        exc = exc_info.value
        assert exc.route_to_model == "on-prem-model"
        assert exc.session_id == "abc-123"
        assert exc.guardrail_name == "sensitive-data-routing"
        assert exc.detection_info == {"detection_type": detection_type, "rule": rule}
        assert exc.sticky_session_routing is True
        assert exc.session_ttl_seconds == 14400

    @pytest.mark.asyncio
    async def test_detection_never_leaks_the_matched_value(self):
        guardrail = make_guardrail()

        with pytest.raises(SensitiveDataRouteException) as exc_info:
            await guardrail.apply_guardrail(
                inputs={"texts": ["My SSN is 123-45-6789"]},
                request_data={"metadata": {"session_id": "abc-123"}},
                input_type="request",
            )

        assert "123-45-6789" not in str(exc_info.value.detection_info)

    @pytest.mark.asyncio
    async def test_request_without_session_id_reroutes_without_pinning(self):
        """Docs: turns without a session id are still routed, but never pinned."""
        guardrail = make_guardrail()

        with pytest.raises(SensitiveDataRouteException) as exc_info:
            await guardrail.apply_guardrail(
                inputs={"texts": ["My SSN is 123-45-6789"]},
                request_data={"model": "cloud-model"},
                input_type="request",
            )

        assert exc_info.value.route_to_model == "on-prem-model"
        assert exc_info.value.sticky_session_routing is False

    @pytest.mark.asyncio
    async def test_sticky_session_disabled_does_not_pin(self):
        guardrail = make_guardrail(sticky_session=False)

        with pytest.raises(SensitiveDataRouteException) as exc_info:
            await guardrail.apply_guardrail(
                inputs={"texts": ["My SSN is 123-45-6789"]},
                request_data={"metadata": {"session_id": "abc-123"}},
                input_type="request",
            )

        assert exc_info.value.sticky_session_routing is False

    @pytest.mark.asyncio
    async def test_responses_are_not_scanned(self):
        """The guardrail only picks the model, so it has nothing to do on the response."""
        guardrail = make_guardrail()
        inputs = {"texts": ["My SSN is 123-45-6789"]}

        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"metadata": {"session_id": "abc-123"}},
            input_type="response",
        )

        assert result["texts"] == ["My SSN is 123-45-6789"]
