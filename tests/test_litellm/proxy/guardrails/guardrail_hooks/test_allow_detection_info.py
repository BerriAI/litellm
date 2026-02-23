"""Tests for allow(detection_info) and detect() - detector-style guardrails."""

import pytest

from litellm.proxy.guardrails.guardrail_hooks.custom_code import CustomCodeGuardrail

# Custom code that always allows and attaches detection_info (detector pattern)
ALLOW_DETECTION_INFO_CODE = '''
def apply_guardrail(inputs, request_data, input_type):
    return allow(detection_info={"is_english": True, "language": "en"})
'''

DETECT_FALSE_CODE = '''
def apply_guardrail(inputs, request_data, input_type):
    return detect(False, language="es")
'''


@pytest.fixture
def detector_guardrail():
    """Guardrail that returns allow with detection_info."""
    return CustomCodeGuardrail(
        guardrail_name="test_detector",
        custom_code=ALLOW_DETECTION_INFO_CODE,
    )


@pytest.mark.asyncio
async def test_allow_detection_info_returns_inputs_unchanged(detector_guardrail):
    """allow(detection_info=...) should return inputs unchanged."""
    inputs = {"texts": ["hello"]}
    request_data = {"metadata": {}}
    result = await detector_guardrail.apply_guardrail(
        inputs=inputs,
        request_data=request_data,
        input_type="response",
    )
    assert result == inputs


@pytest.mark.asyncio
async def test_allow_detection_info_appends_to_metadata_detections(detector_guardrail):
    """allow(detection_info=...) should append to request_data['metadata']['detections']."""
    request_data = {"metadata": {}}
    await detector_guardrail.apply_guardrail(
        inputs={"texts": ["hello"]},
        request_data=request_data,
        input_type="response",
    )
    detections = request_data.get("metadata", {}).get("detections", [])
    assert len(detections) == 1
    assert detections[0]["guardrail_name"] == "test_detector"
    assert detections[0]["detection_info"] == {"is_english": True, "language": "en"}


@pytest.mark.asyncio
async def test_detect_primitve_flows_through():
    """detect(False, ...) returns allow with detection_info (same as allow(detection_info=...))."""
    guardrail = CustomCodeGuardrail(
        guardrail_name="test_detect",
        custom_code=DETECT_FALSE_CODE,
    )
    request_data = {"metadata": {}}
    result = await guardrail.apply_guardrail(
        inputs={"texts": ["foo"]},
        request_data=request_data,
        input_type="response",
    )
    assert result == {"texts": ["foo"]}
    detections = request_data.get("metadata", {}).get("detections", [])
    assert len(detections) == 1
    assert detections[0]["detection_info"]["detected"] is False
    assert detections[0]["detection_info"]["language"] == "es"
