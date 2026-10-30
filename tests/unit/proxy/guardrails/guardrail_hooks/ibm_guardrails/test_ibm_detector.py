import json

import pytest
import respx

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.ibm_guardrails.ibm_detector import IBMGuardrailDetector

pytestmark = pytest.mark.usefixtures("httpx_transport")

_BASE_URL = "https://ibm-detector.example.test"
_DETECTOR_SERVER_URL = f"{_BASE_URL}/api/v1/text/contents"
_ORCHESTRATOR_URL = f"{_BASE_URL}/api/v2/text/detection/content"
_NESTED_PARAMS: dict[str, object] = {
    "threshold": 0.5,
    "max_hits": 3,
    "strict": True,
    "labels": ["hap", None],
    "regex": {"email": ".+@.+"},
}
_DETECTOR_PARAMS_CASES = [
    pytest.param(None, {}, id="omitted"),
    pytest.param({}, {}, id="empty"),
    pytest.param(_NESTED_PARAMS, _NESTED_PARAMS, id="nested-mixed-types"),
]
_HAP_DETECTION = {
    "start": 0,
    "end": 5,
    "text": "hello",
    "detection": "HAP",
    "detection_type": "hap",
    "score": 0.97,
}


def _guardrail(*, is_detector_server: bool, detector_params: dict[str, object] | None) -> IBMGuardrailDetector:
    return IBMGuardrailDetector(
        guardrail_name="ibm-guard",
        auth_token="ibm-token",
        base_url=_BASE_URL,
        detector_id="hap-detector",
        is_detector_server=is_detector_server,
        detector_params=detector_params,
        event_hook="pre_call",
        default_on=True,
    )


def _request_data() -> dict[str, object]:
    return {"messages": [{"role": "user", "content": "hello there"}]}


def test_constructor_keeps_the_configured_detector_params_object() -> None:
    guardrail = _guardrail(is_detector_server=True, detector_params=_NESTED_PARAMS)

    assert guardrail.detector_params is _NESTED_PARAMS


@pytest.mark.asyncio
@pytest.mark.parametrize(("detector_params", "sent_params"), _DETECTOR_PARAMS_CASES)
async def test_detector_server_request_carries_configured_detector_params(
    respx_mock: respx.MockRouter, detector_params: dict[str, object] | None, sent_params: dict[str, object]
) -> None:
    route = respx_mock.post(_DETECTOR_SERVER_URL).respond(json=[[]])
    data = _request_data()

    returned = await _guardrail(is_detector_server=True, detector_params=detector_params).async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(), cache=DualCache(), data=data, call_type="completion"
    )

    sent = route.calls.last.request
    assert returned is data
    assert repr(json.loads(sent.content)) == repr({"contents": ["hello there"], "detector_params": sent_params})
    assert sent.headers["detector-id"] == "hap-detector"
    assert sent.headers["Authorization"] == "Bearer ibm-token"


@pytest.mark.asyncio
@pytest.mark.parametrize(("detector_params", "sent_params"), _DETECTOR_PARAMS_CASES)
async def test_orchestrator_request_keys_detector_params_by_detector_id(
    respx_mock: respx.MockRouter, detector_params: dict[str, object] | None, sent_params: dict[str, object]
) -> None:
    route = respx_mock.post(_ORCHESTRATOR_URL).respond(json={"detections": []})
    data = _request_data()

    returned = await _guardrail(is_detector_server=False, detector_params=detector_params).async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(), cache=DualCache(), data=data, call_type="completion"
    )

    sent = route.calls.last.request
    assert returned is data
    assert repr(json.loads(sent.content)) == repr(
        {"content": "hello there", "detectors": {"hap-detector": sent_params}}
    )
    assert "detector-id" not in sent.headers


@pytest.mark.asyncio
async def test_detector_server_detection_blocks_the_request_with_a_violation_summary(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.post(_DETECTOR_SERVER_URL).respond(json=[[_HAP_DETECTION]])

    with pytest.raises(ValueError, match=r"IBM Guardrail Detector failed: 1 violation\(s\) detected") as blocked:
        await _guardrail(is_detector_server=True, detector_params=_NESTED_PARAMS).async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(), cache=DualCache(), data=_request_data(), call_type="completion"
        )

    assert "  - HAP (score: 0.970)\n    Text: 'hello'" in str(blocked.value)


@pytest.mark.asyncio
async def test_orchestrator_detection_blocks_the_request_and_names_the_detector(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.post(_ORCHESTRATOR_URL).respond(json={"detections": [_HAP_DETECTION]})

    with pytest.raises(ValueError, match=r"IBM Guardrail Detector failed: 1 violation\(s\) detected") as blocked:
        await _guardrail(is_detector_server=False, detector_params=_NESTED_PARAMS).async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(), cache=DualCache(), data=_request_data(), call_type="completion"
        )

    assert "- HAP (detector: hap-detector, score: 0.970)\n  Text: 'hello'" in str(blocked.value)
