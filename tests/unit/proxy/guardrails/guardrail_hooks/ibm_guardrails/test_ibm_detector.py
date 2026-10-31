import json

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Final
import pytest
import respx

import litellm

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral
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


_INPUT_HOOKS = [
    pytest.param(
        "pre_call",
        lambda guardrail, data, call_type: guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(), cache=DualCache(), data=data, call_type=call_type
        ),
        id="pre_call",
    ),
    pytest.param(
        "during_call",
        lambda guardrail, data, call_type: guardrail.async_moderation_hook(
            data=data, user_api_key_dict=UserAPIKeyAuth(), call_type=call_type
        ),
        id="during_call",
    ),
]
_REQUEST_TEXT_CASES = [
    pytest.param({"messages": [{"role": "user", "content": "hello there"}]}, "completion", ["ibm-guard"], id="messages"),
    pytest.param({"state": "hello there"}, "aextraction", None, id="no-messages-on-extraction"),
    pytest.param({"state": "hello there"}, "completion", ["ibm-guard"], id="no-messages-on-chat"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("event_hook", "run_hook"), _INPUT_HOOKS)
@pytest.mark.parametrize(("request_fields", "call_type", "applied"), _REQUEST_TEXT_CASES)
async def test_input_hooks_list_the_guardrail_only_when_they_scanned_text(
    respx_mock: respx.MockRouter,
    event_hook: str,
    run_hook: Callable[[IBMGuardrailDetector, Mapping[str, object], CallTypesLiteral], Awaitable[object]],
    request_fields: Mapping[str, object],
    call_type: CallTypesLiteral,
    applied: Sequence[str] | None,
) -> None:
    respx_mock.post(_DETECTOR_SERVER_URL).respond(json=[[]])
    guardrail: Final = IBMGuardrailDetector(
        guardrail_name="ibm-guard",
        auth_token="ibm-token",
        base_url=_BASE_URL,
        detector_id="hap-detector",
        is_detector_server=True,
        event_hook=event_hook,
        default_on=True,
    )
    metadata: Final[dict[str, object]] = {}

    await run_hook(guardrail, {**request_fields, "metadata": metadata}, call_type)

    assert metadata.get("applied_guardrails") == applied


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "request_route", "applied"),
    [
        pytest.param(
            litellm.ModelResponse(
                choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}]
            ),
            "/v1/chat/completions",
            ["ibm-guard"],
            id="chat-reply",
        ),
        pytest.param(
            {"answers": {"aws_key": {"type": "text", "value": None}}},
            "/v1/extraction",
            None,
            id="no-choices-on-extraction",
        ),
        pytest.param(
            {"answers": {"aws_key": {"type": "text", "value": None}}},
            "/v1/responses",
            ["ibm-guard"],
            id="no-choices-on-responses",
        ),
    ],
)
async def test_post_call_hook_lists_the_guardrail_only_when_it_scanned_text(
    respx_mock: respx.MockRouter,
    response: litellm.ModelResponse | Mapping[str, object],
    request_route: str,
    applied: Sequence[str] | None,
) -> None:
    respx_mock.post(_DETECTOR_SERVER_URL).respond(json=[[]])
    guardrail: Final = IBMGuardrailDetector(
        guardrail_name="ibm-guard",
        auth_token="ibm-token",
        base_url=_BASE_URL,
        detector_id="hap-detector",
        is_detector_server=True,
        event_hook="post_call",
        default_on=True,
    )
    metadata: Final[dict[str, object]] = {}

    await guardrail.async_post_call_success_hook(
        data={"messages": [{"role": "user", "content": "hello"}], "metadata": metadata},
        user_api_key_dict=UserAPIKeyAuth(request_route=request_route),
        response=response,
    )

    assert metadata.get("applied_guardrails") == applied
