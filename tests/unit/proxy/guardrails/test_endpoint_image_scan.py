import json
from collections.abc import Callable, Mapping
from typing import Final, Literal

import httpx
import pytest
from fastapi import HTTPException

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy.guardrails.endpoint_image_scan import scan_images
from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrail
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.llms.openai import AllMessageValues
from litellm.types.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrailResponse

_PNG_BASE64: Final = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
_PNG: Final = f"data:image/png;base64,{_PNG_BASE64}"
_RECORDS_KEY: Final = "standard_logging_guardrail_information"
_EVALUATED: Final = {
    "action": "NONE",
    "outputs": [],
    "assessments": [],
    "guardrailCoverage": {"images": {"guarded": 1, "total": 1}},
}
_NOT_EVALUATED: Final = {
    "action": "NONE",
    "outputs": [],
    "assessments": [],
    "guardrailCoverage": {"images": {"guarded": 0, "total": 1}},
}
_BLOCKED: Final = {
    "action": "GUARDRAIL_INTERVENED",
    "outputs": [{"text": "blocked"}],
    "assessments": [{"contentPolicy": {"filters": [{"type": "VIOLENCE", "confidence": "HIGH", "action": "BLOCKED"}]}}],
    "guardrailCoverage": {"images": {"guarded": 1, "total": 1}},
}


class _Bedrock:
    def __init__(self, answer: Mapping[str, object], status_code: int = 200) -> None:
        self.answer: Final = answer
        self.status_code: Final = status_code
        self.requests: Final[list[httpx.Request]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status_code, json=self.answer)


def _apply_guardrail_api(bedrock: _Bedrock) -> BedrockGuardrail:
    return BedrockGuardrail(
        guardrail_name="image-guard",
        guardrailIdentifier="gr-1",
        guardrailVersion="1",
        aws_region_name="us-east-1",
        aws_access_key_id="test-access-key",
        aws_secret_access_key="test-secret-key",
        async_handler=AsyncHTTPHandler(transport=httpx.MockTransport(bedrock)),
    )


def _guardrail_checks_api(bedrock: _Bedrock) -> BedrockGuardrail:
    return BedrockGuardrail(
        guardrail_name="image-guard",
        checks={"contentFilter": {"categories": [{"category": "VIOLENCE"}]}},
        aws_region_name="us-east-1",
        aws_access_key_id="test-access-key",
        aws_secret_access_key="test-secret-key",
        async_handler=AsyncHTTPHandler(transport=httpx.MockTransport(bedrock)),
    )


def _another_provider(bedrock: _Bedrock) -> CustomGuardrail:
    return CustomGuardrail(guardrail_name="image-guard")


class _OwnBedrockCall(BedrockGuardrail):
    async def make_bedrock_api_request(
        self,
        source: Literal["INPUT", "OUTPUT"],
        messages: list[AllMessageValues] | None = None,
        response: litellm.ModelResponse | None = None,
        request_data: dict | None = None,
        logging_event_type: GuardrailEventHooks | None = None,
    ) -> BedrockGuardrailResponse:
        return {"action": "NONE"}


def _subclass_with_its_own_bedrock_call(bedrock: _Bedrock) -> CustomGuardrail:
    return _OwnBedrockCall(
        guardrail_name="image-guard",
        guardrailIdentifier="gr-1",
        guardrailVersion="1",
        aws_region_name="us-east-1",
        aws_access_key_id="test-access-key",
        aws_secret_access_key="test-secret-key",
        async_handler=AsyncHTTPHandler(transport=httpx.MockTransport(bedrock)),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("input_type", "source", "mode"),
    [("request", "INPUT", "pre_call"), ("response", "OUTPUT", "post_call")],
)
async def test_image_is_scanned_in_the_requested_direction_and_its_verdict_recorded(
    input_type: Literal["request", "response"], source: str, mode: str
) -> None:
    bedrock: Final = _Bedrock(_EVALUATED)
    request_data: Final[dict[str, object]] = {}

    await scan_images(_apply_guardrail_api(bedrock), (_PNG,), request_data, input_type)

    assert [json.loads(request.content) for request in bedrock.requests] == [
        {"source": source, "content": [{"image": {"format": "png", "source": {"bytes": _PNG_BASE64}}}]}
    ]
    assert [
        (record["guardrail_status"], record["guardrail_mode"]) for record in request_data["metadata"][_RECORDS_KEY]
    ] == [("success", mode)]


@pytest.mark.asyncio
async def test_blocked_image_raises_the_block_and_records_the_intervention() -> None:
    request_data: Final[dict[str, object]] = {}

    with pytest.raises(HTTPException) as blocked:
        await scan_images(_apply_guardrail_api(_Bedrock(_BLOCKED)), (_PNG,), request_data, "request")

    assert blocked.value.status_code == 400
    assert [record["guardrail_status"] for record in request_data["metadata"][_RECORDS_KEY]] == ["guardrail_intervened"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "image",
    [
        "https://example.com/cat.png",
        _PNG_BASE64,
        f"data:image/webp;base64,{_PNG_BASE64}",
        "data:image/png;base64,bm90IGFuIGltYWdl",
        "data:image/png;base64,@@@@",
    ],
    ids=["remote-url", "bare-base64", "webp", "not-image-bytes", "invalid-base64"],
)
async def test_unreadable_image_is_refused_without_a_scan_or_a_recorded_block(image: str) -> None:
    bedrock: Final = _Bedrock(_EVALUATED)
    request_data: Final[dict[str, object]] = {}

    with pytest.raises(HTTPException) as refused:
        await scan_images(_apply_guardrail_api(bedrock), (image,), request_data, "request")

    assert refused.value.status_code == 422
    assert "base64 PNG or JPEG data URL" in refused.value.detail
    assert bedrock.requests == []
    assert request_data == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("guardrail", [_guardrail_checks_api, _another_provider, _subclass_with_its_own_bedrock_call])
async def test_guardrail_that_cannot_scan_images_refuses_them(
    guardrail: Callable[[_Bedrock], CustomGuardrail],
) -> None:
    bedrock: Final = _Bedrock(_EVALUATED)

    with pytest.raises(HTTPException) as refused:
        await scan_images(guardrail(bedrock), (_PNG,), {}, "request")

    assert refused.value.status_code == 422
    assert bedrock.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "status_code", "reason", "recorded_status"),
    [
        (_NOT_EVALUATED, 422, "Image modality", "not_run"),
        ({"action": "NONE", "outputs": [], "assessments": []}, 422, "Image modality", "not_run"),
        ({"Output": {"__type": "ServiceException"}}, 502, "ServiceException", "guardrail_failed_to_respond"),
    ],
    ids=["reported-unguarded", "no-coverage-reported", "error-inside-a-200-body"],
)
async def test_image_bedrock_did_not_confirm_evaluating_is_not_reported_as_passed(
    answer: Mapping[str, object], status_code: int, reason: str, recorded_status: str
) -> None:
    request_data: Final[dict[str, object]] = {}

    with pytest.raises(HTTPException) as refused:
        await scan_images(_apply_guardrail_api(_Bedrock(answer)), (_PNG,), request_data, "request")

    assert refused.value.status_code == status_code
    assert reason in refused.value.detail
    assert [record["guardrail_status"] for record in request_data["metadata"][_RECORDS_KEY]] == [recorded_status]


@pytest.mark.asyncio
@pytest.mark.parametrize(("bedrock_status", "status_code"), [(400, 422), (503, 503)])
async def test_image_bedrock_rejects_answers_unscanned_while_its_outages_pass_through(
    bedrock_status: int, status_code: int
) -> None:
    bedrock: Final = _Bedrock({"message": "The images contained in the request are not supported"}, bedrock_status)

    with pytest.raises(HTTPException) as refused:
        await scan_images(_apply_guardrail_api(bedrock), (_PNG,), {}, "request")

    assert refused.value.status_code == status_code
    assert "The images contained in the request are not supported" in refused.value.detail
