from collections.abc import Mapping
from io import BytesIO
from typing import Final

import httpx
import openai
import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.utils import ImageResponse

_NOT_CHARGED: Final = (
    "Unable to show the generated image. The model could not generate the image based on the prompt provided. "
    "You will not be charged for this request. Try rephrasing the prompt."
)
_ROUTES: Final = (
    pytest.param("generation", "gemini/gemini-3.1-flash-image", "gemini", id="gemini-generation"),
    pytest.param("edit", "gemini/gemini-3.1-flash-image", "gemini", id="gemini-edit"),
    pytest.param("generation", "vertex_ai/gemini-3.1-flash-image", "vertex_ai", id="vertex-generation"),
    pytest.param("edit", "vertex_ai/gemini-3.1-flash-image", "vertex_ai", id="vertex-edit"),
)


class _FailureCost(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.cost: float | None = None
        self.usage: object = None

    def log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        response_cost: Final = kwargs.get("response_cost")
        self.cost = float(response_cost) if isinstance(response_cost, (int, float)) else None
        self.usage = kwargs.get("combined_usage_object")


def _request_image(route: str, model: str, reply: Mapping[str, object]) -> ImageResponse:
    client: Final = HTTPHandler(
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=reply)))
    )
    if route == "edit":
        return litellm.image_edit(
            model=model,
            image=BytesIO(b"\x89PNG\r\n\x1a\nimage"),
            prompt="a single small red dot",
            api_key="test-key",
            api_base="https://google.invalid/v1",
            client=client,
        )
    return litellm.image_generation(
        model=model,
        prompt="a single small red dot",
        api_key="test-key",
        api_base="https://google.invalid/v1",
        client=client,
    )


@pytest.mark.parametrize(("route", "model", "provider"), _ROUTES)
@pytest.mark.parametrize("finish_reason", ("IMAGE_RECITATION", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "SAFETY"))
def test_refused_image_is_a_400_carrying_the_finish_reason_and_message(
    route: str, model: str, provider: str, finish_reason: str
):
    with pytest.raises(litellm.ContentPolicyViolationError) as refused:
        _request_image(
            route,
            model,
            {
                "candidates": [{"finishReason": finish_reason, "finishMessage": _NOT_CHARGED, "content": {"role": "model"}}],
                "usageMetadata": {"promptTokenCount": 9, "totalTokenCount": 9},
            },
        )

    assert (refused.value.status_code, refused.value.llm_provider, refused.value.provider_specific_fields) == (
        400,
        provider,
        {"finish_reason": finish_reason, "finish_message": _NOT_CHARGED},
    )
    assert f"finishReason={finish_reason}: {_NOT_CHARGED}" in str(refused.value)


@pytest.mark.parametrize(("route", "model", "provider"), _ROUTES)
def test_blocked_prompt_is_a_400_carrying_the_block_reason(route: str, model: str, provider: str):
    with pytest.raises(litellm.ContentPolicyViolationError) as blocked:
        _request_image(
            route,
            model,
            {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT", "blockReasonMessage": "prompt not allowed"}},
        )

    assert (blocked.value.status_code, blocked.value.llm_provider, blocked.value.provider_specific_fields) == (
        400,
        provider,
        {"block_reason": "PROHIBITED_CONTENT", "block_reason_message": "prompt not allowed"},
    )


@pytest.mark.parametrize(("route", "model", "provider"), _ROUTES)
@pytest.mark.parametrize("finish_reason", ("NO_IMAGE", "IMAGE_OTHER", "OTHER", "LANGUAGE", "STOP"))
def test_answer_without_an_image_is_a_422_that_does_not_echo_the_model_output(
    route: str, model: str, provider: str, finish_reason: str
):
    with pytest.raises(litellm.UnprocessableEntityError) as unanswered:
        _request_image(
            route,
            model,
            {
                "candidates": [
                    {
                        "finishReason": finish_reason,
                        "content": {
                            "parts": [
                                {"text": "planning the dot", "thought": True},
                                {"text": "I can only describe a red dot."},
                            ]
                        },
                    }
                ]
            },
        )

    assert (unanswered.value.status_code, unanswered.value.llm_provider) == (422, provider)
    assert f"finishReason={finish_reason}" in str(unanswered.value)
    assert "red dot" not in str(unanswered.value)
    assert "planning the dot" not in str(unanswered.value)
    assert "dot" not in unanswered.value.response.text


def _failure_log(
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    model: str,
    reply: Mapping[str, object],
    error: type[litellm.ContentPolicyViolationError | litellm.UnprocessableEntityError],
) -> _FailureCost:
    failure_log: Final = _FailureCost()
    monkeypatch.setattr(litellm, "failure_callback", [failure_log])
    with pytest.raises(error):
        _request_image(route, model, reply)
    return failure_log


_NINE_IN_SEVEN_OUT: Final = {"promptTokenCount": 9, "candidatesTokenCount": 7, "totalTokenCount": 16}


@pytest.mark.parametrize(("route", "model", "provider"), _ROUTES)
@pytest.mark.parametrize(
    ("reply", "error"),
    (
        pytest.param(
            {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "a red dot"}]}}]},
            litellm.UnprocessableEntityError,
            id="text-only",
        ),
        pytest.param(
            {"candidates": [{"finishReason": "IMAGE_PROHIBITED_CONTENT", "content": {"role": "model"}}]},
            litellm.ContentPolicyViolationError,
            id="IMAGE_PROHIBITED_CONTENT",
        ),
        pytest.param(
            {"candidates": [{"finishReason": "IMAGE_SAFETY", "content": {"role": "model"}}]},
            litellm.ContentPolicyViolationError,
            id="IMAGE_SAFETY",
        ),
        pytest.param(
            {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}},
            litellm.ContentPolicyViolationError,
            id="blockReason",
        ),
    ),
)
def test_answer_without_an_image_bills_its_tokens(
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    model: str,
    provider: str,
    reply: Mapping[str, object],
    error: type[litellm.ContentPolicyViolationError | litellm.UnprocessableEntityError],
):
    cost: Final = _failure_log(monkeypatch, route, model, {**reply, "usageMetadata": _NINE_IN_SEVEN_OUT}, error).cost

    prices: Final = litellm.get_model_info(model)
    text_price: Final = sum(litellm.cost_per_token(model=model, prompt_tokens=9, completion_tokens=7))
    image_price: Final = text_price + 7 * (prices["output_cost_per_image_token"] - prices["output_cost_per_token"])

    assert cost is not None and text_price <= cost < image_price


@pytest.mark.parametrize(("route", "model", "provider"), _ROUTES)
def test_recitation_refusal_google_does_not_charge_bills_nothing(
    monkeypatch: pytest.MonkeyPatch, route: str, model: str, provider: str
):
    refused: Final = {
        "candidates": [{"finishReason": "IMAGE_RECITATION", "finishMessage": _NOT_CHARGED, "content": {"role": "model"}}],
        "usageMetadata": _NINE_IN_SEVEN_OUT,
    }

    assert _failure_log(monkeypatch, route, model, refused, litellm.ContentPolicyViolationError).cost == 0


@pytest.mark.parametrize(("route", "model", "provider"), _ROUTES)
def test_billed_answer_keeps_its_reasoning_tokens_on_the_failure_log(
    monkeypatch: pytest.MonkeyPatch, route: str, model: str, provider: str
):
    unanswered: Final = {
        "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "a red dot"}]}}],
        "usageMetadata": {
            "promptTokenCount": 9,
            "candidatesTokenCount": 7,
            "thoughtsTokenCount": 30,
            "totalTokenCount": 46,
        },
    }

    usage: Final = _failure_log(monkeypatch, route, model, unanswered, litellm.UnprocessableEntityError).usage

    assert isinstance(usage, litellm.Usage)
    assert usage.completion_tokens_details is not None and usage.completion_tokens_details.reasoning_tokens == 30


@pytest.mark.parametrize(("route", "model", "provider"), _ROUTES)
def test_answer_without_candidates_is_a_422(route: str, model: str, provider: str):
    with pytest.raises(litellm.UnprocessableEntityError) as unanswered:
        _request_image(route, model, {"candidates": [], "usageMetadata": {"promptTokenCount": 9}})

    assert (unanswered.value.status_code, unanswered.value.llm_provider) == (422, provider)
    assert "finishReason=none" in str(unanswered.value)


@pytest.mark.parametrize(("route", "model", "provider"), _ROUTES)
def test_image_next_to_a_refused_candidate_is_still_returned(route: str, model: str, provider: str):
    response: Final = _request_image(
        route,
        model,
        {
            "candidates": [
                {"finishReason": "IMAGE_SAFETY", "content": {"role": "model"}},
                {
                    "finishReason": "STOP",
                    "content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "aW1n"}}]},
                },
            ]
        },
    )

    assert [image.b64_json for image in response.data or []] == ["aW1n"]


def _upstream_calls_through_a_retrying_router(reply: httpx.Response) -> int:
    calls: Final[list[httpx.Request]] = []  # mutable-ok: MockTransport hands each request to a callback
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "image-group",
                "litellm_params": {
                    "model": "gemini/gemini-3.1-flash-image",
                    "api_key": "test-key",
                    "api_base": "https://google.invalid/v1",
                },
            }
        ],
        num_retries=2,
    )

    def answer(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return reply

    with pytest.raises(openai.APIStatusError):
        router.image_generation(
            model="image-group",
            prompt="a single small red dot",
            client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(answer))),
        )
    return len(calls)


@pytest.mark.parametrize(
    "reply",
    (
        pytest.param({"candidates": [{"finishReason": "IMAGE_RECITATION", "content": {"role": "model"}}]}, id="refused"),
        pytest.param(
            {"candidates": [{"finishReason": "NO_IMAGE", "content": {"parts": [{"text": "no"}]}}]}, id="no-image"
        ),
    ),
)
def test_router_does_not_retry_an_answer_without_an_image(reply: Mapping[str, object]):
    assert _upstream_calls_through_a_retrying_router(httpx.Response(200, json=reply)) == 1
    assert _upstream_calls_through_a_retrying_router(httpx.Response(503, json={"error": {"code": 503}})) == 3
