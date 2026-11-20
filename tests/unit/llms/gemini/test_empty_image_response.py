from collections.abc import Mapping
from io import BytesIO
from typing import Final

import httpx
import openai
import pytest

import litellm
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
@pytest.mark.parametrize("finish_reason", ("NO_IMAGE", "IMAGE_OTHER", "STOP"))
def test_answer_without_an_image_is_a_422_carrying_the_model_text_but_not_its_thoughts(
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
    assert f"finishReason={finish_reason}: I can only describe a red dot." in str(unanswered.value)
    assert "planning the dot" not in str(unanswered.value)


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
