import asyncio
from collections.abc import Mapping
from datetime import datetime
from io import BytesIO
from typing import Final

import httpx
import openai
import pytest
import respx

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
    assert f"finishReason={finish_reason}" in str(refused.value)
    assert _NOT_CHARGED not in str(refused.value)


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
    assert "blockReason=PROHIBITED_CONTENT" in str(blocked.value)
    assert "prompt not allowed" not in str(blocked.value)


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


@pytest.mark.parametrize(
    "reply",
    (
        pytest.param({"candidates": [{"finishReason": "IMAGE_SAFETY", "content": {"role": "model"}}]}, id="refused"),
        pytest.param(
            {"candidates": [{"finishReason": "NO_IMAGE", "content": {"parts": [{"text": "no"}]}}]}, id="no-image"
        ),
    ),
)
def test_answer_without_an_image_keeps_its_deployment_in_service_and_names_no_other_group(
    reply: Mapping[str, object],
):
    calls: Final[list[httpx.Request]] = []  # mutable-ok: MockTransport hands each request to a callback

    def answer(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=reply)

    gemini: Final = {
        "model": "gemini/gemini-3.1-flash-image",
        "api_key": "test-key",
        "api_base": "https://google.invalid/v1",
    }
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "image-group",
                "litellm_params": gemini,
                "model_info": {
                    "id": "image-deployment",
                    "allowed_fails_policy": {
                        "ContentPolicyViolationErrorAllowedFails": 0,
                        "BadRequestErrorAllowedFails": 0,
                    },
                },
            },
            {"model_name": "unrelated-group", "litellm_params": gemini},
            {"model_name": "unrelated-backup", "litellm_params": gemini},
        ],
        fallbacks=[{"unrelated-group": ["unrelated-backup"]}],
        num_retries=0,
    )

    for _ in range(3):
        with pytest.raises(openai.APIStatusError) as failed:
            router.image_generation(
                model="image-group",
                prompt="a single small red dot",
                client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(answer))),
            )
        assert "unrelated" not in str(failed.value)

    assert len(calls) == 3


@pytest.mark.asyncio
async def test_refusal_after_a_failed_fallback_names_no_model_group(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": group,
                "litellm_params": {
                    "model": "gemini/gemini-3.1-flash-image",
                    "api_key": "test-key",
                    "api_base": f"https://{host}.invalid/v1",
                },
            }
            for group, host in (("image-group", "first"), ("backup-group", "second"))
        ],
        fallbacks=[{"image-group": ["backup-group"]}],
        num_retries=0,
    )

    with respx.mock(assert_all_called=True) as upstream:
        upstream.post(url__startswith="https://first.invalid/").mock(
            side_effect=lambda request: httpx.Response(
                200, json={"candidates": [{"finishReason": "IMAGE_SAFETY", "content": {"role": "model"}}]}
            )
        )
        upstream.post(url__startswith="https://second.invalid/").mock(
            side_effect=lambda request: httpx.Response(503, json={"error": {"code": 503}})
        )
        with pytest.raises(litellm.ContentPolicyViolationError) as refused:
            await router.aimage_generation(model="image-group", prompt="a single small red dot")

    assert "backup-group" not in str(refused.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "error"),
    (
        pytest.param(
            {"candidates": [{"finishReason": "IMAGE_SAFETY", "content": {"role": "model"}}]},
            litellm.ContentPolicyViolationError,
            id="refused",
        ),
        pytest.param(
            {"candidates": [{"finishReason": "NO_IMAGE", "content": {"parts": [{"text": "no"}]}}]},
            litellm.UnprocessableEntityError,
            id="no-image",
        ),
    ),
)
@pytest.mark.parametrize("fallbacks", (("backup-group",), ("backup-group", "outage-group")))
async def test_fallback_answer_without_an_image_after_a_provider_failure_is_the_error_the_client_gets(
    monkeypatch: pytest.MonkeyPatch,
    reply: Mapping[str, object],
    error: type[litellm.ContentPolicyViolationError | litellm.UnprocessableEntityError],
    fallbacks: tuple[str, ...],
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": group,
                "litellm_params": {
                    "model": "gemini/gemini-3.1-flash-image",
                    "api_key": "test-key",
                    "api_base": f"https://{host}.invalid/v1",
                },
            }
            for group, host in (("image-group", "first"), ("backup-group", "second"), ("outage-group", "third"))
        ],
        fallbacks=[{"image-group": list(fallbacks)}],
        num_retries=0,
    )

    with respx.mock(assert_all_called=False) as upstream:
        upstream.post(url__startswith="https://first.invalid/").mock(
            side_effect=lambda request: httpx.Response(503, json={"error": {"code": 503}})
        )
        upstream.post(url__startswith="https://second.invalid/").mock(
            side_effect=lambda request: httpx.Response(200, json=reply)
        )
        upstream.post(url__startswith="https://third.invalid/").mock(
            side_effect=lambda request: httpx.Response(503, json={"error": {"code": 503}})
        )
        with pytest.raises(error):
            await router.aimage_generation(model="image-group", prompt="a single small red dot")

        assert len(upstream.calls) == 1 + len(fallbacks)


_SERVED: Final = {
    "candidates": [{"finishReason": "STOP", "content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "aW1n"}}]}}],
    "usageMetadata": _NINE_IN_SEVEN_OUT,
}


class _ServedSpend(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.cost: float | None = None
        self.logged: Final = asyncio.Event()

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        response_cost: Final = kwargs.get("response_cost")
        self.cost = float(response_cost) if isinstance(response_cost, (int, float)) else None
        self.logged.set()


async def _cost_of_image_served_after(
    monkeypatch: pytest.MonkeyPatch, primary_answer: Mapping[str, object]
) -> float | None:
    spend: Final = _ServedSpend()
    monkeypatch.setattr(litellm, "callbacks", [spend])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": group,
                "litellm_params": {
                    "model": "gemini/gemini-3.1-flash-image",
                    "api_key": "test-key",
                    "api_base": f"https://{group}.invalid/v1",
                },
            }
            for group in ("primary", "backup")
        ],
        fallbacks=[{"primary": ["backup"]}],
        num_retries=0,
    )
    logging_obj, request = litellm.utils.function_setup(
        original_function="aimage_generation",
        rules_obj=litellm.utils.Rules(),
        start_time=datetime.now(),
        model="primary",
        prompt="a single small red dot",
        litellm_call_id="image-fallback",
    )
    with respx.mock(assert_all_called=False) as upstream:
        upstream.post(url__startswith="https://primary.invalid/").mock(
            return_value=httpx.Response(200, json=primary_answer)
        )
        upstream.post(url__startswith="https://backup.invalid/").mock(return_value=httpx.Response(200, json=_SERVED))
        await router.aimage_generation(**request, litellm_logging_obj=logging_obj)
        await asyncio.wait_for(spend.logged.wait(), 10)
    return spend.cost


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("finish_reason", "content", "error"),
    (
        pytest.param("STOP", {"parts": [{"text": "a red dot"}]}, litellm.UnprocessableEntityError, id="text-only"),
        pytest.param(
            "IMAGE_PROHIBITED_CONTENT", {"role": "model"}, litellm.ContentPolicyViolationError, id="charged-refusal"
        ),
        pytest.param("IMAGE_RECITATION", {"role": "model"}, litellm.ContentPolicyViolationError, id="free-refusal"),
    ),
)
async def test_fallback_that_serves_the_image_also_bills_the_answer_it_replaced(
    monkeypatch: pytest.MonkeyPatch,
    finish_reason: str,
    content: Mapping[str, object],
    error: type[litellm.ContentPolicyViolationError | litellm.UnprocessableEntityError],
):
    answer: Final = {
        "candidates": [{"finishReason": finish_reason, "finishMessage": _NOT_CHARGED, "content": content}],
        "usageMetadata": _NINE_IN_SEVEN_OUT,
    }
    replaced: Final = _failure_log(monkeypatch, "generation", "gemini/gemini-3.1-flash-image", answer, error).cost
    served_alone: Final = await _cost_of_image_served_after(monkeypatch, _SERVED)
    through_fallback: Final = await _cost_of_image_served_after(monkeypatch, answer)

    assert served_alone is not None and served_alone > 0
    assert replaced is not None
    assert through_fallback == pytest.approx(served_alone + replaced)
