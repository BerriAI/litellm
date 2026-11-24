import base64
import json
from collections.abc import Callable
from typing import Final

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import CostCalculatorUtils
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.dashscope.image_edit.transformation import DashScopeImageEditConfig
from litellm.llms.dashscope.text_to_speech.transformation import DashScopeTextToSpeechConfig
from litellm.types.utils import ModelInfo

PNG: Final = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_DATA_URI: Final = f"data:image/png;base64,{base64.b64encode(PNG).decode()}"
INTL_ENDPOINT: Final = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
EDITED_URL: Final = "https://dashscope-result.example.com/edited.png"
EDITED: Final = {
    "output": {"choices": [{"message": {"role": "assistant", "content": [{"image": EDITED_URL}]}}]},
    "usage": {"width": 1024, "height": 1024, "image_count": 1},
}


def _recording(
    seen: list[httpx.Request], reply: Callable[[httpx.Request], httpx.Response]
) -> httpx.MockTransport:  # mutable-ok: test records the requests the transport receives
    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return reply(request)

    return httpx.MockTransport(respond)


def _edited(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=EDITED)


def _sync_client(
    seen: list[httpx.Request], reply: Callable[[httpx.Request], httpx.Response] = _edited
) -> HTTPHandler:  # mutable-ok: shared request log
    return HTTPHandler(client=httpx.Client(transport=_recording(seen, reply)))


@pytest.fixture(autouse=True)
def _no_ambient_dashscope_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_BASE_IMAGE", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_BASE", raising=False)


@pytest.mark.parametrize("asynchronous", (False, True))
async def test_an_edit_posts_every_image_inline_with_provider_params_at_the_top_of_parameters(
    asynchronous: bool,
) -> None:
    seen: Final[list[httpx.Request]] = []  # mutable-ok: request log
    arguments: Final = {
        "model": "dashscope/qwen-image-2.0",
        "image": [PNG, PNG],
        "prompt": "Turn the circle into a square",
        "size": "1024x1024",
        "n": 1,
        "negative_prompt": "blurry",
        "prompt_extend": False,
        "watermark": False,
        "seed": 42,
        "api_key": "sk-dashscope-test",
    }
    response: Final = (
        await litellm.aimage_edit(**arguments, client=AsyncHTTPHandler(transport=_recording(seen, _edited)))
        if asynchronous
        else litellm.image_edit(**arguments, client=_sync_client(seen))
    )

    assert [image.url for image in response.data or []] == [EDITED_URL]
    (request,) = seen
    assert str(request.url) == INTL_ENDPOINT
    assert request.headers["authorization"] == "Bearer sk-dashscope-test"
    assert json.loads(request.content) == {
        "model": "qwen-image-2.0",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": PNG_DATA_URI},
                        {"image": PNG_DATA_URI},
                        {"text": "Turn the circle into a square"},
                    ],
                }
            ]
        },
        "parameters": {
            "n": 1,
            "size": "1024*1024",
            "negative_prompt": "blurry",
            "prompt_extend": False,
            "watermark": False,
            "seed": 42,
        },
    }


@pytest.mark.parametrize(("size", "dashscope_size"), (("1792x1024", "1792*1024"), ("1328x1328", "1328*1328")))
def test_an_openai_size_reaches_dashscope_in_its_star_format(size: str, dashscope_size: str) -> None:
    seen: Final[list[httpx.Request]] = []  # mutable-ok: request log

    litellm.image_edit(
        model="dashscope/qwen-image-2.0", image=PNG, prompt="p", size=size, api_key="k", client=_sync_client(seen)
    )

    assert json.loads(seen[0].content)["parameters"] == {"size": dashscope_size}


CHINA_ENDPOINT: Final = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


@pytest.mark.parametrize(
    ("api_base", "env_base", "image_base", "endpoint"),
    (
        ("https://dashscope.aliyuncs.com/compatible-mode/v1", None, None, CHINA_ENDPOINT),
        ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1/", None, None, INTL_ENDPOINT),
        ("https://dashscope.aliyuncs.com", None, None, CHINA_ENDPOINT),
        ("https://dashscope-intl.aliyuncs.com/apps/anthropic", None, None, INTL_ENDPOINT),
        (None, "https://dashscope.aliyuncs.com/apps/anthropic", None, CHINA_ENDPOINT),
        (
            "https://gw.example.com/compatible-mode/v1",
            None,
            None,
            "https://gw.example.com/api/v1/services/aigc/multimodal-generation/generation",
        ),
        (None, "https://dashscope.aliyuncs.com/compatible-mode/v1", None, CHINA_ENDPOINT),
        (None, "https://dashscope.aliyuncs.com/api/v1", None, CHINA_ENDPOINT),
        (None, None, None, INTL_ENDPOINT),
        (
            "https://images.example.com/edit",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            None,
            "https://images.example.com/edit",
        ),
        ("https://dashscope.aliyuncs.com/compatible-mode/v1", None, "https://images.example.com/edit", CHINA_ENDPOINT),
        (
            None,
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "https://images.example.com/edit",
            "https://images.example.com/edit",
        ),
    ),
)
def test_an_edit_goes_to_the_image_endpoint_on_the_host_its_dashscope_base_names(
    monkeypatch: pytest.MonkeyPatch, api_base: str | None, env_base: str | None, image_base: str | None, endpoint: str
) -> None:
    if env_base is not None:
        monkeypatch.setenv("DASHSCOPE_API_BASE", env_base)
    if image_base is not None:
        monkeypatch.setenv("DASHSCOPE_API_BASE_IMAGE", image_base)
    seen: Final[list[httpx.Request]] = []  # mutable-ok: request log

    litellm.image_edit(
        model="dashscope/qwen-image-2.0",
        image=PNG,
        prompt="p",
        api_key="k",
        api_base=api_base,
        client=_sync_client(seen),
    )

    assert str(seen[0].url) == endpoint


@pytest.mark.parametrize("api_base", ("https://dashscope.aliyuncs.com/compatible-mode/v1", None))
def test_an_edit_and_a_speech_call_with_the_same_dashscope_base_reach_the_same_host(
    monkeypatch: pytest.MonkeyPatch, api_base: str | None
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_BASE", "https://gw.example.com/compatible-mode/v1")
    edit_url: Final = DashScopeImageEditConfig().get_complete_url(
        model="qwen-image-2.0", api_base=api_base, litellm_params={}
    )
    speech_url: Final = DashScopeTextToSpeechConfig().get_complete_url(
        model="qwen3-tts-flash", api_base=api_base, litellm_params={}
    )

    assert httpx.URL(edit_url).host == httpx.URL(speech_url).host


def test_a_fourth_image_is_refused_with_a_400_before_any_request_even_under_drop_params() -> None:
    seen: Final[list[httpx.Request]] = []  # mutable-ok: request log

    with pytest.raises(litellm.BadRequestError) as refused:
        litellm.image_edit(
            model="dashscope/qwen-image-2.0",
            image=[PNG, PNG, PNG, PNG],
            prompt="p",
            api_key="k",
            drop_params=True,
            client=_sync_client(seen),
        )

    assert refused.value.status_code == 400
    assert "at most 3 input images, got 4" in str(refused.value)
    assert seen == []


def test_three_images_are_accepted() -> None:
    seen: Final[list[httpx.Request]] = []  # mutable-ok: request log

    litellm.image_edit(
        model="dashscope/qwen-image-2.0", image=[PNG, PNG, PNG], prompt="p", api_key="k", client=_sync_client(seen)
    )

    content: Final = json.loads(seen[0].content)["input"]["messages"][0]["content"]
    assert content == [{"image": PNG_DATA_URI}] * 3 + [{"text": "p"}]


def test_an_unsupported_openai_param_answers_400_and_is_dropped_under_drop_params() -> None:
    seen: Final[list[httpx.Request]] = []  # mutable-ok: request log
    client: Final = _sync_client(seen)

    with pytest.raises(litellm.UnsupportedParamsError) as refused:
        litellm.image_edit(
            model="dashscope/qwen-image-2.0", image=PNG, prompt="p", quality="high", api_key="k", client=client
        )
    litellm.image_edit(
        model="dashscope/qwen-image-2.0",
        image=PNG,
        prompt="p",
        quality="high",
        api_key="k",
        drop_params=True,
        client=client,
    )

    assert refused.value.status_code == 400
    assert "quality" in str(refused.value)
    (request,) = seen
    assert json.loads(request.content)["parameters"] == {}


@pytest.mark.parametrize(
    ("reply", "error", "message"),
    (
        (
            httpx.Response(200, json={"code": "InvalidParameter", "message": "Image is too small"}),
            litellm.APIError,
            "Image is too small",
        ),
        (
            httpx.Response(400, json={"code": "InvalidParameter", "message": "Model not exist."}),
            litellm.BadRequestError,
            "Model not exist.",
        ),
    ),
)
def test_a_dashscope_rejection_reaches_the_caller_with_its_message(
    reply: httpx.Response, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        litellm.image_edit(
            model="dashscope/qwen-image-2.0",
            image=PNG,
            prompt="p",
            api_key="k",
            client=_sync_client([], lambda _: reply),
        )


@pytest.mark.parametrize("model", ("qwen-image-2.0", "qwen-image-2.0-pro", "qwen-image-3.0"))
@pytest.mark.usefixtures("local_model_cost_map")
def test_each_edited_image_is_billed_at_the_model_output_image_price(model: str) -> None:
    response: Final = litellm.ImageResponse(data=[{"url": EDITED_URL}, {"url": EDITED_URL}])

    cost: Final = litellm.completion_cost(
        completion_response=response,
        model=f"dashscope/{model}",
        call_type="aimage_edit",
        custom_llm_provider="dashscope",
    )

    assert cost == pytest.approx(2 * litellm.model_cost[f"dashscope/{model}"]["output_cost_per_image"])
    assert cost > 0


@pytest.mark.usefixtures("local_model_cost_map")
def test_the_input_images_dashscope_reports_are_billed_on_top_of_the_edited_images() -> None:
    def reply(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={**EDITED, "usage": {"input_image_count": 2, "output_image_count": 1}})

    response: Final = litellm.image_edit(
        model="dashscope/qwen-image-3.0", image=[PNG, PNG], prompt="p", api_key="k", client=_sync_client([], reply)
    )

    assert isinstance(response, litellm.ImageResponse)
    assert CostCalculatorUtils.route_image_generation_cost_calculator(
        model="qwen-image-3.0",
        completion_response=response,
        custom_llm_provider="dashscope",
        call_type="aimage_edit",
        model_info=ModelInfo(input_cost_per_image=0.004, output_cost_per_image=0.04),
    ) == pytest.approx(2 * 0.004 + 0.04)


def _edited_images(count: int, input_images: int) -> litellm.ImageResponse:
    response: Final = litellm.ImageResponse(data=[{"url": EDITED_URL}] * count)
    response.hidden_params = {"billable_input_image_count": input_images}
    return response


@pytest.mark.parametrize(
    ("model", "deployment_price", "size", "expected_cost"),
    (
        ("qwen-image-2.0", ModelInfo(input_cost_per_image=0.02), None, 2 * 0.02),
        ("qwen-image-3.0-pro", ModelInfo(input_cost_per_pixel=1e-08), "1024x1024", 2 * 1024 * 1024 * 1e-08),
    ),
)
@pytest.mark.usefixtures("local_model_cost_map")
def test_a_deployment_image_price_bills_the_edit_the_way_it_bills_any_image_call(
    model: str, deployment_price: ModelInfo, size: str | None, expected_cost: float
) -> None:
    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model=model,
        completion_response=_edited_images(count=2, input_images=3),
        custom_llm_provider="dashscope",
        size=size,
        call_type="aimage_edit",
        model_info=deployment_price,
    )

    assert cost == pytest.approx(expected_cost)


@pytest.mark.usefixtures("local_model_cost_map")
def test_a_deployment_that_copies_both_image_prices_bills_edited_and_input_images_at_its_own_prices() -> None:
    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="qwen-image-3.0",
        completion_response=_edited_images(count=2, input_images=3),
        custom_llm_provider="dashscope",
        call_type="aimage_edit",
        model_info=ModelInfo(input_cost_per_image=0.004, output_cost_per_image=0.04),
    )

    assert cost == pytest.approx(2 * 0.04 + 3 * 0.004)


@pytest.mark.usefixtures("local_model_cost_map")
def test_a_deployment_token_price_keeps_the_shared_per_image_billing() -> None:
    prices: Final = litellm.model_cost["dashscope/qwen-image-3.0"]

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="qwen-image-3.0",
        completion_response=_edited_images(count=1, input_images=2),
        custom_llm_provider="dashscope",
        call_type="aimage_edit",
        model_info=ModelInfo(input_cost_per_token=1e-06),
    )

    assert cost == pytest.approx(2 * prices.get("input_cost_per_image", 0.0) + prices["output_cost_per_image"])
