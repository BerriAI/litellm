import asyncio
import itertools
import json
from datetime import datetime
from collections.abc import Coroutine
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from openai.types.image import Image

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import CallTypes, StandardLoggingPayload
import os


def test_image_generation_keeps_an_internal_prefixed_kwarg_out_of_the_provider_request(
    respx_mock: respx.MockRouter,
) -> None:
    api_base: Final = "http://localhost:12346/v1"
    mock_route: Final = respx_mock.post(url__regex=rf"{api_base}/images/generations.*").mock(
        return_value=httpx.Response(status_code=200, json={"created": 1712697600, "data": [{"b64_json": "aW1n"}]})
    )

    litellm.image_generation(
        model="openai/gpt-image-1",
        prompt="a red circle",
        api_base=api_base,
        api_key="fake_openai_api_key",
        _litellm_undeclared_sentinel="internal",
    )

    assert mock_route.called
    sent: Final = json.loads(respx_mock.calls[0].request.content)
    assert "_litellm_undeclared_sentinel" not in sent, sent
    assert sent["prompt"] == "a red circle"


def test_image_edit_prices_a_vertex_deployment_at_its_configured_location(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_base: Final = "http://localhost:12347/generateContent"
    respx_mock.post(api_base).mock(
        return_value=httpx.Response(
            status_code=200,
            json={"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "aW1n"}}]}}]},
        )
    )
    monkeypatch.setitem(
        litellm.model_cost,
        "vertex_ai/gemini-fake-regional-edit-model",
        {
            "litellm_provider": "vertex_ai-language-models",
            "mode": "image_generation",
            "output_cost_per_image": 0.04,
            "regional_endpoint_uplift_multiplier": 1.1,
        },
    )

    def cost_at(location: str) -> float:
        logging_obj: Final = Logging(
            model="gemini-fake-regional-edit-model",
            messages=[],
            stream=False,
            call_type=CallTypes.image_edit.value,
            start_time=datetime.now(),
            litellm_call_id=f"vertex-edit-{location}",
            function_id="f",
        )
        response: Final = litellm.image_edit(
            model="vertex_ai/gemini-fake-regional-edit-model",
            image=b"\x89PNG\r\n\x1a\nfakepng",
            prompt="make the circle blue",
            api_base=api_base,
            vertex_location=location,
            litellm_logging_obj=logging_obj,
        )
        return logging_obj.response_cost_calculator(result=response)

    assert cost_at("global") == pytest.approx(0.04)
    assert cost_at("us-central1") == pytest.approx(0.044)


class TestCustomLogger(CustomLogger):
    __test__ = False

    def __init__(self) -> None:
        super().__init__()
        self.standard_logging_payload: StandardLoggingPayload | None = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.standard_logging_payload = kwargs.get("standard_logging_object")


class TestAimlImageGeneration:
    def get_base_image_generation_call_args(self) -> dict:
        return {"model": "aiml/flux-pro/v1.1"}

    @pytest.mark.asyncio(scope="module")
    @pytest.mark.flaky(retries=0)
    async def test_basic_image_generation(self):
        """Test basic image generation"""
        from unittest.mock import AsyncMock, patch

        mock_aiml_response = {
            "created": 1703658209,
            "data": [{"url": "https://example.com/generated_image.png"}],
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_aiml_response
        mock_response.text = json.dumps(mock_aiml_response)
        mock_response.headers = {}

        with (
            patch(
                "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
                new_callable=AsyncMock,
            ) as mock_async_post,
            patch(
                "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            ) as mock_sync_post,
        ):
            mock_async_post.return_value = mock_response
            mock_sync_post.return_value = mock_response

            try:
                litellm.turn_on_debug()
                custom_logger = TestCustomLogger()
                litellm.logging_callback_manager._reset_all_callbacks()
                litellm.callbacks = [custom_logger]
                base_image_generation_call_args = (
                    self.get_base_image_generation_call_args()
                )
                litellm.set_verbose = True
                # Pass dummy api_key so validate_environment passes; HTTP is mocked
                response = await litellm.aimage_generation(
                    **base_image_generation_call_args,
                    prompt="A image of a otter",
                    api_key="test-key-mocked-no-credits-needed",
                )
                print("FAL AI RESPONSE: ", response)

                await asyncio.sleep(1)

                # assert response._hidden_params["response_cost"] is not None
                # assert response._hidden_params["response_cost"] > 0
                # print("response_cost", response._hidden_params["response_cost"])

                logged_standard_logging_payload = custom_logger.standard_logging_payload
                print(
                    "logged_standard_logging_payload", logged_standard_logging_payload
                )
                assert logged_standard_logging_payload is not None
                assert logged_standard_logging_payload["response_cost"] is not None
                assert logged_standard_logging_payload["response_cost"] > 0
                import openai
                from openai.types.images_response import ImagesResponse

                # print openai version
                print("openai version=", openai.__version__)

                response_dict = dict(response)
                if "usage" in response_dict:
                    response_dict["usage"] = dict(response_dict["usage"])
                print("response usage=", response_dict.get("usage"))

                assert (
                    response.data is not None
                )  # type guard for iteration (base fails here if None)
                for d in response.data:
                    assert isinstance(d, Image)
                    print("data in response.data", d)
                    assert d.b64_json is not None or d.url is not None
            except litellm.RateLimitError as e:
                pass
            except litellm.ContentPolicyViolationError:
                pass  # Azure randomly raises these errors - skip when they occur
            except litellm.InternalServerError:
                pass
            except Exception as e:
                if "Your task failed as a result of our safety system." in str(e):
                    pass
                else:
                    pytest.fail(f"An exception occurred - {str(e)}")


def _capturing_client(response_json: dict[str, object]) -> tuple[AsyncHTTPHandler, list[httpx.Request]]:
    requests: Final[list[httpx.Request]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=response_json)

    return AsyncHTTPHandler(transport=httpx.MockTransport(handler)), requests


@pytest.mark.asyncio
async def test_aiml_image_generation_with_dynamic_api_key():
    client, requests = _capturing_client(
        {"created": 1703658209, "data": [{"url": "https://example.com/generated_image.png"}]}
    )

    await litellm.aimage_generation(
        prompt="A cute baby sea otter",
        model="aiml/flux-pro/v1.1",
        api_key="test-dynamic-api-key-12345",
        client=client,
    )

    assert len(requests) == 1
    assert requests[0].headers["Authorization"] == "Bearer test-dynamic-api-key-12345"
    assert requests[0].url.host == "api.aimlapi.com"
    assert requests[0].url.path.endswith("/v1/images/generations")
    body: Final = json.loads(requests[0].content)
    assert body["prompt"] == "A cute baby sea otter"
    assert body["model"] == "flux-pro/v1.1"


@pytest.mark.asyncio
async def test_aiml_openai_gpt_image_2_request_uses_openai_param_shape():
    client, requests = _capturing_client(
        {"created": 1703658209, "data": [{"url": "https://example.com/gpt-image-2.png"}]}
    )

    await litellm.aimage_generation(
        prompt="A T-Rex relaxing on a beach",
        model="aiml/openai/gpt-image-2",
        api_key="test-key-mocked-no-credits-needed",
        size="1024x1536",
        quality="high",
        response_format="b64_json",
        n=1,
        client=client,
    )

    assert len(requests) == 1
    assert requests[0].url.host == "api.aimlapi.com"
    assert requests[0].url.path.endswith("/v1/images/generations")
    body: Final = json.loads(requests[0].content)
    assert body["model"] == "openai/gpt-image-2"
    assert body["prompt"] == "A T-Rex relaxing on a beach"
    assert body["size"] == "1024x1536"
    assert body["quality"] == "high"
    assert body["response_format"] == "b64_json"
    assert body["n"] == 1
    assert "image_size" not in body
    assert "num_images" not in body
    assert "output_format" not in body


@pytest.mark.asyncio
async def test_azure_image_generation_request_body():
    """Azure deployment URL selects the model; JSON body omits ``model`` (#26316)."""
    from litellm import aimage_generation

    test_dir = os.path.dirname(__file__)
    expected_path = os.path.join(test_dir, "request_payloads", "azure_gpt_image_1.json")
    with open(expected_path, "r") as f:
        expected_body = json.load(f)

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.side_effect = Exception("test")

        with pytest.raises(litellm.APIConnectionError):
            await aimage_generation(
                model="azure/gpt-image-1",
                prompt="test prompt",
                api_base="https://example.azure.com",
                api_key="test-key",
                api_version="2025-04-01-preview",
            )

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        request_json = call_args.kwargs.get("json", {})
        assert request_json == expected_body


@pytest.mark.asyncio
async def test_aimage_generation_runs_vertex_requests_concurrently_on_the_event_loop() -> None:
    response_json: Final = {
        "candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "aW1n"}}]}}],
    }
    arrivals: Final = itertools.count(1)
    both_in_flight: Final = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if next(arrivals) == 2:
            both_in_flight.set()
        await both_in_flight.wait()
        return httpx.Response(status_code=200, json=response_json)

    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(handler))

    def generate() -> Coroutine[object, object, litellm.ImageResponse]:
        return litellm.aimage_generation(  # pyright: ignore[reportUnknownMemberType]  # aimage_generation kwargs are untyped
            model="vertex_ai/gemini-2.5-flash-image",
            prompt="a red circle",
            api_base="http://localhost:1/generateContent",
            vertex_location="us-central1",
            client=client,
        )

    results: Final = await asyncio.wait_for(asyncio.gather(generate(), generate()), timeout=5)

    assert [result.data[0].b64_json if result.data else None for result in results] == ["aW1n", "aW1n"]
