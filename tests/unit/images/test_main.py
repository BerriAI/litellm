import asyncio
import base64
import json
from datetime import datetime
from pathlib import Path
from collections.abc import Generator
from queue import SimpleQueue
from typing import Final, Literal, TypedDict
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from openai import AsyncOpenAI, OpenAI
from openai.types.image import Image
from pydantic import JsonValue, TypeAdapter
from typing_extensions import ReadOnly

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.utils import CallTypes, StandardLoggingPayload
import os


class ImageRetryParameters(TypedDict, total=False):
    max_retries: ReadOnly[int | None]


@pytest.fixture
def image_retry_delays() -> Generator[None, None, None]:
    with patch("time.sleep"), patch("asyncio.sleep", new_callable=AsyncMock):
        yield


def _image_retry_transport(
    requests: SimpleQueue[httpx.Request], outcome: Literal["503", "timeout", "success"]
) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        requests.put(request)
        assert request.method == "POST"
        assert request.url.path == "/v1/images/generations"
        body: Final = TypeAdapter(dict[str, JsonValue]).validate_json(request.content)
        assert body == {"model": "gpt-image-1-mini", "prompt": "a red circle"}
        if outcome == "timeout":
            raise httpx.ReadTimeout("scripted image timeout", request=request)
        if outcome == "503":
            return httpx.Response(503, json={"error": {"message": "scripted unavailable", "type": "server_error"}})
        return httpx.Response(200, json={"created": 1700000000, "data": [{"b64_json": "aW1n"}]})

    return httpx.MockTransport(respond)


@pytest.mark.asyncio
@pytest.mark.usefixtures("image_retry_delays")
@pytest.mark.parametrize("is_async", (False, True), ids=("sync", "async"))
@pytest.mark.parametrize("outcome", ("503", "timeout"))
@pytest.mark.parametrize(
    ("retries", "attempts"),
    ((0, 1), (1, 2), (None, 3), ("unset", 3), ("global", 3)),
    ids=("zero", "one", "none", "unset", "global"),
)
async def test_image_generation_honors_sdk_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
    is_async: bool,
    outcome: Literal["503", "timeout"],
    retries: int | None | Literal["unset", "global"],
    attempts: int,
) -> None:
    requests: Final[SimpleQueue[httpx.Request]] = SimpleQueue()
    transport: Final = _image_retry_transport(requests, outcome)
    if retries == "global":
        monkeypatch.setattr(litellm, "max_retries", 0, raising=False)
    retry_params: Final[ImageRetryParameters] = {} if isinstance(retries, str) else {"max_retries": retries}
    error: Final = litellm.Timeout if outcome == "timeout" else litellm.ServiceUnavailableError
    if is_async:
        async with AsyncOpenAI(
            api_key="synthetic-image-key",
            base_url="https://image.test/v1",
            max_retries=5,
            http_client=httpx.AsyncClient(transport=transport),
        ) as client:
            with pytest.raises(error):
                await litellm.aimage_generation(
                    model="openai/gpt-image-1-mini",
                    prompt="a red circle",
                    api_key="synthetic-image-key",
                    api_base="https://image.test/v1",
                    client=client,
                    **retry_params,
                )
    else:
        with OpenAI(
            api_key="synthetic-image-key",
            base_url="https://image.test/v1",
            max_retries=5,
            http_client=httpx.Client(transport=transport),
        ) as client:
            with pytest.raises(error):
                litellm.image_generation(
                    model="openai/gpt-image-1-mini",
                    prompt="a red circle",
                    api_key="synthetic-image-key",
                    api_base="https://image.test/v1",
                    client=client,
                    **retry_params,
                )
    assert requests.qsize() == attempts


@pytest.mark.asyncio
@pytest.mark.usefixtures("image_retry_delays")
@pytest.mark.parametrize("is_async", (False, True), ids=("sync", "async"))
@pytest.mark.parametrize("provider", ("litellm_proxy", "hosted_vllm"))
async def test_compatible_image_generation_honors_zero_retries(provider: str, is_async: bool) -> None:
    requests: Final[SimpleQueue[httpx.Request]] = SimpleQueue()
    transport: Final = _image_retry_transport(requests, "503")
    if is_async:
        async with AsyncOpenAI(
            api_key="synthetic-image-key",
            base_url="https://image.test/v1",
            http_client=httpx.AsyncClient(transport=transport),
        ) as client:
            with pytest.raises(litellm.ServiceUnavailableError):
                await litellm.aimage_generation(
                    model=f"{provider}/gpt-image-1-mini",
                    prompt="a red circle",
                    api_key="synthetic-image-key",
                    api_base="https://image.test/v1",
                    client=client,
                    max_retries=0,
                )
    else:
        with OpenAI(
            api_key="synthetic-image-key",
            base_url="https://image.test/v1",
            http_client=httpx.Client(transport=transport),
        ) as client:
            with pytest.raises(litellm.ServiceUnavailableError):
                litellm.image_generation(
                    model=f"{provider}/gpt-image-1-mini",
                    prompt="a red circle",
                    api_key="synthetic-image-key",
                    api_base="https://image.test/v1",
                    client=client,
                    max_retries=0,
                )
    assert requests.qsize() == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures("image_retry_delays")
@pytest.mark.parametrize("is_async", (False, True), ids=("sync", "async"))
async def test_image_generation_rejects_non_integer_retries_before_sending(is_async: bool) -> None:
    requests: Final[SimpleQueue[httpx.Request]] = SimpleQueue()
    transport: Final = _image_retry_transport(requests, "success")
    if is_async:
        async with AsyncOpenAI(
            api_key="synthetic-image-key",
            base_url="https://image.test/v1",
            http_client=httpx.AsyncClient(transport=transport),
        ) as client:
            with pytest.raises(litellm.BadRequestError) as raised:
                await litellm.aimage_generation(
                    model="openai/gpt-image-1-mini",
                    prompt="a red circle",
                    api_key="synthetic-image-key",
                    api_base="https://image.test/v1",
                    client=client,
                    max_retries="invalid",
                )
    else:
        with OpenAI(
            api_key="synthetic-image-key",
            base_url="https://image.test/v1",
            http_client=httpx.Client(transport=transport),
        ) as client:
            with pytest.raises(litellm.BadRequestError) as raised:
                litellm.image_generation(
                    model="openai/gpt-image-1-mini",
                    prompt="a red circle",
                    api_key="synthetic-image-key",
                    api_base="https://image.test/v1",
                    client=client,
                    max_retries="invalid",
                )
    assert raised.value.status_code == 422
    assert "max retries must be an int" in raised.value.message
    assert requests.qsize() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", (False, True), ids=("sync", "async"))
async def test_image_generation_returns_success_without_forwarding_retry_control(is_async: bool) -> None:
    requests: Final[SimpleQueue[httpx.Request]] = SimpleQueue()
    transport: Final = _image_retry_transport(requests, "success")
    if is_async:
        async with AsyncOpenAI(
            api_key="synthetic-image-key",
            base_url="https://image.test/v1",
            http_client=httpx.AsyncClient(transport=transport),
        ) as client:
            async_response: Final = await litellm.aimage_generation(
                model="openai/gpt-image-1-mini",
                prompt="a red circle",
                api_key="synthetic-image-key",
                api_base="https://image.test/v1",
                client=client,
                max_retries=0,
            )
            async_data: Final = TypeAdapter(tuple[Image, ...]).validate_python(async_response.data)
            assert async_data[0].b64_json == "aW1n"
    else:
        with OpenAI(
            api_key="synthetic-image-key",
            base_url="https://image.test/v1",
            http_client=httpx.Client(transport=transport),
        ) as client:
            sync_response: Final = litellm.image_generation(
                model="openai/gpt-image-1-mini",
                prompt="a red circle",
                api_key="synthetic-image-key",
                api_base="https://image.test/v1",
                client=client,
                max_retries=0,
            )
            sync_data: Final = TypeAdapter(tuple[Image, ...]).validate_python(sync_response.data)
            assert sync_data[0].b64_json == "aW1n"
    assert requests.qsize() == 1


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


@pytest.mark.asyncio
async def test_aimage_generation_routes_a_bare_model_name_to_its_explicit_provider(
    respx_mock: respx.MockRouter,
) -> None:
    api_base: Final = "http://localhost:12348/v1"
    mock_route: Final = respx_mock.post(url__regex=rf"{api_base}/images/generations.*").mock(
        return_value=httpx.Response(status_code=200, json={"created": 1712697600, "data": [{"b64_json": "aW1n"}]})
    )

    response: Final = await litellm.aimage_generation(
        model="Qwen-Image",
        prompt="a red circle",
        custom_llm_provider="openai",
        api_base=api_base,
        api_key="fake_openai_api_key",
        client=AsyncOpenAI(api_key="fake_openai_api_key", base_url=api_base, http_client=httpx.AsyncClient()),
    )

    assert mock_route.called
    sent: Final = json.loads(respx_mock.calls[0].request.content)
    assert sent["model"] == "Qwen-Image"
    assert response.data[0].b64_json == "aW1n"


@pytest.mark.asyncio
@pytest.mark.parametrize("hint", ["custom_llm_provider", "api_base"])
async def test_aimage_generation_answers_a_non_string_provider_hint_with_a_bad_request(
    hint: str, respx_mock: respx.MockRouter
) -> None:
    with pytest.raises(litellm.BadRequestError) as raised:
        await litellm.aimage_generation(
            model="Qwen-Image", prompt="a red circle", api_key="fake_openai_api_key", **{hint: 123}
        )

    assert raised.value.status_code == 400
    assert f"Invalid {hint}" in raised.value.message
    assert not respx_mock.calls


@pytest.mark.asyncio
async def test_aimage_generation_ignores_a_non_string_base_url_like_the_sync_path(
    respx_mock: respx.MockRouter,
) -> None:
    api_base: Final = "http://localhost:12349/v1"
    mock_route: Final = respx_mock.post(url__regex=rf"{api_base}/images/generations.*").mock(
        return_value=httpx.Response(status_code=200, json={"created": 1712697600, "data": [{"b64_json": "aW1n"}]})
    )

    response: Final = await litellm.aimage_generation(
        model="gpt-image-2.5-sunburst",
        prompt="a red circle",
        api_base=api_base,
        base_url=httpx.URL(api_base),
        api_key="fake_openai_api_key",
        client=AsyncOpenAI(api_key="fake_openai_api_key", base_url=api_base, http_client=httpx.AsyncClient()),
    )

    assert mock_route.called
    assert response.data[0].b64_json == "aW1n"


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


def test_image_edit_accepts_a_string_additional_drop_params(respx_mock: respx.MockRouter) -> None:
    edits_route: Final = respx_mock.post("https://api.openai.com/v1/images/edits").mock(
        return_value=httpx.Response(status_code=200, json={"created": 1792551838, "data": [{"b64_json": "aW1n"}]})
    )

    response: Final = litellm.image_edit(
        model="openai/gpt-image-2.5-sunburst",
        image=b"\x89PNG\r\n\x1a\nfakepng",
        prompt="make the circle blue",
        api_key="sk-test",
        quality="high",
        additional_drop_params="quality",
    )

    assert edits_route.called
    assert response.data[0].b64_json == "aW1n"


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


@pytest.mark.asyncio
async def test_aiml_image_generation_with_dynamic_api_key():
    """
    Test that when api_key is passed as a dynamic parameter to aimage_generation,
    it gets properly used for AIML provider authentication instead of falling back
    to environment variables.

    This test validates the fix for ensuring dynamic API keys are respected
    when making image generation requests to the AIML provider.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    import httpx

    # Mock AIML response
    mock_aiml_response = {
        "created": 1703658209,
        "data": [{"url": "https://example.com/generated_image.png"}],
    }

    # Track captured arguments
    captured_headers = None
    captured_url = None
    captured_json_data = None

    def capture_post_call(*args, **kwargs):
        nonlocal captured_headers, captured_url, captured_json_data
        captured_url = kwargs.get("url") or (args[0] if args else None)
        captured_headers = kwargs.get("headers", {})
        captured_json_data = kwargs.get("json", {})

        # Create a mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_aiml_response
        mock_response.text = json.dumps(mock_aiml_response)
        return mock_response

    # Mock the HTTP client that actually makes the request (sync version for image generation)
    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
        mock_post.side_effect = capture_post_call

        # Test with dynamic api_key
        test_api_key = "test-dynamic-api-key-12345"

        response = await litellm.aimage_generation(
            prompt="A cute baby sea otter",
            model="aiml/flux-pro/v1.1",
            api_key=test_api_key,  # This should be used instead of env vars
        )

        # Validate the response (mocked response processing might not populate data correctly)
        assert response is not None

        # The most important validations: API key and endpoint usage
        # These prove that the dynamic API key was properly used
        assert captured_headers is not None
        assert "Authorization" in captured_headers
        assert captured_headers["Authorization"] == f"Bearer {test_api_key}"
        print("TESTCAPTURED HEADERS", captured_headers)
        # Validate the correct AIML endpoint was called
        assert captured_url is not None
        assert "api.aimlapi.com" in captured_url
        assert "/v1/images/generations" in captured_url

        # Validate the request data
        assert captured_json_data is not None
        assert captured_json_data["prompt"] == "A cute baby sea otter"
        assert captured_json_data["model"] == "flux-pro/v1.1"


@pytest.mark.asyncio
async def test_aiml_openai_gpt_image_2_request_uses_openai_param_shape():
    """End-to-end check that ``aiml/openai/gpt-image-2`` keeps the upstream
    OpenAI request shape (``size``/``n``/``response_format``) instead of
    being remapped to the AI/ML flux schema (``image_size``/``num_images``/
    ``output_format``), and hits the correct upstream model name.
    """
    import json as _json
    from unittest.mock import MagicMock, patch

    mock_aiml_response = {
        "created": 1703658209,
        "data": [{"url": "https://example.com/gpt-image-2.png"}],
    }

    captured = {}

    def capture_post_call(*args, **kwargs):
        captured["url"] = kwargs.get("url") or (args[0] if args else None)
        captured["headers"] = kwargs.get("headers", {})
        captured["json"] = kwargs.get("json", {})
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_aiml_response
        mock_response.text = _json.dumps(mock_aiml_response)
        return mock_response

    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
        mock_post.side_effect = capture_post_call

        await litellm.aimage_generation(
            prompt="A T-Rex relaxing on a beach",
            model="aiml/openai/gpt-image-2",
            api_key="test-key-mocked-no-credits-needed",
            size="1024x1536",
            quality="high",
            response_format="b64_json",
            n=1,
        )

    assert captured["url"] is not None
    assert "api.aimlapi.com" in captured["url"]
    assert "/v1/images/generations" in captured["url"]

    body = captured["json"]
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
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("transport", ["headers", "extra_headers", "scoped", "api_key"])
@pytest.mark.parametrize("saved_key", [None, "fake-saved-gateway-key"])
async def test_chatgpt_images_forward_client_login_and_workspace(
    asynchronous: bool,
    transport: str,
    saved_key: str | None,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    claims: Final = {"https://api.openai.com/auth": {"chatgpt_account_id": "client-account"}}
    payload: Final = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    token: Final = f"test.{payload}.test"
    headers: Final = {"aUtHoRiZaTiOn": f"Bearer {token}", "ChatGPT-Account-Id": "selected-workspace"}
    credentials: Final = (
        {"provider_specific_header": {"custom_llm_provider": "chatgpt", "extra_headers": headers}}
        if transport == "scoped"
        else {"api_key": token, "extra_headers": {"cHaTgPt-AcCoUnT-Id": "selected-workspace"}}
        if transport == "api_key"
        else {transport: {**headers, "cookie": "fake-cookie", "x-litellm-api-key": "fake-gateway-key"}}
    )
    route: Final = respx_mock.post("https://chatgpt.com/backend-api/codex/images/generations").mock(
        return_value=httpx.Response(200, json={"created": 1700000000, "data": [{"b64_json": "aW1n"}]})
    )
    params: Final = {
        "model": "chatgpt/gpt-image-2",
        "prompt": "A blue circle",
        "api_key": saved_key,
        **(
            {
                "client": AsyncOpenAI(
                    api_key="fake", base_url="https://chatgpt.com/backend-api/codex", http_client=httpx.AsyncClient()
                )
            }
            if asynchronous
            else {}
        ),
        **credentials,
    }
    response: Final = await litellm.aimage_generation(**params) if asynchronous else litellm.image_generation(**params)
    assert response.data[0].b64_json == "aW1n"
    request: Final = route.calls.last.request
    assert request.headers["authorization"] == f"Bearer {token}"
    assert request.headers["chatgpt-account-id"] == "selected-workspace"
    assert request.headers.get_list("content-type") == ["application/json"]
    assert "cookie" not in request.headers
    assert "x-litellm-api-key" not in request.headers
    assert json.loads(request.content) == {"model": "gpt-image-2", "prompt": "A blue circle"}


def test_image_generation_without_a_model_or_provider_fails_before_calling_any_provider(
    respx_mock: respx.MockRouter,
) -> None:
    with pytest.raises(litellm.BadRequestError, match="model is required"):
        litellm.image_generation(prompt="a red circle", api_key="fake_openai_api_key")

    assert not respx_mock.calls


def test_topaz_image_variation_works_without_a_model(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://api.topazlabs.com/image/v1/enhance").mock(
        return_value=httpx.Response(200, content=b"enhanced-image", headers={"content-type": "image/png"})
    )

    response: Final = litellm.image_variation(image=b"source-image", custom_llm_provider="topaz", api_key="fake-key")

    assert route.call_count == 1
    assert response.data is not None
    assert response.data[0].b64_json == base64.b64encode(b"enhanced-image").decode()
