import asyncio
import io
from collections.abc import Iterator, Mapping
from datetime import datetime
from typing import Final

import httpx
import pytest
import respx
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict, override

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ImageResponse

_PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"
_FIRST_IMAGE: Final = _PNG_SIGNATURE + b"first-reference-image"
_SECOND_IMAGE: Final = _PNG_SIGNATURE + b"second-reference-image"
_EDITED_IMAGE_B64: Final = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
)
_TEXT_TOKENS: Final = 50
_IMAGE_TOKENS: Final = 50
_OUTPUT_TOKENS: Final = 1000
_EDIT_RESPONSE: Final = {
    "created": 1589478378,
    "data": [{"b64_json": _EDITED_IMAGE_B64}],
    "usage": {
        "total_tokens": _TEXT_TOKENS + _IMAGE_TOKENS + _OUTPUT_TOKENS,
        "input_tokens": _TEXT_TOKENS + _IMAGE_TOKENS,
        "input_tokens_details": {"image_tokens": _IMAGE_TOKENS, "text_tokens": _TEXT_TOKENS},
        "output_tokens": _OUTPUT_TOKENS,
    },
}


class _LoggedImageEdit(TypedDict):
    model: ReadOnly[str]
    custom_llm_provider: ReadOnly[str]
    response_cost: ReadOnly[float]


_LOGGED_IMAGE_EDIT: Final = TypeAdapter(_LoggedImageEdit)


class _SuccessLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.payload: _LoggedImageEdit | None = None
        self.logged: Final = asyncio.Event()

    @override
    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.payload = _LOGGED_IMAGE_EDIT.validate_python(kwargs.get("standard_logging_object"))
        self.logged.set()


@pytest.fixture
def httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


def _multipart_image_parts(request: httpx.Request) -> tuple[bytes, ...]:
    body: Final = request.read()
    boundary: Final = request.headers["content-type"].split("boundary=", 1)[1].encode()
    parts: Final = body.split(b"--" + boundary)
    return tuple(part.split(b"\r\n\r\n", 1)[1].removesuffix(b"\r\n") for part in parts if b'name="image[]"' in part)


def test_openai_image_edit_sync_accepts_bytesio_images(respx_mock: respx.MockRouter, httpx_transport: None) -> None:
    route: Final = respx_mock.post("https://api.openai.com/v1/images/edits").mock(
        return_value=httpx.Response(200, json=_EDIT_RESPONSE)
    )

    result: Final = litellm.image_edit(
        prompt="combine the reference images",
        model="gpt-image-1",
        image=io.BytesIO(_FIRST_IMAGE),
        api_key="fake-key",
    )

    assert isinstance(result, ImageResponse)
    assert result.data is not None and result.data[0].b64_json == _EDITED_IMAGE_B64
    assert route.call_count == 1
    assert _multipart_image_parts(route.calls[0].request) == (_FIRST_IMAGE,)


@pytest.mark.asyncio
async def test_openai_image_edit_accepts_bytesio_images(respx_mock: respx.MockRouter, httpx_transport: None) -> None:
    route: Final = respx_mock.post("https://api.openai.com/v1/images/edits").mock(
        return_value=httpx.Response(200, json=_EDIT_RESPONSE)
    )

    result: Final = await litellm.aimage_edit(
        prompt="combine the reference images",
        model="gpt-image-1",
        image=[io.BytesIO(_FIRST_IMAGE), io.BytesIO(_SECOND_IMAGE)],
        api_key="fake-key",
    )

    assert isinstance(result, ImageResponse)
    assert result.data is not None and result.data[0].b64_json == _EDITED_IMAGE_B64
    assert route.call_count == 1
    assert _multipart_image_parts(route.calls[0].request) == (_FIRST_IMAGE, _SECOND_IMAGE)


@pytest.mark.asyncio
async def test_openai_image_edit_accepts_mixed_bytes_and_bytesio(
    respx_mock: respx.MockRouter, httpx_transport: None
) -> None:
    route: Final = respx_mock.post("https://api.openai.com/v1/images/edits").mock(
        return_value=httpx.Response(200, json=_EDIT_RESPONSE)
    )

    result: Final = await litellm.aimage_edit(
        prompt="Create a cohesive artistic style across all images",
        model="gpt-image-1",
        image=[_FIRST_IMAGE, io.BytesIO(_SECOND_IMAGE)],
        api_key="fake-key",
    )

    assert isinstance(result, ImageResponse)
    assert result.data is not None and len(result.data) == 1
    assert result.data[0].b64_json == _EDITED_IMAGE_B64
    assert route.call_count == 1
    assert _multipart_image_parts(route.calls[0].request) == (_FIRST_IMAGE, _SECOND_IMAGE)


@pytest.mark.asyncio
async def test_azure_image_edit_logs_deployment_model_and_positive_cost(
    respx_mock: respx.MockRouter, httpx_transport: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger: Final = _SuccessLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    route: Final = respx_mock.post(
        url__startswith="https://fake.openai.azure.com/openai/deployments/CUSTOM_AZURE_DEPLOYMENT_NAME/images/edits"
    ).mock(return_value=httpx.Response(200, json=_EDIT_RESPONSE))

    result: Final = await litellm.aimage_edit(
        prompt="combine the reference images",
        model="azure/CUSTOM_AZURE_DEPLOYMENT_NAME",
        base_model="azure/gpt-image-1",
        image=[_FIRST_IMAGE, _SECOND_IMAGE],
        api_key="fake-key",
        api_base="https://fake.openai.azure.com",
        api_version="2025-04-01-preview",
    )
    await logger.logged.wait()

    assert isinstance(result, ImageResponse)
    assert route.call_count == 1
    payload: Final = logger.payload
    assert payload is not None
    assert payload["model"] == "CUSTOM_AZURE_DEPLOYMENT_NAME"
    assert payload["custom_llm_provider"] == "azure"
    pricing: Final = litellm.model_cost["azure/gpt-image-1"]
    expected_cost: Final = (
        _TEXT_TOKENS * pricing["input_cost_per_token"]
        + _IMAGE_TOKENS * pricing["input_cost_per_image_token"]
        + _OUTPUT_TOKENS * pricing["output_cost_per_image_token"]
    )
    assert expected_cost > 0
    assert payload["response_cost"] == pytest.approx(expected_cost)


@pytest.mark.asyncio
async def test_router_image_edit_returns_the_provider_image(
    respx_mock: respx.MockRouter, httpx_transport: None
) -> None:
    route: Final = respx_mock.post("https://api.openai.com/v1/images/edits").mock(
        return_value=httpx.Response(200, json=_EDIT_RESPONSE)
    )
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-image-1",
                "litellm_params": {
                    "model": "gpt-image-1",
                    "api_key": "fake-key",
                },
            }
        ]
    )

    result: Final = await router.aimage_edit(
        prompt="combine the reference images",
        model="gpt-image-1",
        image=_FIRST_IMAGE,
    )

    assert isinstance(result, ImageResponse)
    assert result.data is not None and result.data[0].b64_json == _EDITED_IMAGE_B64
    assert route.call_count == 1
