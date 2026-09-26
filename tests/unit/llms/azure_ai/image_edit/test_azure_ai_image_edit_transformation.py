import base64
import contextlib
import io
import json
import pathlib
import struct
import tempfile
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime
from typing import Final

import httpx
import pytest

import litellm
from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.azure_ai.image_edit.flux2_transformation import (
    UNMEASURED_REFERENCE_IMAGE_PIXELS,
    AzureFoundryFlux2ImageEditConfig,
)
from litellm.llms.azure_ai.image_edit.transformation import (
    AzureFoundryFluxImageEditConfig,
)
from litellm.llms.custom_httpx import llm_http_handler as llm_http_handler_module
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


def test_azure_ai_validate_environment():
    """Test Azure AI environment validation"""
    config = AzureFoundryFluxImageEditConfig()

    headers = {}
    config.validate_environment(headers, "FLUX.1-Kontext-pro", api_key="test-key")
    assert "Api-Key" in headers
    assert headers["Api-Key"] == "test-key"


def test_azure_ai_url_generation():
    """Test Azure AI URL generation"""
    config = AzureFoundryFluxImageEditConfig()

    api_base = "https://test-endpoint.eastus2.inference.ai.azure.com"
    complete_url = config.get_complete_url(
        model="FLUX.1-Kontext-pro",
        api_base=api_base,
        litellm_params={"api_version": "2025-04-01-preview"},
    )
    expected_url = f"{api_base}/openai/deployments/FLUX.1-Kontext-pro/images/edits?api-version=2025-04-01-preview"
    assert complete_url == expected_url


def test_azure_ai_validate_environment_with_entra_token(monkeypatch, no_ambient_azure_credentials):
    monkeypatch.delenv("AZURE_AI_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    config = AzureFoundryFluxImageEditConfig()

    headers = config.validate_environment(
        {},
        "FLUX.1-Kontext-pro",
        litellm_params={"azure_ad_token": "entra-token"},
    )

    assert headers == {"Authorization": "Bearer entra-token"}


def test_flux2_validate_environment_with_entra_token(monkeypatch, no_ambient_azure_credentials):
    monkeypatch.delenv("AZURE_AI_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    config = AzureFoundryFlux2ImageEditConfig()

    headers = config.validate_environment(
        {},
        "flux.2-pro",
        litellm_params={"azure_ad_token": "entra-token"},
    )

    assert headers["Authorization"] == "Bearer entra-token"
    assert headers["Content-Type"] == "application/json"


def test_flux2_image_edit_maps_openai_and_provider_parameters():
    config = AzureFoundryFlux2ImageEditConfig()
    requested_params = ImageEditRequestUtils.get_requested_image_edit_optional_param(
        {
            "n": 2,
            "size": "1536x1024",
            "guidance": 4.5,
            "steps": 32,
            "unrelated": "discarded",
        },
        provider_supported_params=config.get_supported_openai_params("FLUX.2-flex"),
    )
    mapped_params = config.map_openai_params(
        image_edit_optional_params=requested_params,
        model="FLUX.2-flex",
        drop_params=False,
    )

    assert mapped_params == {
        "num_images": 2,
        "width": 1536,
        "height": 1024,
        "guidance": 4.5,
        "steps": 32,
    }


@pytest.mark.parametrize(
    ("model", "max_reference_images"),
    [
        ("FLUX.2-flex", 10),
        ("FLUX.2-pro", 8),
    ],
)
def test_flux2_image_edit_uses_all_reference_fields(model: str, max_reference_images: int):
    images = [f"image-{index}".encode() for index in range(1, max_reference_images + 1)]
    request, files = AzureFoundryFlux2ImageEditConfig().transform_image_edit_request(
        model=model,
        prompt="Blend every reference",
        image=images,
        image_edit_optional_request_params={"guidance": 4.5, "steps": 20},
        litellm_params={},
        headers={},
    )

    assert files == []
    assert request["input_image"] == base64.b64encode(images[0]).decode()
    assert request[f"input_image_{max_reference_images}"] == base64.b64encode(images[-1]).decode()
    assert "input_image_1" not in request
    assert "image" not in request
    assert len([key for key in request if key.startswith("input_image")]) == max_reference_images
    assert request["guidance"] == 4.5
    assert request["steps"] == 20


@pytest.mark.parametrize(
    ("model", "reference_images"),
    [
        ("FLUX.2-flex", 11),
        ("FLUX.2-pro", 9),
    ],
)
def test_flux2_image_edit_rejects_too_many_references(model: str, reference_images: int):
    with pytest.raises(ValueError, match=f"at most {reference_images - 1} reference images"):
        AzureFoundryFlux2ImageEditConfig().transform_image_edit_request(
            model=model,
            prompt="Blend every reference",
            image=[b"image"] * reference_images,
            image_edit_optional_request_params={},
            litellm_params={},
            headers={},
        )


@pytest.mark.parametrize(
    "dimensions", ({"size": "2048x1024"}, {"width": 2048, "height": 1024}, {"width": "2048", "height": "1024"})
)
@pytest.mark.usefixtures("local_model_cost_map")
def test_flux2_image_edit_preserves_controls_and_pixel_cost(dimensions: Mapping[str, int | str]):
    def respond(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        assert body == {
            "model": "FLUX.2-flex",
            "prompt": "Add a hat",
            "input_image": base64.b64encode(_png(512, 512)).decode(),
            "num_images": 2,
            "width": 2048,
            "height": 1024,
            "guidance": 4.5,
            "steps": 32,
        }
        return httpx.Response(200, json={"data": [{"b64_json": "aW1n"}, {"b64_json": "aW1n"}]})

    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    response: Final = litellm.image_edit(
        model="azure_ai/FLUX.2-flex",
        image=_png(512, 512),
        prompt="Add a hat",
        api_key="test-key",
        api_base="https://example.services.ai.azure.com",
        client=client,
        n=2,
        guidance="4.5",
        steps="32",
        **dimensions,
    )
    generated_rate, reference_rate = _flex_rates()

    assert response._hidden_params["response_cost"] == pytest.approx(
        generated_rate * 2048 * 1024 * 2 + reference_rate * 1024 * 1024
    )


def test_flux2_image_edit_accepts_and_drops_openai_only_parameters():
    optional_params: Final = ImageEditRequestUtils.get_optional_params_image_edit(
        model="FLUX.2-pro",
        image_edit_provider_config=AzureFoundryFlux2ImageEditConfig(),
        image_edit_optional_params={"n": 1, "size": "auto", "quality": "high", "user": "end-user-1"},
        drop_params=False,
    )

    assert optional_params == {"num_images": 1}


def _png(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x02\x00\x00\x00"
    )


def _jpeg(width: int, height: int) -> bytes:
    return (
        b"\xff\xd8\xff\xc0" + struct.pack(">HBHHB", 17, 8, height, width, 3) + b"\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    )


def _webp(width: int, height: int) -> bytes:
    payload: Final = b"\x00\x00\x00\x9d\x01\x2a" + struct.pack("<HH", width, height)
    return (
        b"RIFF" + struct.pack("<I", 12 + len(payload)) + b"WEBP" + b"VP8 " + struct.pack("<I", len(payload)) + payload
    )


def _flex_rates() -> tuple[float, float]:
    row: Final = litellm.model_cost["azure_ai/FLUX.2-flex"]
    return row["input_cost_per_pixel"], row["input_cost_per_reference_pixel"]


def _edit_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"data": [{"b64_json": "aW1n"}]})


def test_flux2_image_edit_measures_every_reference_but_bills_each_of_several_as_one_megapixel():
    sent: Final[dict[str, object]] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return _edit_ok(request)

    references: Final = (_png(1024, 1024), _jpeg(800, 600), _webp(640, 480))
    response: Final = litellm.image_edit(
        model="azure_ai/FLUX.2-flex",
        image=list(references),
        prompt="Blend every reference",
        api_key="test-key",
        api_base="https://example.services.ai.azure.com",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond))),
        size="1024x1024",
    )
    generated_rate, reference_rate = _flex_rates()

    assert sent["input_image"] == base64.b64encode(references[0]).decode()
    assert sent["input_image_3"] == base64.b64encode(references[2]).decode()
    assert response._hidden_params["reference_image_pixels"] == (1024 * 1024, 800 * 600, 640 * 480)
    assert response._hidden_params["response_cost"] == pytest.approx(
        generated_rate * 1024 * 1024 + reference_rate * 3 * 1024 * 1024
    )


class _ReadOnlyUpload:
    def __init__(self, data: bytes) -> None:
        self._data: Final = data

    def read(self) -> bytes:
        return self._data


class _SeekableUploadWithoutSeekable:
    def __init__(self, data: bytes) -> None:
        self._stream: Final = io.BytesIO(data)
        self._stream.read()

    def read(self) -> bytes:
        return self._stream.read()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._stream.seek(offset, whence)


class _UploadWithBrokenSeek:
    def __init__(self, data: bytes) -> None:
        self._data: Final = data

    def read(self) -> bytes:
        return self._data

    def seek(self, offset: int, whence: int = 0) -> int:
        raise io.UnsupportedOperation("seek")


class _NonSeekableUpload:
    def __init__(self, data: bytes) -> None:
        self._data: Final = data

    def read(self) -> bytes:
        return self._data

    def seekable(self) -> bool:
        return False

    def seek(self, offset: int, whence: int = 0) -> int:
        raise AssertionError("a non-seekable upload must not be seeked")


def _written_temporary_file(stack: contextlib.ExitStack, data: bytes, spooled: bool) -> object:
    upload: Final = stack.enter_context(tempfile.SpooledTemporaryFile() if spooled else tempfile.NamedTemporaryFile())
    upload.write(data)
    upload.flush()
    return upload


UPLOADS: Final[Mapping[str, Callable[[contextlib.ExitStack, pathlib.Path, bytes], object]]] = {
    "bytesio": lambda _stack, _path, data: io.BytesIO(data),
    "buffered-reader": lambda stack, path, _data: stack.enter_context(path.open("rb")),
    "named-temporary-file-at-eof": lambda stack, _path, data: _written_temporary_file(stack, data, spooled=False),
    "spooled-temporary-file-at-eof": lambda stack, _path, data: _written_temporary_file(stack, data, spooled=True),
    "duck-typed-read-only": lambda _stack, _path, data: _ReadOnlyUpload(data),
    "duck-typed-seek-without-seekable-at-eof": lambda _stack, _path, data: _SeekableUploadWithoutSeekable(data),
    "duck-typed-seek-raises": lambda _stack, _path, data: _UploadWithBrokenSeek(data),
    "non-seekable-stream": lambda _stack, _path, data: _NonSeekableUpload(data),
}


@pytest.fixture
def exit_stack() -> Iterator[contextlib.ExitStack]:
    with contextlib.ExitStack() as stack:
        yield stack


@pytest.mark.parametrize("upload_kind", tuple(UPLOADS))
def test_flux2_image_edit_sends_and_bills_every_readable_upload(
    upload_kind: str, tmp_path: pathlib.Path, exit_stack: contextlib.ExitStack
):
    reference: Final = _png(2048, 1024)
    path: Final = tmp_path / "reference.png"
    path.write_bytes(reference)
    sent_images: Final[list[str]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent_images.append(json.loads(request.content)["input_image"])
        return _edit_ok(request)

    response: Final = litellm.image_edit(
        model="azure_ai/FLUX.2-flex",
        image=UPLOADS[upload_kind](exit_stack, path, reference),
        prompt="Make it a watercolor",
        api_key="test-key",
        api_base="https://example.services.ai.azure.com",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond))),
        size="1024x1024",
    )
    generated_rate, reference_rate = _flex_rates()

    assert sent_images == [base64.b64encode(reference).decode()]
    assert response._hidden_params["response_cost"] == pytest.approx(
        generated_rate * 1024 * 1024 + reference_rate * 2 * 1024 * 1024
    )


@pytest.mark.parametrize("upload_kind", ("bytesio", "buffered-reader", "named-temporary-file-at-eof"))
def test_flux2_image_edit_leaves_a_seekable_upload_rewound_for_reuse(
    upload_kind: str, tmp_path: pathlib.Path, exit_stack: contextlib.ExitStack
):
    path: Final = tmp_path / "reference.png"
    path.write_bytes(_png(1024, 1024))
    upload: Final = UPLOADS[upload_kind](exit_stack, path, _png(1024, 1024))

    litellm.image_edit(
        model="azure_ai/FLUX.2-flex",
        image=upload,
        prompt="Make it a watercolor",
        api_key="test-key",
        api_base="https://example.services.ai.azure.com",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_edit_ok))),
        size="1024x1024",
    )

    assert upload.tell() == 0


def test_flux2_image_edit_rejects_a_text_mode_upload_with_a_clear_error(tmp_path: pathlib.Path):
    path: Final = tmp_path / "reference.png"
    path.write_bytes(_png(1024, 1024))

    with path.open("r", encoding="latin-1") as text_upload, pytest.raises(TypeError, match="binary mode"):
        AzureFoundryFlux2ImageEditConfig().transform_image_edit_request(
            model="FLUX.2-flex",
            prompt="Make it a watercolor",
            image=text_upload,
            image_edit_optional_request_params={},
            litellm_params={},
            headers={},
        )


def test_flux2_image_edit_rejects_a_reference_it_cannot_read():
    with pytest.raises(ValueError, match="Unsupported image type"):
        AzureFoundryFlux2ImageEditConfig().transform_image_edit_request(
            model="FLUX.2-flex",
            prompt="Make it a watercolor",
            image="reference.png",
            image_edit_optional_request_params={},
            litellm_params={},
            headers={},
        )


def test_flux2_image_edit_measures_a_stream_reference():
    uploaded: Final = io.BytesIO(_png(2048, 2048))
    response: Final = litellm.image_edit(
        model="azure_ai/FLUX.2-flex",
        image=uploaded,
        prompt="Make it a watercolor",
        api_key="test-key",
        api_base="https://example.services.ai.azure.com",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_edit_ok))),
        size="1024x1024",
    )
    generated_rate, reference_rate = _flex_rates()

    assert response._hidden_params["reference_image_pixels"] == (2048 * 2048,)
    assert response._hidden_params["response_cost"] == pytest.approx(
        generated_rate * 1024 * 1024 + reference_rate * 2048 * 2048
    )


# Billable megapixels for a lone reference as Azure's FLUX.2-pro request_meta reported them on 2026-09-25
@pytest.mark.parametrize(
    ("reference", "billed_megapixels"),
    (
        pytest.param(_png(640, 640), 1, id="small-reference-rounds-up"),
        pytest.param(_png(1024, 1280), 2, id="fractional-reference-rounds-up"),
        pytest.param(_jpeg(4032, 3024), 4, id="photo-reference-caps-at-four-megapixels"),
    ),
)
def test_flux2_image_edit_bills_a_lone_reference_in_whole_megapixels(reference: bytes, billed_megapixels: int):
    response: Final = litellm.image_edit(
        model="azure_ai/FLUX.2-flex",
        image=reference,
        prompt="Make it a watercolor",
        api_key="test-key",
        api_base="https://example.services.ai.azure.com",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_edit_ok))),
        size="1024x1024",
    )
    generated_rate, reference_rate = _flex_rates()

    assert response._hidden_params["response_cost"] == pytest.approx(
        generated_rate * 1024 * 1024 + reference_rate * billed_megapixels * 1024 * 1024
    )


@pytest.mark.parametrize(
    "reference",
    (
        pytest.param(b"BM not a parseable header", id="unsupported-format"),
        pytest.param(b"\x89PNG\r\n\x1a\n\x00\x00", id="truncated-header"),
        pytest.param(_png(0, 640), id="zero-width-header"),
    ),
)
def test_flux2_image_edit_bills_a_lone_unmeasurable_reference_as_one_megapixel(reference: bytes):
    response: Final = litellm.image_edit(
        model="azure_ai/FLUX.2-flex",
        image=[reference],
        prompt="Make it a watercolor",
        api_key="test-key",
        api_base="https://example.services.ai.azure.com",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_edit_ok))),
        size="1024x1024",
    )
    generated_rate, reference_rate = _flex_rates()

    assert response._hidden_params["reference_image_pixels"] == (UNMEASURED_REFERENCE_IMAGE_PIXELS,)
    assert response._hidden_params["response_cost"] == pytest.approx(
        generated_rate * 1024 * 1024 + reference_rate * 1024 * 1024
    )


def test_flux2_image_edit_still_bills_every_reference_when_one_header_reports_zero_pixels():
    response: Final = litellm.image_edit(
        model="azure_ai/FLUX.2-flex",
        image=[_png(1024, 1024), _jpeg(640, 0)],
        prompt="Blend both references",
        api_key="test-key",
        api_base="https://example.services.ai.azure.com",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_edit_ok))),
        size="1024x1024",
    )
    generated_rate, reference_rate = _flex_rates()

    assert response._hidden_params["response_cost"] == pytest.approx(
        generated_rate * 1024 * 1024 + reference_rate * 2 * 1024 * 1024
    )


@pytest.mark.parametrize("stream_position", ("start", "end"))
def test_flux2_image_edit_resends_and_rebills_a_reused_stream(stream_position: str):
    sent_images: Final[list[str]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent_images.append(json.loads(request.content)["input_image"])
        return _edit_ok(request)

    reference: Final = _png(2048, 1024)
    uploaded: Final = io.BytesIO(reference)
    if stream_position == "end":
        uploaded.read()
    responses: Final = tuple(
        litellm.image_edit(
            model="azure_ai/FLUX.2-flex",
            image=[uploaded],
            prompt="Make it a watercolor",
            api_key="test-key",
            api_base="https://example.services.ai.azure.com",
            client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond))),
            size="1024x1024",
        )
        for _attempt in range(2)
    )

    assert sent_images == [base64.b64encode(reference).decode()] * 2
    assert [response._hidden_params["reference_image_pixels"] for response in responses] == [(2048 * 1024,)] * 2


def test_flux2_pro_image_edit_bills_references_on_the_pro_reference_rate():
    pro_row: Final = litellm.model_cost["azure_ai/flux.2-pro"]
    response: Final = litellm.image_edit(
        model="azure_ai/flux.2-pro",
        image=[_png(1024, 1024), _jpeg(4032, 3024)],
        prompt="Blend both references",
        api_key="test-key",
        api_base="https://example.services.ai.azure.com",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_edit_ok))),
        size="1024x1024",
    )

    assert pro_row["input_cost_per_reference_pixel"] > 0
    assert response._hidden_params["response_cost"] == pytest.approx(
        pro_row["output_cost_per_image"] + pro_row["input_cost_per_reference_pixel"] * 2 * 1024 * 1024
    )


async def test_flux2_router_image_edit_bills_the_deployment_rates(monkeypatch: pytest.MonkeyPatch):
    mock_client: Final = AsyncHTTPHandler()
    mock_client.client = httpx.AsyncClient(transport=httpx.MockTransport(_edit_ok))
    monkeypatch.setattr(llm_http_handler_module, "get_async_httpx_client", lambda **_kwargs: mock_client)
    generated_rate: Final = 1e-07
    reference_rate: Final = 2e-07
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "flux2-flex-deployment",
                "litellm_params": {
                    "model": "azure_ai/FLUX.2-flex",
                    "api_base": "https://example.services.ai.azure.com",
                    "api_key": "test-key",
                    "input_cost_per_pixel": generated_rate,
                    "input_cost_per_reference_pixel": reference_rate,
                },
            }
        ]
    )

    response: Final = await router.aimage_edit(
        model="flux2-flex-deployment",
        prompt="Make it a watercolor",
        image=[_png(1024, 1280)],
        size="1024x1280",
    )

    assert response._hidden_params["response_cost"] == pytest.approx(
        generated_rate * 2 * 1024 * 1024 + reference_rate * 2 * 1024 * 1024
    )


@pytest.mark.parametrize(
    ("deployment_prices", "expected_cost"),
    (
        ({"output_cost_per_image": 0.5}, 0.5),
        (
            {"input_cost_per_pixel": 1e-07, "input_cost_per_reference_pixel": 2e-07},
            1e-07 * 2 * 1024 * 1024 + 2e-07 * 2 * 1024 * 1024,
        ),
    ),
    ids=("flat", "per-pixel"),
)
async def test_flux2_router_image_edit_bills_the_deployment_rates_with_a_logger_built_before_routing(
    monkeypatch: pytest.MonkeyPatch, deployment_prices: Mapping[str, float], expected_cost: float
):
    mock_client: Final = AsyncHTTPHandler()
    mock_client.client = httpx.AsyncClient(transport=httpx.MockTransport(_edit_ok))
    monkeypatch.setattr(llm_http_handler_module, "get_async_httpx_client", lambda **_kwargs: mock_client)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "flux2-pro-deployment",
                "litellm_params": {
                    "model": "azure_ai/flux.2-pro",
                    "api_base": "https://example.services.ai.azure.com",
                    "api_key": "test-key",
                    **deployment_prices,
                },
            }
        ]
    )
    request: Final = {
        "model": "flux2-pro-deployment",
        "prompt": "Make it a watercolor",
        "image": [_png(1024, 1280)],
        "size": "1024x1280",
        "litellm_call_id": "proxy-call-id",
    }
    logging_obj, routed_request = litellm.utils.function_setup(
        original_function="aimage_edit", rules_obj=litellm.utils.Rules(), start_time=datetime.now(), **request
    )

    response: Final = await router.aimage_edit(**routed_request, litellm_logging_obj=logging_obj)

    assert response._hidden_params["response_cost"] == pytest.approx(expected_cost)


def test_flux2_image_edit_surfaces_a_response_that_is_not_json():
    with pytest.raises(litellm.APIError, match="gateway timeout page"):
        litellm.image_edit(
            model="azure_ai/flux.2-pro",
            image=_png(1024, 1024),
            prompt="Make it a watercolor",
            api_key="test-key",
            api_base="https://example.services.ai.azure.com",
            client=HTTPHandler(
                client=httpx.Client(
                    transport=httpx.MockTransport(lambda _request: httpx.Response(200, text="gateway timeout page"))
                )
            ),
            size="1024x1024",
        )


def _edit_returning(image: bytes) -> Callable[[httpx.Request], httpx.Response]:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(image).decode()}]})

    return respond


@pytest.mark.parametrize("size", (None, "1024x1024"), ids=("no-size", "size-azure-did-not-use"))
def test_flux2_image_edit_bills_the_generated_image_azure_returned(size: str | None):
    response: Final = litellm.image_edit(
        model="azure_ai/flux.2-pro",
        image=_png(1024, 1280),
        prompt="Make it a watercolor",
        api_key="test-key",
        api_base="https://example.services.ai.azure.com",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_edit_returning(_png(1024, 1280))))),
        **({} if size is None else {"size": size}),
    )
    pro_row: Final = litellm.model_cost["azure_ai/flux.2-pro"]

    assert response._hidden_params["response_cost"] == pytest.approx(
        pro_row["output_cost_per_image"]
        + pro_row["input_cost_per_pixel"] * 1024 * 1024
        + pro_row["input_cost_per_reference_pixel"] * 2 * 1024 * 1024
    )


async def test_flux2_aimage_edit_bills_references_like_image_edit():
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(_edit_ok))
    response: Final = await litellm.aimage_edit(
        model="azure_ai/FLUX.2-flex",
        image=[_png(1024, 1024), _png(1024, 1024)],
        prompt="Blend both references",
        api_key="test-key",
        api_base="https://example.services.ai.azure.com",
        client=client,
        size="1024x1024",
    )
    generated_rate, reference_rate = _flex_rates()

    assert response._hidden_params["reference_image_pixels"] == (1024 * 1024, 1024 * 1024)
    assert response._hidden_params["response_cost"] == pytest.approx(
        generated_rate * 1024 * 1024 + reference_rate * 2 * 1024 * 1024
    )
