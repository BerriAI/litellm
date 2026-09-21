from typing import Final

import pytest

import litellm
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension
MODELS: Final = ("cohere/parse-v5.0", "azure_ai/Cohere-parse-v5.0")
IMAGE: Final = {"type": "image_url", "image_url": "data:image/png;base64,YWJj"}
BOX: Final = {"top_left_x": 0, "top_left_y": 0, "bottom_right_x": 32, "bottom_right_y": 32}
PAYLOAD: Final = {
    "pages": [
        {
            "index": 4,
            "markdown": {"content": "receipt", "images": [{"id": "image", "bounding_box": BOX, "description": "scan"}]},
        },
        {"markdown": {"content": "page two"}},
    ],
    "meta": {"billed_units": {"pages": 3}},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_public_cohere_request_and_normalization(
    recording_server: RecordingServer, model: str, asynchronous: bool
) -> None:
    recording_server.enqueue(ResponseSpec(body=PAYLOAD))
    args: Final = {
        "model": model,
        "document": IMAGE,
        "api_base": recording_server.base_url,
        "api_key": "test-key",
        "req_format": "native",
        "unrecognized": True,
    }
    response: Final = await litellm.aocr(**args) if asynchronous else litellm.ocr(**args)
    request: Final = recording_server.requests[0]
    assert request.path == ("/providers/cohere/v2/parse" if model.startswith("azure_ai/") else "/v2/parse")
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.body == {"model": model.split("/", 1)[1], "document": IMAGE, "output_format": "markdown"}
    assert [page.index for page in response.pages] == [4, 1]
    assert response.pages[0].markdown == "receipt"
    assert response.pages[0].images[0].bbox == BOX
    assert response.pages[0].images[0].model_extra["description"] == "scan"
    assert response.pages[1].images is None
    assert response.usage_info.pages_processed == 3
    assert response.get_provider_native_response() == PAYLOAD


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
async def test_public_cohere_blocks_and_usage_fallback(recording_server: RecordingServer, model: str) -> None:
    blocks: Final = [{"type": "text", "text": "total"}]
    recording_server.enqueue(ResponseSpec(body={"pages": [{"blocks": blocks}]}))
    response: Final = await litellm.aocr(
        model=model, document=IMAGE, api_base=recording_server.base_url, api_key="test-key", output_format="blocks"
    )
    assert recording_server.requests[0].body["output_format"] == "blocks"
    assert response.pages[0].model_extra["blocks"] == blocks
    assert response.pages[0].markdown == ""
    assert response.usage_info.pages_processed == 1
    assert response.get_provider_native_response() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "document",
    [
        {"type": "document_url", "document_url": "https://example.com/file.pdf"},
        {"type": "image_url", "image_url": "data:application/pdf;base64,YQ=="},
        {"type": "image_url", "image_url": ""},
    ],
)
async def test_public_cohere_rejects_non_images_before_network(
    recording_server: RecordingServer, model: str, document: dict[str, str]
) -> None:
    recording_server.expected_requests = 0
    with pytest.raises(litellm.BadRequestError, match="only accepts `image_url`"):
        await litellm.aocr(model=model, document=document, api_base=recording_server.base_url, api_key="test-key")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
async def test_public_cohere_rejects_unknown_format(recording_server: RecordingServer, model: str) -> None:
    recording_server.expected_requests = 0
    with pytest.raises(litellm.BadRequestError, match="output_format"):
        await litellm.aocr(
            model=model, document=IMAGE, api_base=recording_server.base_url, api_key="test-key", output_format="html"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
async def test_public_cohere_provider_failure(recording_server: RecordingServer, model: str) -> None:
    recording_server.enqueue(ResponseSpec(status=400, body={"message": "output_format must be blocks or markdown"}))
    with pytest.raises(litellm.BadRequestError, match="output_format must be") as caught:
        await litellm.aocr(model=model, document=IMAGE, api_base=recording_server.base_url, api_key="test-key")
    assert caught.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
async def test_public_cohere_health_check(recording_server: RecordingServer, model: str) -> None:
    recording_server.enqueue(ResponseSpec(body=PAYLOAD))
    response: Final = await litellm.ahealth_check(
        model_params={"model": model, "api_key": "test-key", "api_base": recording_server.base_url}, mode="ocr"
    )
    assert "error" not in response
    assert recording_server.requests[0].body["document"]["image_url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["", "/cohere/", "/v2", "/v2/parse"])
async def test_public_cohere_url_variants(recording_server: RecordingServer, suffix: str) -> None:
    recording_server.enqueue(ResponseSpec(body=PAYLOAD))
    await litellm.aocr(model=MODELS[0], document=IMAGE, api_base=recording_server.base_url + suffix, api_key="test-key")
    assert recording_server.requests[0].path == ("/cohere/v2/parse" if suffix == "/cohere/" else "/v2/parse")


@pytest.mark.asyncio
async def test_public_cohere_environment_key_and_remote_url(
    recording_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "env-key")
    recording_server.enqueue(ResponseSpec(body=PAYLOAD))
    document: Final = {"type": "image_url", "image_url": "https://example.com/receipt.png"}
    await litellm.aocr(model=MODELS[0], document=document, api_base=recording_server.base_url)
    assert recording_server.requests[0].headers["authorization"] == "Bearer env-key"
    assert recording_server.requests[0].body["document"] == document


@pytest.mark.asyncio
async def test_public_cohere_missing_key(recording_server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    recording_server.expected_requests = 0
    with pytest.raises(Exception, match="Missing COHERE_API_KEY"):
        await litellm.aocr(model=MODELS[0], document=IMAGE, api_base=recording_server.base_url)
