import json

import pytest

import litellm

PARSE_URL = "https://api.cohere.com/v2/parse"
MODEL = "cohere/parse-v5.0"
IMAGE_DOCUMENT = {"type": "image_url", "image_url": "https://example.com/receipt.png"}
BOUNDING_BOX = {"top_left_x": 0, "top_left_y": 0, "bottom_right_x": 32, "bottom_right_y": 32}


def _markdown_response(billed_pages: int | None = 2) -> dict:
    return {
        "id": "272900cc-04c0-4da2-a505-2cea58d231bf",
        "pages": [
            {
                "index": 0,
                "type": "markdown",
                "markdown": {
                    "content": "# Receipt\n\nTotal Due: $4.00",
                    "images": [
                        {
                            "id": "img-0",
                            "description": "A parking receipt",
                            "category": "other",
                            "bounding_box": BOUNDING_BOX,
                            "bounding_box_normalized": {
                                "top_left_x": 0,
                                "top_left_y": 0,
                                "bottom_right_x": 1,
                                "bottom_right_y": 1,
                            },
                        }
                    ],
                },
            },
            {"index": 1, "type": "markdown", "markdown": {"content": "Page two"}},
        ],
        **(
            {"meta": {"api_version": {"version": "2"}, "billed_units": {"pages": billed_pages}}} if billed_pages else {}
        ),
    }


def _blocks_response() -> dict:
    return {
        "id": "94474f83-e30d-4763-b4bc-52af6e12c4f7",
        "pages": [
            {
                "index": 0,
                "type": "blocks",
                "blocks": [{"type": "text", "text": "Total Due: $4.00"}],
            }
        ],
        "meta": {"api_version": {"version": "2"}, "billed_units": {"pages": 1}},
    }


@pytest.fixture()
def disable_aiohttp_transport(monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.asyncio
async def test_aocr_sends_markdown_parse_request_and_normalizes_pages(disable_aiohttp_transport, respx_mock):
    route = respx_mock.post(PARSE_URL).respond(json=_markdown_response())

    response = await litellm.aocr(model=MODEL, document=IMAGE_DOCUMENT, api_key="test-key")

    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {
        "model": "parse-v5.0",
        "document": IMAGE_DOCUMENT,
        "output_format": "markdown",
    }
    assert response.object == "ocr"
    assert [page.index for page in response.pages] == [0, 1]
    assert response.pages[0].markdown == "# Receipt\n\nTotal Due: $4.00"
    assert response.pages[1].markdown == "Page two"
    assert response.pages[1].images is None
    image = response.pages[0].images[0]
    assert image.bbox == BOUNDING_BOX
    assert image.model_extra["description"] == "A parking receipt"
    assert image.model_extra["bounding_box_normalized"]["bottom_right_x"] == 1
    assert response.usage_info.pages_processed == 2
    assert response.get_provider_native_response() is None


@pytest.mark.asyncio
async def test_aocr_usage_prefers_billed_units_over_page_count(disable_aiohttp_transport, respx_mock):
    respx_mock.post(PARSE_URL).respond(json=_markdown_response(billed_pages=3))

    response = await litellm.aocr(model=MODEL, document=IMAGE_DOCUMENT, api_key="test-key")

    assert response.usage_info.pages_processed == 3


@pytest.mark.asyncio
async def test_aocr_usage_falls_back_to_page_count_without_meta(disable_aiohttp_transport, respx_mock):
    respx_mock.post(PARSE_URL).respond(json=_markdown_response(billed_pages=None))

    response = await litellm.aocr(model=MODEL, document=IMAGE_DOCUMENT, api_key="test-key")

    assert response.usage_info.pages_processed == 2


@pytest.mark.asyncio
async def test_aocr_blocks_output_format_forwards_param_and_keeps_blocks(disable_aiohttp_transport, respx_mock):
    route = respx_mock.post(PARSE_URL).respond(json=_blocks_response())

    response = await litellm.aocr(model=MODEL, document=IMAGE_DOCUMENT, api_key="test-key", output_format="blocks")

    assert json.loads(route.calls.last.request.content)["output_format"] == "blocks"
    assert response.pages[0].markdown == ""
    assert response.pages[0].model_extra["blocks"] == [{"type": "text", "text": "Total Due: $4.00"}]
    assert response.usage_info.pages_processed == 1


@pytest.mark.asyncio
async def test_aocr_native_format_carries_provider_payload(disable_aiohttp_transport, respx_mock):
    payload = _markdown_response()
    route = respx_mock.post(PARSE_URL).respond(json=payload)

    response = await litellm.aocr(model=MODEL, document=IMAGE_DOCUMENT, api_key="test-key", req_format="native")

    assert "req_format" not in json.loads(route.calls.last.request.content)
    assert response.get_provider_native_response() == payload
    assert response.pages[0].markdown == "# Receipt\n\nTotal Due: $4.00"


@pytest.mark.asyncio
async def test_aocr_rejects_unknown_output_format_before_calling_provider(disable_aiohttp_transport, respx_mock):
    route = respx_mock.post(PARSE_URL).respond(json=_markdown_response())

    with pytest.raises(litellm.BadRequestError, match="Invalid `output_format`: 'html'") as exc_info:
        await litellm.aocr(model=MODEL, document=IMAGE_DOCUMENT, api_key="test-key", output_format="html")

    assert exc_info.value.status_code == 400
    assert not route.called


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "document",
    [
        {"type": "document_url", "document_url": "https://example.com/doc.pdf"},
        {"type": "image_url", "image_url": "data:application/pdf;base64,JVBERi0="},
        {"type": "image_url", "image_url": ""},
    ],
)
async def test_aocr_rejects_non_image_documents_before_calling_provider(
    disable_aiohttp_transport, respx_mock, document
):
    route = respx_mock.post(PARSE_URL).respond(json=_markdown_response())

    with pytest.raises(litellm.BadRequestError, match="only accepts `image_url` documents") as exc_info:
        await litellm.aocr(model=MODEL, document=document, api_key="test-key")

    assert exc_info.value.status_code == 400
    assert not route.called


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "api_base, expected_url",
    [
        ("https://gateway.example.com", "https://gateway.example.com/v2/parse"),
        ("https://gateway.example.com/cohere/", "https://gateway.example.com/cohere/v2/parse"),
        ("https://gateway.example.com/v2", "https://gateway.example.com/v2/parse"),
        ("https://gateway.example.com/v2/parse", "https://gateway.example.com/v2/parse"),
    ],
)
async def test_aocr_posts_to_api_base_variants(disable_aiohttp_transport, respx_mock, api_base, expected_url):
    route = respx_mock.post(expected_url).respond(json=_markdown_response())

    await litellm.aocr(model=MODEL, document=IMAGE_DOCUMENT, api_key="test-key", api_base=api_base)

    assert route.called


@pytest.mark.asyncio
async def test_aocr_surfaces_provider_error_with_its_status_and_message(disable_aiohttp_transport, respx_mock):
    respx_mock.post(PARSE_URL).respond(
        status_code=400, json={"id": "83b0d95e", "message": "output_format must be `blocks` or `markdown`"}
    )

    with pytest.raises(litellm.BadRequestError, match="output_format must be") as exc_info:
        await litellm.aocr(model=MODEL, document=IMAGE_DOCUMENT, api_key="test-key")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_aocr_reads_api_key_from_environment(disable_aiohttp_transport, respx_mock, monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "env-key")
    route = respx_mock.post(PARSE_URL).respond(json=_markdown_response())

    await litellm.aocr(model=MODEL, document=IMAGE_DOCUMENT)

    assert route.calls.last.request.headers["Authorization"] == "Bearer env-key"


@pytest.mark.asyncio
async def test_aocr_without_api_key_names_the_env_var(disable_aiohttp_transport, respx_mock, monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "cohere_key", None)
    route = respx_mock.post(PARSE_URL).respond(json=_markdown_response())

    with pytest.raises(Exception, match="Missing COHERE_API_KEY"):
        await litellm.aocr(model=MODEL, document=IMAGE_DOCUMENT)

    assert not route.called


@pytest.mark.asyncio
async def test_ahealth_check_ocr_sends_an_image_cohere_parse_accepts(disable_aiohttp_transport, respx_mock):
    route = respx_mock.post(PARSE_URL).respond(json=_markdown_response())

    result = await litellm.ahealth_check(model_params={"model": MODEL, "api_key": "test-key"}, mode="ocr")

    document = json.loads(route.calls.last.request.content)["document"]
    assert document["type"] == "image_url"
    assert document["image_url"].startswith("data:image/png;base64,")
    assert "error" not in result
