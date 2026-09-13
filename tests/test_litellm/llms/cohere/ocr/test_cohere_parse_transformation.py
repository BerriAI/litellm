from typing import Final
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.cohere.ocr.transformation import CohereParseConfig

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


@pytest.mark.parametrize("output_format", ["markdown", "blocks"])
def test_transform_cohere_request_filters_options(output_format: str) -> None:
    config: Final = CohereParseConfig()
    params: Final = config.map_ocr_params(
        {"output_format": output_format, "req_format": "native", "unknown": True}, {}, "parse-v5.0"
    )
    request: Final = config.transform_ocr_request("parse-v5.0", IMAGE_DOCUMENT, params, {})
    assert request.data == {"model": "parse-v5.0", "document": IMAGE_DOCUMENT, "output_format": output_format}


@pytest.mark.parametrize("native", [False, True])
def test_transform_cohere_response_keeps_images_and_native_payload(native: bool) -> None:
    payload: Final = _markdown_response(3)
    response: Final = CohereParseConfig().transform_ocr_response(
        "parse-v5.0", httpx.Response(200, json=payload), Mock(), {"req_format": "native" if native else "litellm"}
    )
    assert response.pages[0].markdown == "# Receipt\n\nTotal Due: $4.00"
    assert response.pages[0].images[0].bbox == BOUNDING_BOX
    assert response.pages[0].images[0].model_extra["description"] == "A parking receipt"
    assert response.pages[1].images is None
    assert response.usage_info.pages_processed == 3
    assert response.get_provider_native_response() == (payload if native else None)


def test_transform_cohere_blocks() -> None:
    response: Final = CohereParseConfig().transform_ocr_response(
        "parse-v5.0", httpx.Response(200, json=_blocks_response()), Mock()
    )
    assert response.pages[0].model_extra["blocks"] == [{"type": "text", "text": "Total Due: $4.00"}]
    assert response.pages[0].markdown == ""


def test_transform_cohere_rejects_unsupported_output_format() -> None:
    with pytest.raises(litellm.UnsupportedParamsError, match="output_format"):
        CohereParseConfig().map_ocr_params({"output_format": "html"}, {}, "parse-v5.0")
