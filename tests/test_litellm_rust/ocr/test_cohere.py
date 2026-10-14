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
async def test_public_cohere_health_check(recording_server: RecordingServer, model: str) -> None:
    recording_server.enqueue(ResponseSpec(body=PAYLOAD))
    response: Final = await litellm.ahealth_check(
        model_params={"model": model, "api_key": "test-key", "api_base": recording_server.base_url}, mode="ocr"
    )
    assert "error" not in response
    assert recording_server.requests[0].body["document"]["image_url"].startswith("data:image/png;base64,")
