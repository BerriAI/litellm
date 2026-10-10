import json
from typing import Final

import pytest
import respx
from httpx import Response

import litellm

VISION_MODEL: Final = next(
    key.removeprefix("fireworks_ai/")
    for key, info in litellm.model_cost.items()
    if key.startswith("fireworks_ai/accounts/fireworks/models/") and info.get("supports_vision") is True
)
API_BASE: Final = "https://api.fireworks.ai/inference/v1"
COMPLETION: Final = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": VISION_MODEL,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@pytest.mark.parametrize("disable_add_transform_inline_image_block", [True, False])
@respx.mock
def test_document_inlining_example(disable_add_transform_inline_image_block) -> None:
    pdf_url: Final = "https://storage.googleapis.com/fireworks-public/test/sample_resume.pdf"
    request: Final = respx.post(f"{API_BASE}/chat/completions").mock(return_value=Response(200, json=COMPLETION))
    litellm.completion(
        model=f"fireworks_ai/{VISION_MODEL}",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": pdf_url}},
                    {"type": "text", "text": "What are the candidate's BA and MBA GPAs?"},
                ],
            }
        ],
        disable_add_transform_inline_image_block=disable_add_transform_inline_image_block,
        api_key="fireworks-test-key",
        api_base=API_BASE,
    )
    assert request.call_count == 1
    body: Final = json.loads(request.calls.last.request.content)
    sent_url: Final = body["messages"][0]["content"][0]["image_url"]["url"]
    assert sent_url == pdf_url


@respx.mock
def test_global_disable_flag_with_transform_messages_helper(monkeypatch) -> None:
    image_url: Final = "https://awsmp-logos.s3.amazonaws.com/seller-xw5kijmvmzasy/c233c9ade2ccb5491072ae232c814942.png"
    request: Final = respx.post(f"{API_BASE}/chat/completions").mock(return_value=Response(200, json=COMPLETION))
    monkeypatch.setattr(litellm, "disable_add_transform_inline_image_block", True)
    litellm.completion(
        model=f"fireworks_ai/{VISION_MODEL}",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        api_key="fireworks-test-key",
        api_base=API_BASE,
    )
    assert request.call_count == 1
    body: Final = json.loads(request.calls.last.request.content)
    sent_url: Final = body["messages"][0]["content"][1]["image_url"]["url"]
    assert sent_url == image_url
