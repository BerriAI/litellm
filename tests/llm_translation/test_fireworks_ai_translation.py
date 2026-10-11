import json
import pytest

import litellm
from litellm.litellm_core_utils.get_supported_openai_params import (
    get_supported_openai_params,
)
from litellm.llms.fireworks_ai.chat.transformation import FireworksAIConfig

fireworks = FireworksAIConfig()

VISION_MODEL = next(
    key.removeprefix("fireworks_ai/")
    for key, info in litellm.model_cost.items()
    if key.startswith("fireworks_ai/accounts/fireworks/models/") and info.get("supports_vision") is True
)


@pytest.mark.parametrize(
    "disable_add_transform_inline_image_block",
    [True, False],
)
def test_document_inlining_example(disable_add_transform_inline_image_block):
    """
    Fireworks document inlining has been removed from the platform. LiteLLM
    must not append ``#transform=inline`` regardless of the legacy disable flag.
    """
    from unittest.mock import patch

    from litellm import completion
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    pdf_url = "https://storage.googleapis.com/fireworks-public/test/sample_resume.pdf"

    with patch.object(client, "post") as mock_post:
        try:
            completion(
                model=f"fireworks_ai/{VISION_MODEL}",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": pdf_url},
                            },
                            {
                                "type": "text",
                                "text": "What are the candidate's BA and MBA GPAs?",
                            },
                        ],
                    }
                ],
                disable_add_transform_inline_image_block=disable_add_transform_inline_image_block,
                client=client,
            )
        except Exception:
            pass

        mock_post.assert_called_once()
        json_data = json.loads(mock_post.call_args.kwargs["data"])
        sent_url = json_data["messages"][0]["content"][0]["image_url"]["url"]
        assert sent_url == pdf_url
        assert "#transform=inline" not in sent_url


def test_global_disable_flag_with_transform_messages_helper(monkeypatch):
    from unittest.mock import patch
    from litellm import completion
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    monkeypatch.setattr(litellm, "disable_add_transform_inline_image_block", True)

    with patch.object(
        client,
        "post",
    ) as mock_post:
        try:
            completion(
                model=f"fireworks_ai/{VISION_MODEL}",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://awsmp-logos.s3.amazonaws.com/seller-xw5kijmvmzasy/c233c9ade2ccb5491072ae232c814942.png"
                                },
                            },
                        ],
                    }
                ],
                client=client,
            )
        except Exception:
            pass

        mock_post.assert_called_once()
        json_data = json.loads(mock_post.call_args.kwargs["data"])
        assert (
            "#transform=inline"
            not in json_data["messages"][0]["content"][1]["image_url"]["url"]
        )
