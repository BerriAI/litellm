"""
Tests for the image_edit() function in litellm/images/main.py
"""

import io
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("../.."))

import litellm


def test_image_edit_fallback_forwards_non_default_params():
    """
    Regression test: non_default_params must reach the fallback image_edit_handler
    path (i.e. providers other than bedrock / stability / black_forest_labs).

    Before gh-30814 the fallback path was missing::

        image_edit_request_params.update(non_default_params)

    so provider-specific kwargs such as ``image_config`` were silently dropped
    before the call to base_llm_http_handler.image_edit_handler.
    """
    captured = {}

    def _capture_handler(**kwargs):
        captured.update(kwargs)
        return litellm.ImageResponse(data=[])

    mock_image = io.BytesIO(b"\x89PNGfake")
    mock_image.name = "test.png"

    mock_provider_config = MagicMock()
    mock_utils = MagicMock()
    mock_utils.get_requested_image_edit_optional_param.return_value = MagicMock()
    mock_utils.get_optional_params_image_edit.return_value = {}

    with (
        patch(
            "litellm.images.main.base_llm_http_handler.image_edit_handler",
            side_effect=_capture_handler,
        ),
        patch(
            "litellm.utils.ProviderConfigManager.get_provider_image_edit_config",
            return_value=mock_provider_config,
        ),
        patch(
            "litellm.images.main._get_ImageEditRequestUtils",
            return_value=mock_utils,
        ),
    ):
        litellm.image_edit(
            model="openrouter/gpt-image-1",
            image=mock_image,
            prompt="a cute sea otter",
            image_config={"style": "vivid"},
            api_key="fake-key",
        )

    forwarded = captured.get("image_edit_optional_request_params", {})
    assert "image_config" in forwarded, (
        f"Provider-specific kwargs not forwarded to fallback handler. "
        f"Got image_edit_optional_request_params={forwarded!r}. "
        f"Expected 'image_config' to be present (regression: gh-30814)."
    )
    assert forwarded["image_config"] == {"style": "vivid"}
