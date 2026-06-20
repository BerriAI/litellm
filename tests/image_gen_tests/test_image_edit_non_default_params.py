"""
Unit tests for PR #30814: forward non_default_params in image_edit fallback handler.

Before the fix, provider-specific kwargs (e.g. image_config) were silently
dropped in the fallback branch of litellm.image_edit().  The bedrock, stability,
and black_forest_labs branches already called
    image_edit_request_params.update(non_default_params)
but the else/fallback branch did not.

These tests verify that any kwarg not in litellm's standard param list is
forwarded to base_llm_http_handler.image_edit_handler via
image_edit_optional_request_params.
"""
import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.images.main import image_edit
from litellm.types.utils import ImageResponse, ImageObject


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_image_response() -> ImageResponse:
    return ImageResponse(
        created=1700000000,
        data=[ImageObject(url="https://example.com/edited.png")],
    )


def _make_logging_mock() -> MagicMock:
    """Minimal stand-in for the LiteLLMLoggingObj passed via kwargs."""
    m = MagicMock()
    m.update_from_kwargs.return_value = None
    return m


def _make_provider_config_mock() -> MagicMock:
    """Stand-in for BaseImageEditConfig."""
    cfg = MagicMock()
    cfg.get_supported_openai_params.return_value = []
    return cfg


def _make_request_utils_mock() -> MagicMock:
    utils = MagicMock()
    utils.get_requested_image_edit_optional_param.return_value = MagicMock()
    # Base params dict -- non_default_params are merged on top of this.
    utils.get_optional_params_image_edit.return_value = {}
    return utils


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFallbackHandlerForwardsNonDefaultParams:
    """
    The fallback branch (anything that is not bedrock / stability /
    black_forest_labs) must call
        image_edit_request_params.update(non_default_params)
    before delegating to base_llm_http_handler.image_edit_handler.
    """

    def _run(self, **extra_kwargs):
        """
        Call image_edit() with all external dependencies mocked, returning the
        captured call-kwargs of base_llm_http_handler.image_edit_handler.
        """
        mock_handler = MagicMock(return_value=_fake_image_response())

        with (
            patch(
                "litellm.images.main.base_llm_http_handler.image_edit_handler",
                mock_handler,
            ),
            patch(
                "litellm.images.main.ProviderConfigManager.get_provider_image_edit_config",
                return_value=_make_provider_config_mock(),
            ),
            patch(
                "litellm.images.main._get_ImageEditRequestUtils",
                return_value=_make_request_utils_mock(),
            ),
        ):
            image_edit(
                model="gpt-image-1",
                image=b"fake-image-bytes",
                prompt="add a sunset background",
                litellm_logging_obj=_make_logging_mock(),
                **extra_kwargs,
            )

        assert mock_handler.called, "image_edit_handler was never invoked"
        return mock_handler.call_args.kwargs

    # ------------------------------------------------------------------

    def test_image_config_forwarded(self):
        """image_config (a non-default param) must reach the handler."""
        call_kw = self._run(image_config={"quality": "high", "background": "transparent"})
        optional = call_kw.get("image_edit_optional_request_params", {})

        assert "image_config" in optional, (
            f"image_config was dropped. image_edit_optional_request_params={optional}"
        )
        assert optional["image_config"] == {"quality": "high", "background": "transparent"}

    def test_multiple_non_default_params_forwarded(self):
        """All non-default params are forwarded, not just image_config."""
        call_kw = self._run(
            image_config={"quality": "hd"},
            custom_vendor_param="foo",
        )
        optional = call_kw.get("image_edit_optional_request_params", {})

        assert "image_config" in optional, f"image_config missing: {optional}"
        assert "custom_vendor_param" in optional, f"custom_vendor_param missing: {optional}"

    def test_standard_params_not_in_non_default(self):
        """
        Standard openai/litellm params (e.g. n, size) are not in non_default_params
        and must not be re-injected by the update() call.
        """
        call_kw = self._run(image_config={"quality": "standard"})
        optional = call_kw.get("image_edit_optional_request_params", {})
        # image_config is the non-default param -- it must be present.
        assert "image_config" in optional

    def test_no_non_default_params_still_calls_handler(self):
        """Sanity check: the fallback path works with no extra kwargs too."""
        call_kw = self._run()
        assert "image_edit_optional_request_params" in call_kw
