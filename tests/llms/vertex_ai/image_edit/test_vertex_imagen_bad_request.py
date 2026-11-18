"""
Tests for VertexAIImagenImageEditConfig.transform_image_edit_request
Verifies that missing image or prompt raise litellm.BadRequestError (HTTP 400)
instead of a raw ValueError, which would be mapped to a 500 by the proxy.

There are three raise sites in transform_image_edit_request():
  1. image is None                           → BadRequestError
  2. reference_images is empty after prep    → BadRequestError
  3. prompt is None                          → BadRequestError

Related issue: https://github.com/BerriAI/litellm/issues/52366
Fixed in: litellm/llms/vertex_ai/image_edit/vertex_imagen_transformation.py
"""

import pytest
from unittest.mock import MagicMock, patch

import litellm
from litellm.llms.vertex_ai.image_edit.vertex_imagen_transformation import (
    VertexAIImagenImageEditConfig,
)


MODEL = "vertex_ai/imagen-3.0-capability-001"
FAKE_IMAGE = b"fake_image_bytes"
FAKE_PROMPT = "turn the background into a watercolor painting"


@pytest.fixture()
def config() -> VertexAIImagenImageEditConfig:
    return VertexAIImagenImageEditConfig()


class TestTransformImageEditRequestValidation:
    """Validation path: missing required inputs must raise BadRequestError (HTTP 400)."""

    # ------------------------------------------------------------------ #
    # Raise site 1: image is None                                         #
    # ------------------------------------------------------------------ #

    def test_image_none_raises_bad_request_error(self, config: VertexAIImagenImageEditConfig) -> None:
        """When image=None, transform_image_edit_request must raise BadRequestError (HTTP 400)."""
        with pytest.raises(litellm.BadRequestError) as exc_info:
            config.transform_image_edit_request(
                model=MODEL,
                prompt=FAKE_PROMPT,
                image=None,
                image_edit_optional_request_params={},
                litellm_params=MagicMock(),
                headers={},
            )
        assert exc_info.value.status_code == 400, (
            "Missing image must produce HTTP 400, not 500"
        )
        assert "reference image" in str(exc_info.value).lower()

    def test_image_none_not_value_error(self, config: VertexAIImagenImageEditConfig) -> None:
        """Missing image must NOT raise a raw ValueError (proxy maps that to APIConnectionError / HTTP 500)."""
        with pytest.raises(Exception) as exc_info:
            config.transform_image_edit_request(
                model=MODEL,
                prompt=FAKE_PROMPT,
                image=None,
                image_edit_optional_request_params={},
                litellm_params=MagicMock(),
                headers={},
            )
        assert not isinstance(exc_info.value, ValueError), (
            "Raw ValueError is caught by exception_mapping and re-raised as "
            "APIConnectionError, which the proxy maps to HTTP 500"
        )

    def test_image_none_bad_request_error_llm_provider(self, config: VertexAIImagenImageEditConfig) -> None:
        """BadRequestError for missing image must carry llm_provider='vertex_ai'."""
        with pytest.raises(litellm.BadRequestError) as exc_info:
            config.transform_image_edit_request(
                model=MODEL,
                prompt=FAKE_PROMPT,
                image=None,
                image_edit_optional_request_params={},
                litellm_params=MagicMock(),
                headers={},
            )
        assert exc_info.value.llm_provider == "vertex_ai"

    # ------------------------------------------------------------------ #
    # Raise site 2: image resolves to an empty reference list             #
    # ------------------------------------------------------------------ #

    def test_empty_reference_images_raises_bad_request_error(
        self, config: VertexAIImagenImageEditConfig
    ) -> None:
        """When _prepare_reference_images returns [], must raise BadRequestError (HTTP 400)."""
        with patch.object(config, "_prepare_reference_images", return_value=[]):
            with pytest.raises(litellm.BadRequestError) as exc_info:
                config.transform_image_edit_request(
                    model=MODEL,
                    prompt=FAKE_PROMPT,
                    image=FAKE_IMAGE,
                    image_edit_optional_request_params={},
                    litellm_params=MagicMock(),
                    headers={},
                )
        assert exc_info.value.status_code == 400
        assert "reference image" in str(exc_info.value).lower()

    def test_empty_reference_images_not_value_error(
        self, config: VertexAIImagenImageEditConfig
    ) -> None:
        """Empty reference image list must NOT raise a raw ValueError."""
        with patch.object(config, "_prepare_reference_images", return_value=[]):
            with pytest.raises(Exception) as exc_info:
                config.transform_image_edit_request(
                    model=MODEL,
                    prompt=FAKE_PROMPT,
                    image=FAKE_IMAGE,
                    image_edit_optional_request_params={},
                    litellm_params=MagicMock(),
                    headers={},
                )
        assert not isinstance(exc_info.value, ValueError)

    # ------------------------------------------------------------------ #
    # Raise site 3: prompt is None                                        #
    # ------------------------------------------------------------------ #

    def test_prompt_none_raises_bad_request_error(self, config: VertexAIImagenImageEditConfig) -> None:
        """When prompt=None, transform_image_edit_request must raise BadRequestError (HTTP 400)."""
        with pytest.raises(litellm.BadRequestError) as exc_info:
            config.transform_image_edit_request(
                model=MODEL,
                prompt=None,
                image=FAKE_IMAGE,
                image_edit_optional_request_params={},
                litellm_params=MagicMock(),
                headers={},
            )
        assert exc_info.value.status_code == 400, (
            "Missing prompt must produce HTTP 400, not 500"
        )
        assert "prompt" in str(exc_info.value).lower()

    def test_prompt_none_not_value_error(self, config: VertexAIImagenImageEditConfig) -> None:
        """Missing prompt must NOT raise a raw ValueError."""
        with pytest.raises(Exception) as exc_info:
            config.transform_image_edit_request(
                model=MODEL,
                prompt=None,
                image=FAKE_IMAGE,
                image_edit_optional_request_params={},
                litellm_params=MagicMock(),
                headers={},
            )
        assert not isinstance(exc_info.value, ValueError)

    def test_prompt_none_bad_request_error_llm_provider(self, config: VertexAIImagenImageEditConfig) -> None:
        """BadRequestError for missing prompt must carry llm_provider='vertex_ai'."""
        with pytest.raises(litellm.BadRequestError) as exc_info:
            config.transform_image_edit_request(
                model=MODEL,
                prompt=None,
                image=FAKE_IMAGE,
                image_edit_optional_request_params={},
                litellm_params=MagicMock(),
                headers={},
            )
        assert exc_info.value.llm_provider == "vertex_ai"
