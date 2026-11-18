"""
Tests for VertexAIImagenImageEditConfig.transform_image_edit_request
Verifies that missing image or prompt raise litellm.BadRequestError (HTTP 400)
instead of a raw ValueError, which would be mapped to a 500 by the proxy.

Related issue: https://github.com/BerriAI/litellm/issues/52366
Fixed in: litellm/llms/vertex_ai/image_edit/vertex_imagen_transformation.py
"""

import pytest
from unittest.mock import MagicMock

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
    """Validation path: missing required inputs must raise BadRequestError."""

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
        """Missing image must NOT raise a raw ValueError (which the proxy maps to HTTP 500)."""
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
            "Raw ValueError would be misclassified as APIConnectionError / 500 by the proxy"
        )

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
        assert not isinstance(exc_info.value, ValueError), (
            "Raw ValueError would be misclassified as APIConnectionError / 500 by the proxy"
        )

    def test_bad_request_error_llm_provider(self, config: VertexAIImagenImageEditConfig) -> None:
        """BadRequestError must carry llm_provider='vertex_ai' for correct error routing."""
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
