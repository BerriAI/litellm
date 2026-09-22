import io
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, Final, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

import litellm
from litellm.images.utils import (
    ImageEditRequestUtils,
    measure_reference_image,
    measure_reference_pixels,
)
from litellm.litellm_core_utils.litellm_logging import use_custom_pricing_for_model
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.types.images.main import ImageEditOptionalRequestParams

CAT_JPEG: Final = Path(__file__).parents[2] / "e2e" / "llm_translation" / "fixtures" / "cat.jpg"
JPEG_SOI: Final = b"\xff\xd8"
JPEG_APP1: Final = b"\xff\xe1"
JPEG_APP2: Final = b"\xff\xe2"
JPEG_SOF0: Final = b"\xff\xc0"
JPEG_MARKER_AND_LENGTH_SIZE: Final = 4


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload))


def _png(width: int, height: int) -> bytes:
    header: Final = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    rows: Final = b"".join(b"\x00" + bytes(width) for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(rows))
        + _png_chunk(b"IEND", b"")
    )


def _pillow_bytes(width: int, height: int, mode: str, image_format: str, **save_options: object) -> bytes:
    buffer: Final = io.BytesIO()
    Image.new(mode, (width, height)).save(buffer, format=image_format, **save_options)
    return buffer.getvalue()


def _jpeg_app_segment(marker: bytes, payload_size: int) -> bytes:
    return marker + struct.pack(">H", payload_size + 2) + bytes(payload_size)


def _jpeg_with_leading_segments(jpeg: bytes, *segments: bytes) -> bytes:
    return JPEG_SOI + b"".join(segments) + jpeg[len(JPEG_SOI) :]


class _NonSeekableStream(io.BytesIO):
    def seekable(self) -> bool:
        return False


@pytest.mark.parametrize("initial_position", (0, 3))
def test_measure_reference_image_reads_png_jpeg_webp_headers_and_keeps_position(initial_position: int):
    pillow_jpeg: Final = _pillow_bytes(1024, 768, "RGB", "JPEG")
    large_app1_jpeg: Final = _jpeg_with_leading_segments(pillow_jpeg, _jpeg_app_segment(JPEG_APP1, 60_000))
    far_sof_jpeg: Final = _jpeg_with_leading_segments(
        pillow_jpeg, _jpeg_app_segment(JPEG_APP2, 40_000), _jpeg_app_segment(JPEG_APP2, 40_000)
    )
    assert far_sof_jpeg.index(JPEG_SOF0) > 70_000
    cases: Final = (
        (_png(1024, 1024), (1024, 1024)),
        (CAT_JPEG.read_bytes(), (512, 512)),
        (_pillow_bytes(640, 480, "RGB", "WEBP"), (640, 480)),
        (_pillow_bytes(640, 480, "RGB", "WEBP", lossless=True), (640, 480)),
        (_pillow_bytes(640, 480, "RGBA", "WEBP"), (640, 480)),
        (large_app1_jpeg, (1024, 768)),
        (far_sof_jpeg, (1024, 768)),
    )
    layouts: Final = tuple(image[12:16] for image, _ in cases[2:5])
    assert layouts == (b"VP8 ", b"VP8L", b"VP8X")

    for image, expected in cases:
        stream: Final = io.BytesIO(bytes(initial_position) + image)
        stream.seek(initial_position)
        assert measure_reference_image(stream) == expected, image[:16]
        assert stream.tell() == initial_position, image[:16]


def test_measure_reference_image_returns_none_on_truncated_jpeg_and_keeps_position():
    jpeg: Final = _pillow_bytes(1024, 768, "RGB", "JPEG")
    sof_offset: Final = jpeg.index(JPEG_SOF0)
    prefixes: Final = (32, 160, sof_offset, sof_offset + JPEG_MARKER_AND_LENGTH_SIZE)

    for prefix in prefixes:
        stream: Final = io.BytesIO(jpeg[:prefix])
        assert measure_reference_image(stream) is None, prefix
        assert stream.tell() == 0, prefix


def test_measure_reference_image_returns_none_for_unreadable_bytes_paths_tuples_and_non_seekable_streams(
    tmp_path: Path,
):
    png: Final = _png(64, 64)
    path: Final = tmp_path / "ref.png"
    path.write_bytes(png)
    non_seekable: Final = _NonSeekableStream(png)

    assert measure_reference_image(io.BytesIO(b"image")) is None
    assert measure_reference_image(b"GIF89a" + bytes(26)) is None
    assert measure_reference_image(path) is None
    assert measure_reference_image(("ref.png", path, "image/png", {})) is None
    assert measure_reference_image(non_seekable) is None
    assert non_seekable.tell() == 0
    assert measure_reference_image(("ref.png", png)) == (64, 64)
    assert measure_reference_image(("ref.png", io.BytesIO(png), "image/png")) == (64, 64)


def test_measure_reference_pixels_returns_none_when_any_reference_is_unmeasurable():
    assert measure_reference_pixels([_png(64, 64), CAT_JPEG.read_bytes()]) == 64 * 64 + 512 * 512
    assert measure_reference_pixels([_png(64, 64), io.BytesIO(b"image"), CAT_JPEG.read_bytes()]) is None
    assert measure_reference_pixels([]) == 0


class MockImageEditConfig(BaseImageEditConfig):
    def get_supported_openai_params(self, model: str) -> List[str]:
        return ["size", "quality"]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict[str, Any]:
        return dict(image_edit_optional_params)

    def get_complete_url(self, model: str, api_base: str, litellm_params: dict) -> str:
        return "https://example.com/api"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[dict] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return headers

    def transform_image_edit_request(self, *args, **kwargs):
        return {}, []

    def transform_image_edit_response(self, *args, **kwargs):
        return MagicMock()


class TestImageEditRequestUtilsDropParams:
    def setup_method(self):
        self.config = MockImageEditConfig()
        self.model = "test-model"
        self._original_drop_params = getattr(litellm, "drop_params", None)

    def teardown_method(self):
        if self._original_drop_params is None:
            if hasattr(litellm, "drop_params"):
                delattr(litellm, "drop_params")
        else:
            litellm.drop_params = self._original_drop_params

    def test_unsupported_params_raises_without_drop(self):
        litellm.drop_params = False
        optional_params: ImageEditOptionalRequestParams = {
            "size": "1024x1024",
            "unsupported_param": "value",
        }

        with pytest.raises(litellm.UnsupportedParamsError) as exc_info:
            ImageEditRequestUtils.get_optional_params_image_edit(
                model=self.model,
                image_edit_provider_config=self.config,
                image_edit_optional_params=optional_params,
            )

        assert "unsupported_param" in str(exc_info.value)

    def test_drop_params_global_setting(self):
        litellm.drop_params = True
        optional_params: ImageEditOptionalRequestParams = {
            "size": "1024x1024",
            "unsupported_param": "value",
        }

        result = ImageEditRequestUtils.get_optional_params_image_edit(
            model=self.model,
            image_edit_provider_config=self.config,
            image_edit_optional_params=optional_params,
        )

        assert "size" in result
        assert "unsupported_param" not in result

    def test_drop_params_explicit_parameter(self):
        litellm.drop_params = False
        optional_params: ImageEditOptionalRequestParams = {
            "size": "1024x1024",
            "unsupported_param": "value",
        }

        result = ImageEditRequestUtils.get_optional_params_image_edit(
            model=self.model,
            image_edit_provider_config=self.config,
            image_edit_optional_params=optional_params,
            drop_params=True,
        )

        assert "size" in result
        assert "unsupported_param" not in result

    def test_additional_drop_params(self):
        litellm.drop_params = False
        optional_params: ImageEditOptionalRequestParams = {
            "size": "1024x1024",
            "quality": "high",
        }

        result = ImageEditRequestUtils.get_optional_params_image_edit(
            model=self.model,
            image_edit_provider_config=self.config,
            image_edit_optional_params=optional_params,
            additional_drop_params=["quality"],
        )

        assert "size" in result
        assert "quality" not in result

    def test_drop_params_false_with_global_true(self):
        litellm.drop_params = True
        optional_params: ImageEditOptionalRequestParams = {
            "size": "1024x1024",
            "unsupported_param": "value",
        }

        result = ImageEditRequestUtils.get_optional_params_image_edit(
            model=self.model,
            image_edit_provider_config=self.config,
            image_edit_optional_params=optional_params,
            drop_params=False,
        )

        assert "size" in result
        assert "unsupported_param" not in result

    def test_supported_params_pass_through(self):
        litellm.drop_params = False
        optional_params: ImageEditOptionalRequestParams = {
            "size": "1024x1024",
            "quality": "high",
        }

        result = ImageEditRequestUtils.get_optional_params_image_edit(
            model=self.model,
            image_edit_provider_config=self.config,
            image_edit_optional_params=optional_params,
        )

        assert result["size"] == "1024x1024"
        assert result["quality"] == "high"

    def test_additional_drop_params_with_unsupported_and_drop_true(self):
        litellm.drop_params = True
        optional_params: ImageEditOptionalRequestParams = {
            "size": "1024x1024",
            "quality": "high",
            "unsupported_param": "value",
        }

        result = ImageEditRequestUtils.get_optional_params_image_edit(
            model=self.model,
            image_edit_provider_config=self.config,
            image_edit_optional_params=optional_params,
            additional_drop_params=["quality"],
        )

        assert "size" in result
        assert "quality" not in result
        assert "unsupported_param" not in result


class TestImageEditCustomPricing:
    """
    Regression tests for https://github.com/BerriAI/litellm/issues/22244

    image_edit must forward model_info and metadata into litellm_params
    when calling update_environment_variables, so that custom pricing
    detection works after PR #20679 stripped custom pricing fields from
    the shared backend model key.
    """

    def test_image_edit_passes_model_info_to_logging(self):
        """
        When the router provides model_info with custom pricing fields,
        image_edit should include model_info and metadata in litellm_params.
        """
        from litellm.images.main import image_edit

        custom_model_info = {
            "id": "test-deployment-id",
            "input_cost_per_image": 0.00676128,
            "mode": "image_generation",
        }
        custom_metadata = {
            "model_info": custom_model_info,
        }

        captured_litellm_params = {}

        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}

        original_update = mock_logging_obj.update_from_kwargs

        def capturing_update(**update_kwargs):
            captured_litellm_params.update(update_kwargs.get("litellm_params", {}))
            inner_kwargs = update_kwargs.get("kwargs", {})
            if "metadata" in inner_kwargs:
                captured_litellm_params["metadata"] = inner_kwargs["metadata"]
            return original_update(**update_kwargs)

        mock_logging_obj.update_from_kwargs = capturing_update

        with (
            patch(
                "litellm.images.main.get_llm_provider",
                return_value=("test-model", "openai", None, None),
            ),
            patch(
                "litellm.images.main.ProviderConfigManager.get_provider_image_edit_config",
                return_value=MagicMock(),
            ),
            patch(
                "litellm.images.main._get_ImageEditRequestUtils",
                return_value=MagicMock(
                    get_requested_image_edit_optional_param=MagicMock(return_value={}),
                    get_optional_params_image_edit=MagicMock(return_value={}),
                ),
            ),
            patch("litellm.images.main.base_llm_http_handler") as mock_handler,
        ):
            mock_handler.image_edit_handler.return_value = MagicMock()

            try:
                image_edit(
                    image=b"fake-image-data",
                    prompt="test prompt",
                    model="openai/test-model",
                    litellm_logging_obj=mock_logging_obj,
                    model_info=custom_model_info,
                    metadata=custom_metadata,
                )
            except Exception:
                pass

        assert "model_info" in captured_litellm_params
        assert captured_litellm_params["model_info"] == custom_model_info
        assert "metadata" in captured_litellm_params
        assert captured_litellm_params["metadata"] == custom_metadata

    def test_custom_pricing_detected_from_model_info_in_metadata(self):
        litellm_params = {
            "metadata": {
                "model_info": {
                    "id": "deployment-id",
                    "input_cost_per_image": 0.00676128,
                },
            },
        }
        assert use_custom_pricing_for_model(litellm_params) is True

    def test_custom_pricing_not_detected_without_model_info(self):
        litellm_params = {"litellm_call_id": "test-call-id"}
        assert use_custom_pricing_for_model(litellm_params) is False


class TestImageEditHandlerCredentialsForwarding:
    """
    Regression tests for Vertex AI image_edit credentials bug.

    image_edit handler must forward litellm_params to validate_environment,
    so that credentials passed via YAML config (vertex_ai_project,
    vertex_ai_credentials, etc.) reach the auth layer instead of falling
    through to Application Default Credentials.
    """

    def test_vertex_gemini_image_edit_reads_credentials_from_litellm_params(self):
        """
        VertexAIGeminiImageEditConfig.validate_environment should read
        vertex_ai_project/vertex_ai_credentials from litellm_params first.
        """
        from litellm.llms.vertex_ai.image_edit.vertex_gemini_transformation import (
            VertexAIGeminiImageEditConfig,
        )

        config = VertexAIGeminiImageEditConfig()

        litellm_params = {
            "vertex_ai_project": "test-project-from-params",
            "vertex_ai_credentials": "/path/to/creds.json",
        }

        with patch.object(
            config, "_ensure_access_token", return_value=("token", "project")
        ) as mock_ensure:
            config.validate_environment(
                headers={},
                model="test-model",
                litellm_params=litellm_params,
            )

            mock_ensure.assert_called_once()
            call_kwargs = mock_ensure.call_args[1]

            assert call_kwargs["credentials"] == "/path/to/creds.json"
            assert call_kwargs["project_id"] == "test-project-from-params"

    def test_vertex_imagen_image_edit_reads_credentials_from_litellm_params(self):
        """
        VertexAIImagenImageEditConfig.validate_environment should read
        vertex_ai_project/vertex_ai_credentials from litellm_params first.
        """
        from litellm.llms.vertex_ai.image_edit.vertex_imagen_transformation import (
            VertexAIImagenImageEditConfig,
        )

        config = VertexAIImagenImageEditConfig()

        litellm_params = {
            "vertex_ai_project": "test-project-from-params",
            "vertex_ai_credentials": "/path/to/creds.json",
        }

        with patch.object(
            config, "_ensure_access_token", return_value=("token", "project")
        ) as mock_ensure:
            config.validate_environment(
                headers={},
                model="test-model",
                litellm_params=litellm_params,
            )

            mock_ensure.assert_called_once()
            call_kwargs = mock_ensure.call_args[1]

            assert call_kwargs["credentials"] == "/path/to/creds.json"
            assert call_kwargs["project_id"] == "test-project-from-params"

    def test_vertex_imagen_get_complete_url_reads_project_and_location_from_litellm_params(
        self,
    ):
        """
        VertexAIImagenImageEditConfig.get_complete_url should read
        vertex_ai_project and vertex_ai_location from litellm_params,
        not only from env vars / global settings.
        """
        from litellm.llms.vertex_ai.image_edit.vertex_imagen_transformation import (
            VertexAIImagenImageEditConfig,
        )

        config = VertexAIImagenImageEditConfig()

        litellm_params = {
            "vertex_ai_project": "param-project",
            "vertex_ai_location": "us-east1",
        }

        url = config.get_complete_url(
            model="vertex_ai/imagegeneration@002",
            api_base=None,
            litellm_params=litellm_params,
        )

        assert "param-project" in url
        assert "us-east1" in url

    def test_validate_environment_signature_includes_litellm_params(self):
        """
        All image_edit config validate_environment methods should accept
        litellm_params to allow credentials to be forwarded from the handler.
        """
        import inspect

        from litellm.llms.vertex_ai.image_edit.vertex_gemini_transformation import (
            VertexAIGeminiImageEditConfig,
        )
        from litellm.llms.vertex_ai.image_edit.vertex_imagen_transformation import (
            VertexAIImagenImageEditConfig,
        )
        from litellm.llms.openai.image_edit.transformation import (
            OpenAIImageEditConfig,
        )

        configs = [
            VertexAIGeminiImageEditConfig(),
            VertexAIImagenImageEditConfig(),
            OpenAIImageEditConfig(),
            MockImageEditConfig(),
        ]

        for config in configs:
            sig = inspect.signature(config.validate_environment)
            params = list(sig.parameters.keys())

            assert "litellm_params" in params, (
                f"{config.__class__.__name__}.validate_environment "
                "missing litellm_params parameter"
            )
            assert "api_base" in params, (
                f"{config.__class__.__name__}.validate_environment "
                "missing api_base parameter"
            )
