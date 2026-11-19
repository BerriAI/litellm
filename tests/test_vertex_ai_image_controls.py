"""Regression tests for Vertex Gemini image controls that are rejected or carried.

1. Unmapped size or non-dict imageConfig is rejected when drop_params=False.
2. The edit adapter accepts and carries aspect_ratio, image_size, and imageConfig.
3. An edit mask is rejected when the model does not support it and drop_params=False.
4. Valid controls and aspect ratio mappings (e.g. 1792x1024 -> 16:9, 1536x1024 -> 3:2) are preserved.
"""

import json
import os
import socket
import sys
from unittest.mock import patch

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

try:
    import litellm.rust_bridge._native
except ImportError:
    class _Installed:
        fallback_generalizations = {}
        rows = {}
        model_count = 0

    class _Dummy:
        def __init__(self, *args, **kwargs): pass
        @classmethod
        def bundled(cls, **kwargs):
            class _Inputs:
                loaded = lambda self: {}
                install_bundled = lambda self: _Installed()
                class provenance:
                    revision = "bundled"
            return _Inputs()

    class _NativeModule:
        NativeModelCostMap = _Dummy
        NativeTokenRedactor = _Dummy
        def __getattr__(self, name):
            return _Dummy

    sys.modules["litellm.rust_bridge._native"] = _NativeModule()

import pytest

with patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")):
    import litellm
    from litellm.llms.vertex_ai.image_edit.vertex_gemini_transformation import (
        VertexAIGeminiImageEditConfig,
    )
    from litellm.llms.vertex_ai.image_generation.vertex_gemini_transformation import (
        VertexAIGeminiImageGenerationConfig,
    )
    from litellm.types.router import GenericLiteLLMParams

MODEL = "gemini-3.1-flash-image"
PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture(autouse=True)
def forbid_network():
    with patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")):
        yield


# 1. Invalid supported values must not be silently replaced when not dropping params.
@pytest.mark.parametrize(
    "route,params",
    [
        ("generation", {"size": "not-a-size"}),
        ("edit", {"size": "not-a-size"}),
        ("generation", {"imageConfig": "banana"}),
        ("edit", {"imageConfig": "banana"}),
    ],
    ids=["generation-unmapped-size", "edit-unmapped-size", "generation-non-dict-image-config", "edit-non-dict-image-config"],
)
def test_unmapped_or_invalid_control_rejected_when_not_dropping(route, params):
    with pytest.raises((ValueError, litellm.BadRequestError, litellm.UnsupportedParamsError)):
        if route == "generation":
            VertexAIGeminiImageGenerationConfig().map_openai_params(params, {}, MODEL, drop_params=False)
        else:
            VertexAIGeminiImageEditConfig().map_openai_params(params, MODEL, drop_params=False)


# 2. Edits should accept the image controls that generation accepts.
@pytest.mark.parametrize(
    "params,expected",
    [
        ({"aspect_ratio": "9:16"}, {"aspect_ratio": "9:16"}),
        ({"image_size": "2K"}, {"image_size": "2K"}),
        ({"imageConfig": {"aspectRatio": "9:16", "imageSize": "2K"}}, {"aspect_ratio": "9:16", "image_size": "2K"}),
    ],
    ids=["edit-aspect-ratio", "edit-image-size", "edit-image-config"],
)
def test_edit_request_carries_image_controls(params, expected):
    config = VertexAIGeminiImageEditConfig()
    assert set(params) <= set(config.get_supported_openai_params(MODEL)), (
        f"edit adapter does not support {sorted(set(params) - set(config.get_supported_openai_params(MODEL)))}"
    )
    mapped = config.map_openai_params(params, MODEL, drop_params=False)
    body, _ = config.transform_image_edit_request(
        model=MODEL,
        prompt="Make the sky a sunset",
        image=PNG,
        image_edit_optional_request_params=mapped,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    image_config = json.loads(body)["generationConfig"].get("image_config", {})
    for key, value in expected.items():
        assert image_config.get(key) == value


# 3. A mask must not be silently discarded.
def test_edit_mask_is_not_silently_dropped():
    config = VertexAIGeminiImageEditConfig()
    if "mask" in config.get_supported_openai_params(MODEL):
        return
    with pytest.raises((ValueError, litellm.BadRequestError, litellm.UnsupportedParamsError)):
        config.map_openai_params({"mask": PNG}, MODEL, drop_params=False)


# Controls that already work keep working.
@pytest.mark.parametrize("drop_params", [False, True])
def test_valid_controls_keep_existing_mapping(drop_params):
    generation = VertexAIGeminiImageGenerationConfig()
    edit = VertexAIGeminiImageEditConfig()
    assert generation.map_openai_params({"size": "1792x1024"}, {}, MODEL, drop_params)["aspectRatio"] == "16:9"
    assert generation.map_openai_params({"size": "1536x1024"}, {}, MODEL, drop_params)["aspectRatio"] == "3:2"
    image_config = {"aspectRatio": "16:9", "imageSize": "1K"}
    assert generation.map_openai_params({"imageConfig": image_config}, {}, MODEL, drop_params) == {"imageConfig": image_config}
    assert edit.map_openai_params({"size": "1792x1024"}, MODEL, drop_params)["aspectRatio"] == "16:9"
    assert edit.map_openai_params({"size": "1536x1024"}, MODEL, drop_params)["aspectRatio"] == "3:2"


def test_string_image_config_decoded_properly():
    generation = VertexAIGeminiImageGenerationConfig()
    edit = VertexAIGeminiImageEditConfig()
    cfg_str = json.dumps({"aspectRatio": "9:16", "imageSize": "2K"})
    gen_mapped = generation.map_openai_params({"imageConfig": cfg_str}, {}, MODEL, drop_params=False)
    assert gen_mapped["imageConfig"] == {"aspectRatio": "9:16", "imageSize": "2K"}

    edit_mapped = edit.map_openai_params({"imageConfig": cfg_str}, MODEL, drop_params=False)
    assert edit_mapped["imageConfig"] == {"aspectRatio": "9:16", "imageSize": "2K"}

    body, _ = edit.transform_image_edit_request(
        model=MODEL,
        prompt="Make it night",
        image=PNG,
        image_edit_optional_request_params=edit_mapped,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    image_config = json.loads(body)["generationConfig"].get("image_config", {})
    assert image_config.get("aspect_ratio") == "9:16"
    assert image_config.get("image_size") == "2K"


def test_edit_preserves_explicit_controls_over_size():
    edit = VertexAIGeminiImageEditConfig()
    # If size="1024x1024" (default 1:1) and aspect_ratio="9:16" are passed, explicit aspect_ratio wins
    mapped = edit.map_openai_params({"size": "1024x1024", "aspect_ratio": "9:16"}, MODEL, drop_params=False)
    assert mapped["aspectRatio"] == "9:16"


def test_edit_preserves_nested_image_config_over_size():
    edit = VertexAIGeminiImageEditConfig()
    mapped = edit.map_openai_params(
        {"size": "1024x1024", "imageConfig": {"aspectRatio": "9:16", "imageSize": "2K"}},
        MODEL,
        drop_params=False,
    )
    assert mapped.get("aspectRatio") is None
    body, _ = edit.transform_image_edit_request(
        model=MODEL,
        prompt="Make it night",
        image=PNG,
        image_edit_optional_request_params=mapped,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    image_config = json.loads(body)["generationConfig"].get("image_config", {})
    assert image_config.get("aspect_ratio") == "9:16"
    assert image_config.get("image_size") == "2K"


