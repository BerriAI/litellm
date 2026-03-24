"""
Amazon Nova Canvas image edit on Bedrock (InvokeModel).

Maps OpenAI-style image edit (image + prompt, optional mask) to Nova Canvas task types:
- With mask: INPAINTING (inPaintingParams per AWS docs)
- Without mask: IMAGE_VARIATION (imageVariationParams)

Refs:
- https://docs.aws.amazon.com/nova/latest/userguide/image-gen-access.html
- https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import httpx

from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageObject, ImageResponse
from litellm.utils import (
    _get_model_cost_key,
    _get_potential_model_names,
    get_model_info,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


def _nova_canvas_task_body(
    *,
    image_b64: str,
    mask_b64: Optional[str],
    text: str,
    negative_text: Optional[str],
    similarity_strength: Optional[float],
    task_type: Optional[str],
    mask_prompt: Optional[str],
    out_painting_mode: Optional[str],
) -> Dict[str, Any]:
    """Build InvokeModel body task section (without imageGenerationConfig)."""
    if task_type == "BACKGROUND_REMOVAL":
        return {
            "taskType": "BACKGROUND_REMOVAL",
            "backgroundRemovalParams": {"image": image_b64},
        }
    if task_type == "OUTPAINTING":
        if mask_prompt is None and mask_b64 is None:
            raise ValueError(
                "OUTPAINTING requires either a mask image or a mask prompt. "
                "Pass mask=<file> or maskPrompt=<str> in the request."
            )
        out_params: Dict[str, Any] = {
            "image": image_b64,
            "text": text,
        }
        if mask_prompt is not None:
            out_params["maskPrompt"] = mask_prompt
        elif mask_b64 is not None:
            out_params["maskImage"] = mask_b64
        if negative_text is not None:
            out_params["negativeText"] = negative_text
        if out_painting_mode is not None:
            out_params["outPaintingMode"] = out_painting_mode
        return {
            "taskType": "OUTPAINTING",
            "outPaintingParams": out_params,
        }
    # Honour explicit IMAGE_VARIATION even when a mask is present (mask is ignored
    # for this task type; callers use INPAINTING when they want mask semantics).
    if task_type == "IMAGE_VARIATION":
        var_params_explicit: Dict[str, Any] = {
            "images": [image_b64],
            "text": text,
        }
        if negative_text is not None:
            var_params_explicit["negativeText"] = negative_text
        if similarity_strength is not None:
            var_params_explicit["similarityStrength"] = similarity_strength
        return {
            "taskType": "IMAGE_VARIATION",
            "imageVariationParams": var_params_explicit,
        }
    # Explicit taskType must be INPAINTING or omitted from here on; anything else is invalid.
    if task_type is not None and str(task_type).strip() != "":
        if task_type != "INPAINTING":
            raise ValueError(
                f"Unsupported Amazon Nova Canvas taskType: {task_type!r}. "
                "Use BACKGROUND_REMOVAL, OUTPAINTING, IMAGE_VARIATION, INPAINTING, "
                "or omit taskType for automatic routing (mask → INPAINTING, else IMAGE_VARIATION)."
            )
    if mask_b64 is not None or mask_prompt is not None or task_type == "INPAINTING":
        in_params: Dict[str, Any] = {"image": image_b64, "text": text}
        if mask_prompt is not None:
            in_params["maskPrompt"] = mask_prompt
        elif mask_b64 is not None:
            in_params["maskImage"] = mask_b64
        if negative_text is not None:
            in_params["negativeText"] = negative_text
        if "maskPrompt" not in in_params and "maskImage" not in in_params:
            raise ValueError(
                "Amazon Nova Canvas INPAINTING requires either maskPrompt or maskImage "
                "(use OpenAI mask= for maskImage, or pass maskPrompt in optional params). "
                "See https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html"
            )
        return {"taskType": "INPAINTING", "inPaintingParams": in_params}
    var_params: Dict[str, Any] = {
        "images": [image_b64],
        "text": text,
    }
    if negative_text is not None:
        var_params["negativeText"] = negative_text
    if similarity_strength is not None:
        var_params["similarityStrength"] = similarity_strength
    return {
        "taskType": "IMAGE_VARIATION",
        "imageVariationParams": var_params,
    }


def _file_types_to_b64(image: Optional[FileTypes]) -> str:
    """Encode OpenAI image input to base64 string for Nova Canvas."""
    if image is None:
        raise ValueError("Nova Canvas image edit requires an image input")
    if hasattr(image, "read") and callable(getattr(image, "read", None)):
        image_bytes = image.read()  # type: ignore[union-attr]
        return base64.b64encode(image_bytes).decode("utf-8")
    if isinstance(image, bytes):
        return base64.b64encode(image).decode("utf-8")
    if isinstance(image, str):
        return image
    if isinstance(image, tuple):
        raise ValueError(
            "Nova Canvas image edit does not support tuple FileTypes. "
            "Pass a file-like object, bytes, or a base64-encoded string."
        )
    return base64.b64encode(bytes(image)).decode("utf-8")  # type: ignore[arg-type]


def _supports_nova_canvas_image_edit_from_model_cost(model: str) -> bool:
    """
    True when model_cost marks the model for Nova Canvas image edit.

    Prefer supports_nova_canvas_image_edit (config-driven). If that key is absent on
    older installs, accept a registered Bedrock image_generation model whose catalog
    key is the known Nova Canvas id family (amazon.nova-canvas...).

    get_model_info / ModelInfoBase omit arbitrary JSON keys, so we read model_cost
    directly (same idea as supports_* bare_entry fallback).
    """
    import litellm as _litellm

    if not model:
        return False

    seen: set[str] = set()
    candidates: List[str] = []

    def _add(name: Optional[str]) -> None:
        if name and name not in seen:
            seen.add(name)
            candidates.append(name)

    _add(model)
    if "/" in model:
        suffix = model.split("/")[-1]
        _add(suffix)
        _add(f"bedrock/{suffix}")

    # Cross-region inference ids (e.g. us.amazon.nova-canvas-v1:0) share pricing with
    # the base model id (amazon.nova-canvas-v1:0) in model_cost.
    try:
        from litellm.llms.bedrock.common_utils import BedrockModelInfo

        base_model = BedrockModelInfo.get_base_model(model)
        if base_model and base_model != model:
            _add(base_model)
            _add(f"bedrock/{base_model}")
    except Exception:
        pass

    try:
        potential = _get_potential_model_names(model=model, custom_llm_provider=None)
        for field in (
            "combined_model_name",
            "combined_stripped_model_name",
            "stripped_model_name",
            "split_model",
        ):
            _add(potential[field])
    except Exception:
        pass

    for name in candidates:
        key = _get_model_cost_key(name)
        if key is None:
            continue
        entry = _litellm.model_cost.get(key) or {}
        if entry.get("supports_nova_canvas_image_edit") is True:
            return True
        if (
            entry.get("litellm_provider") == "bedrock"
            and entry.get("mode") == "image_generation"
            and "amazon.nova-canvas" in key.lower()
        ):
            return True
    return False


class BedrockAmazonNovaCanvasImageEditConfig(BaseImageEditConfig):
    """
    Bedrock InvokeModel image edit for amazon.nova-canvas-v1:0 and regional variants.
    """

    @classmethod
    def _is_nova_canvas_image_edit_model(cls, model: Optional[str] = None) -> bool:
        """
        Use model_cost.supports_nova_canvas_image_edit so new Nova Canvas inference IDs
        are added via model_prices_and_context_window.json only (not get_model_info, which
        drops keys not on ModelInfoBase).
        """
        return _supports_nova_canvas_image_edit_from_model_cost(model or "")

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "n",
            "size",
            "response_format",
            "mask",
            "negativeText",
            "similarityStrength",
            "cfgScale",
            "seed",
            "quality",
            "taskType",
            "maskPrompt",
            "outPaintingMode",
            "imageGenerationConfig",
        ]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict[str, Any]:
        supported = set(self.get_supported_openai_params(model))
        mapped: Dict[str, Any] = dict(image_edit_optional_params)
        _size = mapped.pop("size", None)
        if _size is not None and isinstance(_size, str) and "x" in _size:
            w, h = _size.split("x", 1)
            try:
                mapped["width"], mapped["height"] = int(w), int(h)
            except ValueError:
                pass
        _n = mapped.pop("n", None)
        if _n is not None:
            mapped["numberOfImages"] = _n
        _quality = mapped.pop("quality", None)
        if _quality is not None:
            if _quality in ("hd", "premium"):
                mapped["quality"] = "premium"
            elif _quality == "standard":
                mapped["quality"] = "standard"
        for k in ("response_format",):
            mapped.pop(k, None)
        # Drop unknown keys if drop_params
        if drop_params:
            for k in list(mapped.keys()):
                if k.startswith("_"):
                    continue
                if k not in supported and k not in (
                    "width",
                    "height",
                    "numberOfImages",
                    "mask",
                ):
                    mapped.pop(k, None)
        return mapped

    def transform_image_edit_request(
        self,
        model: str,
        prompt: Optional[str],
        image: Optional[FileTypes],
        image_edit_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, Any]:
        op = dict(image_edit_optional_request_params)
        image_b64 = _file_types_to_b64(image)

        mask_raw = op.pop("mask", None)
        mask_b64: Optional[str] = None
        if mask_raw is not None:
            mask_b64 = _file_types_to_b64(mask_raw)  # type: ignore[arg-type]

        _size = op.pop("size", None)
        width = op.pop("width", None)
        height = op.pop("height", None)
        if (
            width is None
            and height is None
            and _size is not None
            and isinstance(_size, str)
            and "x" in _size
        ):
            w, h = _size.split("x", 1)
            try:
                width, height = int(w), int(h)
            except ValueError:
                pass

        number_of_images = op.pop("numberOfImages", None)
        quality = op.pop("quality", None)
        cfg_scale = op.pop("cfgScale", None)
        seed = op.pop("seed", None)

        image_generation_config: Dict[str, Any] = {}
        nested_igc = op.pop("imageGenerationConfig", None)
        if isinstance(nested_igc, dict):
            image_generation_config.update(nested_igc)
        if width is not None:
            image_generation_config["width"] = width
        if height is not None:
            image_generation_config["height"] = height
        if number_of_images is not None:
            image_generation_config["numberOfImages"] = number_of_images
        if quality is not None:
            image_generation_config["quality"] = quality
        if cfg_scale is not None:
            image_generation_config["cfgScale"] = cfg_scale
        if seed is not None:
            image_generation_config["seed"] = seed

        text = prompt if prompt is not None and prompt != "" else " "
        negative_text = op.pop("negativeText", None)
        similarity_strength = op.pop("similarityStrength", None)
        task_type = op.pop("taskType", None)
        mask_prompt = op.pop("maskPrompt", None)
        out_painting_mode = op.pop("outPaintingMode", None)

        body = _nova_canvas_task_body(
            image_b64=image_b64,
            mask_b64=mask_b64,
            text=text,
            negative_text=negative_text,
            similarity_strength=similarity_strength,
            task_type=task_type,
            mask_prompt=mask_prompt,
            out_painting_mode=out_painting_mode,
        )

        # BACKGROUND_REMOVAL InvokeModel body must not include imageGenerationConfig (AWS rejects it).
        if image_generation_config and body.get("taskType") != "BACKGROUND_REMOVAL":
            body["imageGenerationConfig"] = image_generation_config

        return body, {}

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error parsing Nova Canvas image edit response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if raw_response.status_code not in (200,):
            raise self.get_error_class(
                error_message=f"Nova Canvas image edit error: {response_data}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if "errors" in response_data:
            raise self.get_error_class(
                error_message=f"Nova Canvas image edit error: {response_data['errors']}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        finish_reasons = response_data.get("finish_reasons", [])
        if finish_reasons and finish_reasons[0]:
            raise self.get_error_class(
                error_message=f"Nova Canvas image edit error: {finish_reasons[0]}",
                status_code=400,
                headers=raw_response.headers,
            )

        error_msg = response_data.get("message") or response_data.get("error")
        if error_msg:
            if not isinstance(error_msg, str):
                error_msg = str(error_msg)
            raise self.get_error_class(
                error_message=f"Nova Canvas image edit error: {error_msg}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        model_response = ImageResponse()
        model_response.data = []
        images: List[str] = response_data.get("images") or []
        for image_b64 in images:
            if image_b64:
                model_response.data.append(
                    ImageObject(
                        b64_json=image_b64,
                        url=None,
                        revised_prompt=None,
                    )
                )

        if not hasattr(model_response, "_hidden_params"):
            model_response._hidden_params = {}
        if "additional_headers" not in model_response._hidden_params:
            model_response._hidden_params["additional_headers"] = {}

        try:
            model_info = get_model_info(model, custom_llm_provider="bedrock")
            cost_per_image = model_info.get("output_cost_per_image", 0)
            if cost_per_image is not None and model_response.data:
                model_response._hidden_params["additional_headers"][
                    "llm_provider-x-litellm-response-cost"
                ] = float(cost_per_image) * len(model_response.data)
        except Exception:
            pass

        return model_response

    def use_multipart_form_data(self) -> bool:
        return False

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        raise NotImplementedError(
            "Nova Canvas image edit URLs are built in BedrockImageEdit._prepare_request "
            "(AWS runtime endpoint + model invoke path). Do not use get_complete_url for "
            "this config."
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        if headers is None:
            headers = {}
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return headers


def get_bedrock_image_edit_config_for_model(
    model: str,
) -> BaseImageEditConfig:
    """
    Return the correct Bedrock image-edit config for the model id.

    Same routing as ``BedrockImageEdit.get_config_class``: Stability edit models,
    Nova Canvas when marked in model_cost, otherwise **Stability config** (legacy
    compatibility; a warning is logged — prefer explicit model map entries).
    """
    from litellm._logging import verbose_logger
    from litellm.llms.bedrock.image_edit.stability_transformation import (
        BedrockStabilityImageEditConfig,
    )

    if BedrockStabilityImageEditConfig._is_stability_edit_model(model):
        return BedrockStabilityImageEditConfig()
    if BedrockAmazonNovaCanvasImageEditConfig._is_nova_canvas_image_edit_model(model):
        return BedrockAmazonNovaCanvasImageEditConfig()
    verbose_logger.warning(
        "Bedrock image edit: model %r is not a known Stability edit model and is not "
        "marked for Nova Canvas; using Stability image-edit config for compatibility. "
        "Add supports_nova_canvas_image_edit or use a stability.* edit model id if this "
        "is incorrect.",
        model,
    )
    return BedrockStabilityImageEditConfig()
