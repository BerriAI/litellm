"""
OpenAI Image Generation Handler for Unified Guardrails

This module provides guardrail translation support for OpenAI's image generation endpoint.
The handler processes the 'prompt' parameter for guardrails.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, cast

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
    from litellm.utils import ImageResponse


class OpenAIImageGenerationHandler(BaseTranslation):
    """
    Handler for processing OpenAI image generation requests with guardrails.

    This class provides methods to:
    1. Process input prompt (pre-call hook)
    2. Process output response (post-call hook) - typically not needed for images

    The handler specifically processes the 'prompt' parameter which contains
    the text description for image generation.
    """

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> Mapping[str, object]:
        """
        Process input prompt by applying guardrails to text content.

        Args:
            data: Request data dictionary containing 'prompt' parameter
            guardrail_to_apply: The guardrail instance to apply

        Returns:
            Modified data with guardrails applied to prompt
        """
        prompt: Final = data.get("prompt")
        if prompt is None:
            verbose_proxy_logger.debug("OpenAI Image Generation: No prompt found in request data")
            return data

        # Apply guardrail to the prompt
        if isinstance(prompt, str):
            inputs: Final = GenericGuardrailAPIInputs(texts=[prompt])
            # Include model information if available
            model: Final = data.get("model")
            if model:
                inputs["model"] = model
            guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
                inputs=inputs,
                request_data=data,
                input_type="request",
                logging_obj=litellm_logging_obj,
            )
            guardrailed_texts: Final = guardrailed_inputs.get("texts", [])
            data["prompt"] = guardrailed_texts[0] if guardrailed_texts else prompt

            verbose_proxy_logger.debug(
                "OpenAI Image Generation: Applied guardrail to prompt. Original length: %d, New length: %d",
                len(prompt),
                len(data["prompt"]),
            )
        else:
            verbose_proxy_logger.debug(
                "OpenAI Image Generation: Unexpected prompt type: %s. Expected string.",
                type(prompt),
            )

        return data

    @staticmethod
    def _image_field(item: object, name: str) -> object:
        fields = cast("Mapping[str, object]", item) if isinstance(item, Mapping) else None
        if fields is not None:
            return fields.get(name)
        return cast(
            "object",
            getattr(item, name, None),  # pyright: ignore[reportUnknownArgumentType]  # typed image item
        )

    async def process_output_response(
        self,
        response: "ImageResponse",
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
        user_api_key_dict: "UserAPIKeyAuth | None" = None,
        request_data: dict | None = None,
    ) -> "ImageResponse":
        """Scan generated images with the guardrail before the response is returned.

        ``b64_json`` outputs are passed as base64 data URLs so the bytes the
        guardrail sees are exactly the bytes the caller receives. URL outputs
        are passed through as URLs and image guardrails that cannot guarantee
        the same bytes should reject them.
        """
        data_items: Final = cast("list[object]", getattr(response, "data", None) or [])
        images: Final[list[str]] = []
        for item in data_items:
            b64_json = self._image_field(item, "b64_json")
            url = self._image_field(item, "url")
            if isinstance(b64_json, str) and b64_json:
                images.append(b64_json if b64_json.startswith("data:") else f"data:image/png;base64,{b64_json}")
            elif isinstance(url, str) and url:
                images.append(url)

        if not images:
            verbose_proxy_logger.debug("OpenAI Image Generation: No images found in response for guardrail")
            return response

        inputs: Final = GenericGuardrailAPIInputs(images=images)
        model: Final = getattr(response, "model", None)
        if isinstance(model, str) and model:
            inputs["model"] = model
        await guardrail_to_apply.apply_guardrail(  # pyright: ignore[reportUnknownMemberType]  # guardrail hook is untyped
            inputs=inputs,
            request_data=request_data or {},
            input_type="response",
            logging_obj=litellm_logging_obj,
        )
        return response
