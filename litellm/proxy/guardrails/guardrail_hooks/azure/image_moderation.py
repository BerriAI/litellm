import base64
import binascii
from typing import TYPE_CHECKING, ClassVar, Final, Literal, cast

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,  # pyright: ignore[reportUnknownVariableType]  # decorator is untyped
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

from .base import (
    AZURE_CONTENT_SAFETY_MAX_IMAGE_BYTES,
    AzureGuardrailBase,
    check_categories_against_thresholds,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.azure.azure_image_moderation import (
        AzureContentSafetyImageModerationConfigModel,
        AzureImageModerationGuardrailResponse,
        AzureImageModerationImage,
    )

GUARDRAIL_LABEL: Final = "Azure Content Safety Guardrail"


class AzureContentSafetyImageModerationGuardrail(AzureGuardrailBase, CustomGuardrail):
    """LiteLLM guardrail for Azure Content Safety image analysis.

    Sends request and response images to the Azure ``image:analyze`` endpoint and
    applies the same per-category severity thresholds as the text moderation
    guardrail.
    """

    use_native_lifecycle_hooks: ClassVar[bool] = False

    default_severity_threshold: int = 2

    def __init__(
        self,
        guardrail_name: str,
        api_key: object,
        api_base: object,
        severity_threshold: object = None,
        severity_threshold_by_category: object = None,
        categories: object = None,
        **kwargs: object,
    ):
        from litellm.types.proxy.guardrails.guardrail_hooks.azure.azure_image_moderation import (
            AZURE_CONTENT_SAFETY_IMAGE_CATEGORIES,
        )

        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))
        super().__init__(
            api_key=cast("str", api_key),
            api_base=cast("str", api_base),
            guardrail_name=guardrail_name,
            **kwargs,
        )

        raw_severity_threshold: Final = cast("int | float | str | None", severity_threshold)
        self.severity_threshold = int(raw_severity_threshold) if raw_severity_threshold else None
        self.severity_threshold_by_category = cast("dict[str, int] | None", severity_threshold_by_category)
        configured_categories: Final = cast("list[object]", categories) if isinstance(categories, list) else None
        self.categories = (
            [str(category) for category in configured_categories]
            if configured_categories
            else list(AZURE_CONTENT_SAFETY_IMAGE_CATEGORIES)
        )

        verbose_proxy_logger.info("Initialized Azure Image Moderation Guardrail: %s", guardrail_name)

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ]

    @staticmethod
    def get_config_model() -> type["AzureContentSafetyImageModerationConfigModel"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.azure.azure_image_moderation import (
            AzureContentSafetyImageModerationConfigModel,
        )

        return AzureContentSafetyImageModerationConfigModel

    def _to_azure_image(self, image: str) -> "AzureImageModerationImage":
        if image.startswith("data:"):
            header, separator, payload = image.partition(",")
            if not separator or ";base64" not in header:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": f"{GUARDRAIL_LABEL}: image must be a base64 data URL, got {header[:32]}",
                    },
                )
            stripped_payload: Final = payload.strip()
            try:
                decoded: Final = base64.b64decode(stripped_payload, validate=True)
            except (binascii.Error, ValueError) as e:
                raise HTTPException(
                    status_code=400,
                    detail={"error": f"{GUARDRAIL_LABEL}: image data URL is not valid base64"},
                ) from e
            if len(decoded) > AZURE_CONTENT_SAFETY_MAX_IMAGE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "error": (
                            f"{GUARDRAIL_LABEL}: image is {len(decoded)} bytes, "
                            f"the limit is {AZURE_CONTENT_SAFETY_MAX_IMAGE_BYTES} bytes"
                        ),
                    },
                )
            return {"content": stripped_payload}
        if image.startswith(("http://", "https://")):
            return {"blobUrl": image}
        raise HTTPException(
            status_code=400,
            detail={"error": f"{GUARDRAIL_LABEL}: image must be a base64 data URL or an http(s) URL"},
        )

    async def async_make_request(self, image: str) -> "AzureImageModerationGuardrailResponse":
        from litellm.types.proxy.guardrails.guardrail_hooks.azure.azure_image_moderation import (
            AzureImageModerationGuardrailRequestBody,
        )

        request_body: Final[AzureImageModerationGuardrailRequestBody] = {
            "image": self._to_azure_image(image),
            "categories": list(self.categories),
            "outputType": "FourSeverityLevels",
        }
        response_json: Final = await self._post_to_content_safety(
            "image:analyze", cast(dict[str, object], request_body)
        )
        response: Final = cast("AzureImageModerationGuardrailResponse", response_json)

        check_categories_against_thresholds(
            response["categoriesAnalysis"],
            severity_threshold=self.severity_threshold,
            severity_threshold_by_category=self.severity_threshold_by_category,
            default_severity_threshold=self.default_severity_threshold,
            guardrail_label=GUARDRAIL_LABEL,
        )
        return response

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]  # matches CustomGuardrail
        input_type: Literal["request", "response"],
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> GenericGuardrailAPIInputs:
        for image in inputs.get("images") or ():
            if image:
                await self.async_make_request(image=image)
        return inputs
