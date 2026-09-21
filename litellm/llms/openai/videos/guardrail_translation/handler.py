from typing import TYPE_CHECKING, Final

from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy._types import UserAPIKeyAuth


class OpenAIVideoGenerationHandler(BaseTranslation):
    """Scans the text `prompt` of video create, remix, edit and extension requests."""

    async def process_input_messages(
        self,
        data: dict[str, object],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> dict[str, object]:
        prompt: Final = data.get("prompt")
        if not isinstance(prompt, str):
            return data

        model: Final = data.get("model")
        inputs: Final = (
            GenericGuardrailAPIInputs(texts=[prompt], model=model)
            if isinstance(model, str)
            else GenericGuardrailAPIInputs(texts=[prompt])
        )
        guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(  # pyright: ignore[reportUnknownMemberType]  # request_data is a bare dict
            inputs=inputs,
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )
        guardrailed_texts: Final = guardrailed_inputs.get("texts", [])
        return {**data, "prompt": guardrailed_texts[0] if guardrailed_texts else prompt}

    async def process_output_response(
        self,
        response: object,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
        user_api_key_dict: "UserAPIKeyAuth | None" = None,
        request_data: dict[str, object] | None = None,
    ) -> object:
        return response
