from typing import TYPE_CHECKING, Final

from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy._types import UserAPIKeyAuth


class OpenAIVideoGenerationHandler(BaseTranslation):
    async def process_input_messages(
        self,
        data: dict[str, object],  # mutable-ok: BaseTranslation contract passes the proxy's request dict through
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> dict[str, object]:  # mutable-ok: BaseTranslation contract returns the proxy's request dict
        prompt: Final = data.get("prompt")
        if not isinstance(prompt, str):
            return data

        model: Final = data.get("model")
        texts: Final = [prompt]  # mutable-ok: GenericGuardrailAPIInputs.texts is declared list[str]
        inputs: Final = (
            GenericGuardrailAPIInputs(texts=texts, model=model)
            if isinstance(model, str)
            else GenericGuardrailAPIInputs(texts=texts)
        )
        guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(  # pyright: ignore[reportUnknownMemberType]  # request_data is a bare dict
            inputs=inputs,
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )
        guardrailed_texts: Final = guardrailed_inputs.get("texts")
        guardrailed_prompt: Final = guardrailed_texts[0] if guardrailed_texts else prompt
        return {**data, "prompt": guardrailed_prompt}  # mutable-ok: BaseTranslation contract returns a dict

    async def process_output_response(
        self,
        response: object,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
        user_api_key_dict: "UserAPIKeyAuth | None" = None,
        request_data: dict[str, object] | None = None,  # mutable-ok: BaseTranslation contract
    ) -> object:
        return response
