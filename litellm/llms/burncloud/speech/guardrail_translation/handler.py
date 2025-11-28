from typing import TYPE_CHECKING, Any, Optional

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class BurnCloudTextToSpeechHandler(BaseTranslation):
    async def process_input_messages(
            self,
            data: dict,
            guardrail_to_apply: "CustomGuardrail",
            litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> Any:
        input_text = data.get("input")
        if input_text is None:
            verbose_proxy_logger.debug(
                "BurnCloud Text-to-Speech: No input text found in request data"
            )
            return data

        if isinstance(input_text, str):
            guardrailed_input = await guardrail_to_apply.apply_guardrail(
                text=input_text
            )
            data["input"] = guardrailed_input

            verbose_proxy_logger.debug(
                "BurnCloud Text-to-Speech: Applied guardrail to input text. "
                "Original length: %d, New length: %d",
                len(input_text),
                len(guardrailed_input),
            )
        else:
            verbose_proxy_logger.debug(
                "BurnCloud Text-to-Speech: Unexpected input type: %s. Expected string.",
                type(input_text),
            )

        return data

    async def process_output_response(
            self,
            response: Any,
            guardrail_to_apply: "CustomGuardrail",
            litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> Any:
        verbose_proxy_logger.debug(
            "BurnCloud Text-to-Speech: Output processing not applicable "
            "(output is audio data, not text)"
        )
        return response
