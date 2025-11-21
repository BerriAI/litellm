from typing import TYPE_CHECKING, Any

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail


class BurnCloudTextToSpeechHandler(BaseTranslation):
    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
    ) -> Any:
        """
        Process input text by applying guardrails.

        Args:
            data: Request data dictionary containing 'input' parameter
            guardrail_to_apply: The guardrail instance to apply

        Returns:
            Modified data with guardrails applied to input text
        """
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
    ) -> Any:
        """
        Process output - not applicable for text-to-speech.

        The output is audio (binary data), not text, so there's nothing to apply
        guardrails to. This method returns the response unchanged.

        Args:
            response: Binary audio response
            guardrail_to_apply: The guardrail instance (unused)

        Returns:
            Unmodified response (audio data doesn't need text guardrails)
        """
        verbose_proxy_logger.debug(
            "BurnCloud Text-to-Speech: Output processing not applicable "
            "(output is audio data, not text)"
        )
        return response
