"""
OpenAI Text Completion Handler for Unified Guardrails

This module provides guardrail translation support for OpenAI's text completion endpoint.
The handler processes the 'prompt' parameter for guardrails.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation, StreamTransformSink
from litellm.llms.base_llm.guardrail_translation.utils import stream_item_field, stream_item_items
from litellm.types.utils import GenericGuardrailAPIInputs, TextChoices, TextCompletionResponse

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy._types import UserAPIKeyAuth


class OpenAITextCompletionHandler(BaseTranslation):
    """
    Handler for processing OpenAI text completion requests with guardrails.

    This class provides methods to:
    1. Process input prompt (pre-call hook)
    2. Process output response (post-call hook)

    The handler specifically processes the 'prompt' parameter which can be:
    - A single string
    - A list of strings (for batch completions)
    """

    delivers_ended_stream_rewrites = True
    assembles_streamed_response = True

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> Any:
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
            verbose_proxy_logger.debug("OpenAI Text Completion: No prompt found in request data")
            return data

        if isinstance(prompt, str):
            # Single string prompt
            inputs = GenericGuardrailAPIInputs(texts=[prompt])
            # Include model information if available
            model = data.get("model")
            if model:
                inputs["model"] = model
            guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                inputs=inputs,
                request_data=data,
                input_type="request",
                logging_obj=litellm_logging_obj,
            )
            guardrailed_texts = guardrailed_inputs.get("texts", [])
            data["prompt"] = guardrailed_texts[0] if guardrailed_texts else prompt

            verbose_proxy_logger.debug(
                "OpenAI Text Completion: Applied guardrail to string prompt. Original length: %d, New length: %d",
                len(prompt),
                len(data["prompt"]),
            )

        elif isinstance(prompt, list):
            # List of string prompts (batch completion)
            texts_to_check: Final = []
            text_indices: Final = []  # Track which prompts are strings

            for idx, p in enumerate(prompt):
                if isinstance(p, str):
                    texts_to_check.append(p)
                    text_indices.append(idx)

            if texts_to_check:
                inputs = GenericGuardrailAPIInputs(texts=texts_to_check)
                # Include model information if available
                model = data.get("model")
                if model:
                    inputs["model"] = model
                guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                    inputs=inputs,
                    request_data=data,
                    input_type="request",
                    logging_obj=litellm_logging_obj,
                )
                guardrailed_texts = guardrailed_inputs.get("texts", [])

                # Replace guardrailed texts back
                for guardrail_idx, prompt_idx in enumerate(text_indices):
                    if guardrail_idx < len(guardrailed_texts):
                        data["prompt"][prompt_idx] = guardrailed_texts[guardrail_idx]
                        verbose_proxy_logger.debug(
                            "OpenAI Text Completion: Applied guardrail to prompt[%d]. "
                            "Original length: %d, New length: %d",
                            prompt_idx,
                            len(texts_to_check[guardrail_idx]),
                            len(guardrailed_texts[guardrail_idx]),
                        )

        else:
            verbose_proxy_logger.warning(
                "OpenAI Text Completion: Unexpected prompt type: %s. Expected string or list.",
                type(prompt),
            )

        return data

    async def process_output_response(
        self,
        response: "TextCompletionResponse",
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
        user_api_key_dict: Any | None = None,
        request_data: dict | None = None,
    ) -> Any:
        """
        Process output response by applying guardrails to completion text.

        Args:
            response: Text completion response object
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object
            user_api_key_dict: User API key metadata to pass to guardrails

        Returns:
            Modified response with guardrails applied to completion text
        """
        if not hasattr(response, "choices") or not response.choices:
            verbose_proxy_logger.debug("OpenAI Text Completion: No choices in response to process")
            return response

        # Collect all texts to check
        texts_to_check: Final = []
        choice_indices: Final = []

        for idx, choice in enumerate(response.choices):
            if hasattr(choice, "text") and isinstance(choice.text, str):
                texts_to_check.append(choice.text)
                choice_indices.append(idx)

        # Apply guardrails in batch
        if texts_to_check:
            # Use the real request_data if provided (proxy path), otherwise
            # create a standalone dict (SDK / direct-call path).
            if request_data is None:
                request_data = {"response": response}
            else:
                if "response" not in request_data:
                    request_data["response"] = response

            # Add user API key metadata with prefixed keys
            if "litellm_metadata" not in request_data:
                user_metadata: Final = self.transform_user_api_key_dict_to_metadata(user_api_key_dict)
                if user_metadata:
                    request_data["litellm_metadata"] = user_metadata

            inputs: Final = GenericGuardrailAPIInputs(texts=texts_to_check)
            # Include model information from the response if available
            if hasattr(response, "model") and response.model:
                inputs["model"] = response.model
            guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="response",
                logging_obj=litellm_logging_obj,
            )
            guardrailed_texts: Final = guardrailed_inputs.get("texts", [])

            # Apply guardrailed texts back to choices
            for guardrail_idx, choice_idx in enumerate(choice_indices):
                if guardrail_idx < len(guardrailed_texts):
                    original_text = response.choices[choice_idx].text
                    response.choices[choice_idx].text = guardrailed_texts[guardrail_idx]

                    verbose_proxy_logger.debug(
                        "OpenAI Text Completion: Applied guardrail to choice[%d] text. "
                        "Original length: %d, New length: %d",
                        choice_idx,
                        len(original_text),
                        len(guardrailed_texts[guardrail_idx]),
                    )

        return response

    async def process_output_streaming_response(
        self,
        responses_so_far: list[TextCompletionResponse],  # mutable-ok: rewrites the caller's buffered chunks in place
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
        user_api_key_dict: "UserAPIKeyAuth | None" = None,
        request_data: dict | None = None,
        stream_transform_sink: StreamTransformSink | None = None,
        deliver_ended_stream_rewrites: bool = False,
    ) -> list[TextCompletionResponse]:
        if not self._check_streaming_has_ended(responses_so_far):
            return responses_so_far
        assembled: Final = self._assemble_ended_stream(responses_so_far)
        pre_guardrail_texts: Final = tuple(choice.text for choice in assembled.choices)
        await self.process_output_response(
            response=assembled,
            guardrail_to_apply=guardrail_to_apply,
            litellm_logging_obj=litellm_logging_obj,
            user_api_key_dict=user_api_key_dict,
            request_data=request_data,
        )
        if not deliver_ended_stream_rewrites:
            return responses_so_far
        for choice, pre_guardrail_text in zip(assembled.choices, pre_guardrail_texts):
            if choice.text != pre_guardrail_text:
                self._write_choice_text(responses_so_far, choice.index, choice.text)
        return responses_so_far

    def _check_streaming_has_ended(self, responses_so_far: Sequence[object]) -> bool:
        return any(
            stream_item_field(choice, "finish_reason") is not None
            for item in responses_so_far
            for choice in stream_item_items(item, "choices")
        )

    @staticmethod
    def _assemble_ended_stream(responses_so_far: Sequence[TextCompletionResponse]) -> TextCompletionResponse:
        first: Final = responses_so_far[0]
        streamed_choices: Final = tuple(choice for item in responses_so_far for choice in item.choices)
        choice_indices: Final = tuple(dict.fromkeys(choice.index for choice in streamed_choices))
        assembled_choices: Final = [  # mutable-ok: TextCompletionResponse only accepts a list of choices
            TextChoices(
                index=index,
                text="".join(
                    choice.text for choice in streamed_choices if choice.index == index and isinstance(choice.text, str)
                ),
                finish_reason=next(
                    (
                        choice.finish_reason
                        for choice in reversed(streamed_choices)
                        if choice.index == index and choice.finish_reason is not None
                    ),
                    None,
                ),
            )
            for index in choice_indices
        ]
        return TextCompletionResponse(
            id=first.id,
            created=first.created,
            model=first.model,
            object="text_completion",
            choices=assembled_choices,
            usage=next((item.usage for item in reversed(responses_so_far) if item.usage is not None), None),
        )

    @staticmethod
    def _write_choice_text(responses_so_far: Sequence[TextCompletionResponse], choice_index: int, text: str) -> None:
        carriers: Final = tuple(
            choice
            for item in responses_so_far
            for choice in item.choices
            if choice.index == choice_index and isinstance(choice.text, str)
        )
        for position, carrier in enumerate(carriers):
            carrier.text = text if position == 0 else ""
