"""
OpenAI Text Completion Handler for Unified Guardrails

This module provides guardrail translation support for OpenAI's text completion endpoint.
The handler processes the 'prompt' parameter for guardrails.
"""

import json
from collections.abc import Mapping, Sequence
from itertools import chain
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import (
    BaseTranslation,
    StreamingScanKey,
    StreamTransformSink,
)
from litellm.llms.base_llm.guardrail_translation.utils import (
    blocked_chat_stream_usage,
    stream_item_field,
    stream_item_items,
)
from litellm.types.utils import GenericGuardrailAPIInputs, TextChoices, TextCompletionResponse, Usage

if TYPE_CHECKING:
    from fastapi import HTTPException

    from litellm.integrations.custom_guardrail import CustomGuardrail, ModifyResponseException
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth


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
        user_api_key_dict: "UserAPIKeyAuth | None" = None,
        request_data: dict[str, object] | None = None,
    ) -> "TextCompletionResponse":
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

            self.merge_user_api_key_metadata_into_request(request_data, user_api_key_dict)

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
        responses_so_far: list[object],  # mutable-ok: rewrites the caller's buffered chunks in place
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
        user_api_key_dict: "UserAPIKeyAuth | None" = None,
        request_data: dict[str, object] | None = None,  # mutable-ok: the hooks share and mutate the request payload
        stream_transform_sink: StreamTransformSink | None = None,
        deliver_ended_stream_rewrites: bool = False,
    ) -> Sequence[object]:
        chunks: Final = _text_completion_chunks(responses_so_far)
        if not self._check_streaming_has_ended(chunks):
            return responses_so_far
        assembled: Final = self._assemble_ended_stream(chunks)
        pre_guardrail_texts: Final = _choice_texts(assembled)
        await self.process_output_response(
            response=assembled,
            guardrail_to_apply=guardrail_to_apply,
            litellm_logging_obj=litellm_logging_obj,
            user_api_key_dict=user_api_key_dict,
            request_data=request_data,
        )
        if not deliver_ended_stream_rewrites:
            return responses_so_far
        rewrites: Final = tuple(
            after for before, after in zip(pre_guardrail_texts, _choice_texts(assembled)) if after != before
        )
        for choice_index, text in rewrites:
            self._write_choice_text(chunks, choice_index, text)
        return responses_so_far

    def get_streaming_scan_key(self, responses_so_far: Sequence[object]) -> StreamingScanKey | None:
        chunks: Final = _text_completion_chunks(responses_so_far)
        if not self._check_streaming_has_ended(chunks):
            return None
        assembled: Final = self._assemble_ended_stream(chunks)
        return StreamingScanKey(texts=tuple(text for _, text in _choice_texts(assembled)), stream_ended=True)

    def released_stream_as_ended(self, responses_so_far: Sequence[object]) -> tuple[object, ...]:
        chunks: Final = _text_completion_chunks(responses_so_far)
        streamed_choices: Final = tuple(chain.from_iterable(chunk.choices for chunk in chunks))
        finished_indices: Final = frozenset(
            _choice_index(choice) for choice in streamed_choices if choice.finish_reason
        )
        unfinished_indices: Final = tuple(
            dict.fromkeys(
                _choice_index(choice) for choice in streamed_choices if _choice_index(choice) not in finished_indices
            )
        )
        if not unfinished_indices:
            return tuple(responses_so_far)
        terminator: Final = TextCompletionResponse(
            id=chunks[0].id,
            model=chunks[0].model,
            object="text_completion",
            choices=[TextChoices(index=index, text="", finish_reason="length") for index in unfinished_indices],
        )
        return (*responses_so_far, terminator)

    def build_block_sse_chunks(
        self,
        exc: "ModifyResponseException",
        stream_started: bool = False,
        responses_so_far: Sequence[object] | None = None,
    ) -> Sequence[bytes]:
        streamed: Final = _text_completion_chunks(responses_so_far or ())
        source: Final = streamed[0] if streamed else TextCompletionResponse(model=exc.model)
        message_chunk: Final = TextCompletionResponse(
            id=source.id,
            created=source.created,
            model=source.model or exc.model,
            object="text_completion",
            choices=[TextChoices(index=0, text=exc.message)],
        ).model_copy(update={"usage": None})
        final_chunk: Final = TextCompletionResponse(
            id=source.id,
            created=source.created,
            model=source.model or exc.model,
            object="text_completion",
            choices=[TextChoices(index=0, text="", finish_reason="content_filter")],
        ).model_copy(update={"usage": _blocked_stream_usage(exc.original_response)})
        return tuple(
            f"data: {chunk.model_dump_json(exclude_none=True)}\n\n".encode() for chunk in (message_chunk, final_chunk)
        )

    def build_stream_error_items(
        self,
        exc: "HTTPException",
        responses_so_far: Sequence[object] | None = None,
    ) -> Sequence[bytes] | None:
        from litellm.proxy.common_request_processing import sse_error_payload

        _, error_obj = sse_error_payload(exc)
        return (f'data: {{"error": {json.dumps(error_obj)}}}\n\n'.encode(),)

    def _check_streaming_has_ended(self, responses_so_far: Sequence[object]) -> bool:
        streamed_choices: Final = chain.from_iterable(stream_item_items(item, "choices") for item in responses_so_far)
        return any(stream_item_field(choice, "finish_reason") is not None for choice in streamed_choices)

    @staticmethod
    def _assemble_ended_stream(responses_so_far: Sequence[TextCompletionResponse]) -> TextCompletionResponse:
        streamed_choices: Final = tuple(chain.from_iterable(item.choices for item in responses_so_far))
        choice_indices: Final = tuple(dict.fromkeys(_choice_index(choice) for choice in streamed_choices))
        assembled_choices: Final = [
            TextChoices(
                index=index,
                text="".join(_choice_text(choice) for choice in streamed_choices if _choice_index(choice) == index),
                finish_reason=next(
                    (
                        choice.finish_reason
                        for choice in streamed_choices
                        if _choice_index(choice) == index and choice.finish_reason
                    ),
                    None,
                ),
            )
            for index in choice_indices
        ]
        first: Final = responses_so_far[0]
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
        streamed_choices: Final = chain.from_iterable(item.choices for item in responses_so_far)
        head, *tail = (choice for choice in streamed_choices if _choice_index(choice) == choice_index)
        head.text = text
        for choice in tail:
            if isinstance(stream_item_field(choice, "text"), str):
                choice.text = ""


def _blocked_stream_usage(original_response: object) -> Usage | None:
    prompt_tokens, completion_tokens = blocked_chat_stream_usage(original_response)
    if isinstance(original_response, list) and prompt_tokens == completion_tokens == 0:
        return None
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _text_completion_chunks(responses_so_far: Sequence[object]) -> tuple[TextCompletionResponse, ...]:
    return tuple(item for item in responses_so_far if isinstance(item, TextCompletionResponse))


def _choice_text(choice: TextChoices) -> str:
    text: Final = stream_item_field(choice, "text")
    return text if isinstance(text, str) else ""


def _choice_index(choice: TextChoices) -> int:
    return choice.index or 0


def _choice_texts(response: TextCompletionResponse) -> tuple[tuple[int, str], ...]:
    return tuple((_choice_index(choice), _choice_text(choice)) for choice in response.choices)
