"""
OpenAI Embeddings Handler for Unified Guardrails

This module provides guardrail translation support for OpenAI's embeddings endpoint.
The handler processes the 'input' parameter for guardrails.
"""

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.llms.base_llm.guardrail_translation.utils import unappliable_request_rewrite
from litellm.types.llms.openai import OpenAIChatCompletionTextObject
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
    from litellm.types.utils import EmbeddingResponse


_TEXT_BLOCK: Final = TypeAdapter(OpenAIChatCompletionTextObject)
_BLOCK_FIELDS: Final = TypeAdapter(dict[str, object])
_GROUP: Final = TypeAdapter(list[object])


def _text_block_text(item: object) -> str | None:
    if not isinstance(item, dict):
        return None
    try:
        return _TEXT_BLOCK.validate_python(item, strict=True)["text"]
    except ValidationError:
        return None


def _input_items(input_data: Sequence[object]) -> Iterator[object]:
    """Top-level items, with the items of a grouped (nested) list in place of the group."""
    for item in input_data:
        if isinstance(item, list):
            yield from _GROUP.validate_python(item)
        else:
            yield item


def _has_text_block(input_data: Sequence[object]) -> bool:
    return any(_text_block_text(item) is not None for item in _input_items(input_data))


def _slot_text(item: object) -> str | None:
    return item if isinstance(item, str) else _text_block_text(item)


def _slot_texts(input_data: Sequence[object]) -> tuple[str, ...]:
    """Every string and text block text, top level and inside groups, in input order."""
    return tuple(text for item in _input_items(input_data) if (text := _slot_text(item)) is not None)


def _with_slot_text(item: object, texts: Iterator[str]) -> object:
    if isinstance(item, str):
        return next(texts)
    if _text_block_text(item) is None:
        return item
    return {**_BLOCK_FIELDS.validate_python(item), "text": next(texts)}


def _with_slot_texts(input_data: Sequence[object], texts: Iterator[str]) -> Sequence[object]:
    """`input_data` with each string and text block text swapped for the next text, keeping blocks and groups."""
    return [
        [_with_slot_text(sub_item, texts) for sub_item in _GROUP.validate_python(item)]
        if isinstance(item, list)
        else _with_slot_text(item, texts)
        for item in input_data
    ]


class OpenAIEmbeddingsHandler(BaseTranslation):
    """
    Handler for processing OpenAI embeddings requests with guardrails.

    This class provides methods to:
    1. Process input text (pre-call hook)
    2. Process output response (post-call hook) - embeddings don't typically need output guardrails

    The handler specifically processes the 'input' parameter which can be:
    - A single string
    - A list of strings (for batch embeddings)
    - A list of integers (token IDs - not processed by guardrails)
    - A list of lists of integers (batch token IDs - not processed by guardrails)
    """

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> Mapping[str, object]:
        """
        Process input text by applying guardrails to text content.

        Args:
            data: Request data dictionary containing 'input' parameter
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object

        Returns:
            Modified data with guardrails applied to input
        """
        input_data: Final = data.get("input")
        if input_data is None:
            verbose_proxy_logger.debug("OpenAI Embeddings: No input found in request data")
            return data

        if isinstance(input_data, str):
            data = await self._process_string_input(data, input_data, guardrail_to_apply, litellm_logging_obj)
        elif isinstance(input_data, list):
            data = await self._process_list_input(data, input_data, guardrail_to_apply, litellm_logging_obj)
        else:
            verbose_proxy_logger.warning(
                "OpenAI Embeddings: Unexpected input type: %s. Expected string or list.",
                type(input_data),
            )

        return data

    async def _process_string_input(
        self,
        data: dict,
        input_data: str,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None",
    ) -> dict:
        """Process a single string input through the guardrail."""
        inputs: Final = GenericGuardrailAPIInputs(texts=[input_data])
        if model := data.get("model"):
            inputs["model"] = model

        guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=inputs,
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )

        if guardrailed_texts := guardrailed_inputs.get("texts"):
            data["input"] = guardrailed_texts[0]
            verbose_proxy_logger.debug(
                "OpenAI Embeddings: Applied guardrail to string input. Original length: %d, New length: %d",
                len(input_data),
                len(data["input"]),
            )

        return data

    async def _process_list_input(
        self,
        data: dict,
        input_data: list[str | int | list[int]],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None",
    ) -> dict:
        """Process a list input through the guardrail (if it contains strings or text blocks)."""
        if len(input_data) == 0:
            return data

        first_item: Final = input_data[0]
        has_text_block: Final = _has_text_block(input_data)

        # Skip non-text inputs (token IDs)
        if not has_text_block and isinstance(first_item, (int, list)):
            verbose_proxy_logger.debug("OpenAI Embeddings: Input is token IDs, skipping guardrail processing")
            return data

        if not has_text_block and not isinstance(first_item, str):
            verbose_proxy_logger.warning(
                "OpenAI Embeddings: Unexpected input list item type: %s",
                type(first_item),
            )
            return data

        slot_texts: Final = _slot_texts(input_data) if has_text_block else ()
        inputs: Final = GenericGuardrailAPIInputs(texts=list(slot_texts) if has_text_block else input_data)
        if model := data.get("model"):
            inputs["model"] = model

        guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=inputs,
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )

        if guardrailed_texts := guardrailed_inputs.get("texts"):
            if has_text_block and len(guardrailed_texts) != len(slot_texts):
                raise unappliable_request_rewrite(guardrail_to_apply.guardrail_name)
            data["input"] = (
                _with_slot_texts(input_data, iter(guardrailed_texts)) if has_text_block else guardrailed_texts
            )
            verbose_proxy_logger.debug(
                "OpenAI Embeddings: Applied guardrail to %d inputs",
                len(guardrailed_texts),
            )

        return data

    async def process_output_response(
        self,
        response: "EmbeddingResponse",
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
        user_api_key_dict: "UserAPIKeyAuth | None" = None,
        request_data: dict | None = None,
    ) -> "EmbeddingResponse":
        """
        Process output response - embeddings responses contain vectors, not text.

        For embeddings, the output is numerical vectors, so there's typically
        no text content to apply guardrails to. This method is a no-op but
        is included for interface consistency.

        Args:
            response: Embedding response object
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object
            user_api_key_dict: User API key metadata

        Returns:
            Unmodified response (embeddings don't have text output to guard)
        """
        verbose_proxy_logger.debug(
            "OpenAI Embeddings: Output response processing skipped - embeddings contain vectors, not text"
        )
        return response
