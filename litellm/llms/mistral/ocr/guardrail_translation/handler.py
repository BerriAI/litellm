"""
OCR Handler for Unified Guardrails

Provides guardrail translation support for the OCR endpoint.
Processes the extracted markdown text from OCR pages.
"""

from typing import TYPE_CHECKING, Any, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.llms.base_llm.ocr.transformation import OCRResponse


class OCRHandler(BaseTranslation):
    """
    Handler for processing OCR requests/responses with guardrails.

    Input: The OCR input is a document URL/reference - not text content.
           We pass the document URL as text for guardrails that may want to
           validate or filter document sources.

    Output: OCR responses contain extracted markdown text per page.
            The handler extracts all page markdown, applies guardrails,
            and maps the guardrailed text back to the pages.
    """

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
    ) -> Any:
        """
        Process OCR input by applying guardrails to the document reference.

        The OCR input contains a document dict with a URL. We extract
        the URL and pass it to the guardrail for validation.

        Args:
            data: Request data containing 'document' parameter
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object

        Returns:
            Modified data with guardrails applied
        """
        document = data.get("document")
        if document is None or not isinstance(document, dict):
            verbose_proxy_logger.debug(
                "OCR guardrail: No valid document found in request data"
            )
            return data

        # Extract the document URL for guardrail checking
        texts_to_check: List[str] = []
        doc_type = document.get("type")
        if doc_type == "document_url":
            url = document.get("document_url")
            if url and isinstance(url, str):
                texts_to_check.append(url)
        elif doc_type == "image_url":
            url = document.get("image_url")
            if url and isinstance(url, str):
                texts_to_check.append(url)

        if not texts_to_check:
            return data

        inputs = GenericGuardrailAPIInputs(texts=texts_to_check)
        model = data.get("model")
        if model:
            inputs["model"] = model

        await guardrail_to_apply.apply_guardrail(
            inputs=inputs,
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )

        return data

    async def process_output_response(
        self,
        response: "OCRResponse",
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
        user_api_key_dict: Optional[Any] = None,
    ) -> Any:
        """
        Process OCR output by applying guardrails to extracted page text.

        Extracts markdown text from each OCR page, applies guardrails,
        and maps the guardrailed text back to the pages.

        Args:
            response: OCRResponse with pages containing markdown text
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object
            user_api_key_dict: User API key metadata

        Returns:
            Modified OCRResponse with guardrailed page text
        """
        if not hasattr(response, "pages") or not response.pages:
            verbose_proxy_logger.debug(
                "OCR guardrail: No pages found in OCR response"
            )
            return response

        # Extract markdown text from all pages
        texts_to_check: List[str] = []
        page_indices: List[int] = []
        for i, page in enumerate(response.pages):
            if hasattr(page, "markdown") and page.markdown:
                texts_to_check.append(page.markdown)
                page_indices.append(i)

        if not texts_to_check:
            return response

        inputs = GenericGuardrailAPIInputs(texts=texts_to_check)
        model = getattr(response, "model", None)
        if model:
            inputs["model"] = model

        # Add user metadata if available
        if user_api_key_dict is not None:
            metadata = self.transform_user_api_key_dict_to_metadata(user_api_key_dict)
            inputs.update(metadata)  # type: ignore

        guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="response",
            logging_obj=litellm_logging_obj,
        )

        # Map guardrailed text back to pages
        guardrailed_texts = guardrailed_inputs.get("texts", [])
        for idx, page_idx in enumerate(page_indices):
            if idx < len(guardrailed_texts):
                response.pages[page_idx].markdown = guardrailed_texts[idx]

        verbose_proxy_logger.debug(
            "OCR guardrail: Applied guardrail to %d pages",
            len(guardrailed_texts),
        )

        return response
