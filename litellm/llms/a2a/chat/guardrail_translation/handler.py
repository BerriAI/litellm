"""
A2A Protocol Handler for Unified Guardrails

This module provides guardrail translation support for A2A (Agent-to-Agent) Protocol.
It handles both JSON-RPC 2.0 input requests and output responses, extracting text
from message parts and applying guardrails.

A2A Protocol Format:
- Input: JSON-RPC 2.0 with params.message.parts containing text parts
- Output: JSON-RPC 2.0 with result containing message/artifact parts
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy._types import UserAPIKeyAuth


class A2AGuardrailHandler(BaseTranslation):
    """
    Handler for processing A2A Protocol messages with guardrails.

    This class provides methods to:
    1. Process input messages (pre-call hook) - extracts text from A2A message parts
    2. Process output responses (post-call hook) - extracts text from A2A response parts

    A2A Message Format:
    - Input: params.message.parts[].text (where kind == "text")
    - Output: result.message.parts[].text or result.artifacts[].parts[].text
    """

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> Any:
        """
        Process A2A input messages by applying guardrails to text content.

        Extracts text from A2A message parts and applies guardrails.

        Args:
            data: The A2A JSON-RPC 2.0 request data
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object

        Returns:
            Modified data with guardrails applied to text content
        """
        # A2A request format: { "params": { "message": { "parts": [...] } } }
        params = data.get("params", {})
        message = params.get("message", {})
        parts = message.get("parts", [])

        if not parts:
            verbose_proxy_logger.debug("A2A: No parts in message, skipping guardrail")
            return data

        texts_to_check: List[str] = []
        text_part_indices: List[int] = []  # Track which parts contain text

        # Step 1: Extract text from all text parts
        for part_idx, part in enumerate(parts):
            if part.get("kind") == "text":
                text = part.get("text", "")
                if text:
                    texts_to_check.append(text)
                    text_part_indices.append(part_idx)

        # Step 2: Apply guardrail to all texts in batch
        if texts_to_check:
            inputs = GenericGuardrailAPIInputs(texts=texts_to_check)

            # Pass the structured A2A message to guardrails
            inputs["structured_messages"] = [message]

            # Include agent model info if available
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

            # Step 3: Apply guardrailed text back to original parts
            if guardrailed_texts and len(guardrailed_texts) == len(text_part_indices):
                for task_idx, part_idx in enumerate(text_part_indices):
                    parts[part_idx]["text"] = guardrailed_texts[task_idx]

        verbose_proxy_logger.debug("A2A: Processed input message: %s", message)

        return data

    async def process_output_response(
        self,
        response: Any,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
        user_api_key_dict: Optional["UserAPIKeyAuth"] = None,
    ) -> Any:
        """
        Process A2A output response by applying guardrails to text content.

        Handles multiple A2A response formats:
        - Direct message: {"result": {"kind": "message", "parts": [...]}}
        - Nested message: {"result": {"message": {"parts": [...]}}}
        - Task with artifacts: {"result": {"kind": "task", "artifacts": [{"parts": [...]}]}}
        - Task with status message: {"result": {"kind": "task", "status": {"message": {"parts": [...]}}}}

        Args:
            response: A2A JSON-RPC 2.0 response dict or object
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object
            user_api_key_dict: User API key metadata

        Returns:
            Modified response with guardrails applied to text content
        """
        # Handle both dict and Pydantic model responses
        if hasattr(response, "model_dump"):
            response_dict = response.model_dump()
            is_pydantic = True
        elif isinstance(response, dict):
            response_dict = response
            is_pydantic = False
        else:
            verbose_proxy_logger.warning(
                "A2A: Unknown response type %s, skipping guardrail", type(response)
            )
            return response

        result = response_dict.get("result", {})
        if not result or not isinstance(result, dict):
            verbose_proxy_logger.debug("A2A: No result in response, skipping guardrail")
            return response

        # Find all text-containing parts in the response
        texts_to_check: List[str] = []
        # Each mapping is (path_to_parts_list, part_index)
        # path_to_parts_list is a tuple of keys to navigate to the parts list
        task_mappings: List[Tuple[Tuple[str, ...], int]] = []

        # Extract texts from all possible locations
        self._extract_texts_from_result(
            result=result,
            texts_to_check=texts_to_check,
            task_mappings=task_mappings,
        )

        if not texts_to_check:
            verbose_proxy_logger.debug("A2A: No text content in response")
            return response

        # Step 2: Apply guardrail to all texts in batch
        # Create a request_data dict with response info and user API key metadata
        request_data: dict = {"response": response_dict}

        # Add user API key metadata with prefixed keys
        user_metadata = self.transform_user_api_key_dict_to_metadata(user_api_key_dict)
        if user_metadata:
            request_data["litellm_metadata"] = user_metadata

        inputs = GenericGuardrailAPIInputs(texts=texts_to_check)

        guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="response",
            logging_obj=litellm_logging_obj,
        )

        guardrailed_texts = guardrailed_inputs.get("texts", [])

        # Step 3: Apply guardrailed text back to original response
        if guardrailed_texts and len(guardrailed_texts) == len(task_mappings):
            for task_idx, (path, part_idx) in enumerate(task_mappings):
                self._apply_text_to_path(
                    result=result,
                    path=path,
                    part_idx=part_idx,
                    text=guardrailed_texts[task_idx],
                )

        verbose_proxy_logger.debug("A2A: Processed output response")

        # Update the original response
        if is_pydantic:
            # For Pydantic models, we need to update the underlying dict
            # and the model will reflect the changes
            response_dict["result"] = result
            return response
        else:
            response["result"] = result
            return response

    async def process_output_streaming_response(
        self,
        responses_so_far: List[Any],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
        user_api_key_dict: Optional["UserAPIKeyAuth"] = None,
    ) -> List[Any]:
        """
        Process A2A streaming output by applying guardrails to accumulated text.

        responses_so_far can be a list of JSON-RPC 2.0 objects (dict or NDJSON str), e.g.:
        - task with history, status-update, artifact-update (with result.artifact.parts),
        - then status-update (final). Text is extracted from result.artifact.parts,
        result.message.parts, result.parts, etc., concatenated in order, guardrailed once,
        then the combined guardrailed text is written into the first chunk that had text
        and all other text parts in other chunks are cleared (in-place).
        """
        from litellm.llms.a2a.common_utils import extract_text_from_a2a_response

        # Parse each item; keep alignment with responses_so_far (None where unparseable)
        parsed: List[Optional[Dict[str, Any]]] = [None] * len(responses_so_far)
        for i, item in enumerate(responses_so_far):
            if isinstance(item, dict):
                obj = item
            elif isinstance(item, str):
                try:
                    obj = json.loads(item.strip())
                except (json.JSONDecodeError, TypeError):
                    continue
            else:
                continue
            if isinstance(obj.get("result"), dict):
                parsed[i] = obj

        valid_parsed = [(i, obj) for i, obj in enumerate(parsed) if obj is not None]
        if not valid_parsed:
            return responses_so_far

        # Collect text from each chunk in order (by original index in responses_so_far)
        text_parts: List[str] = []
        chunk_indices_with_text: List[int] = []  # indices into valid_parsed
        for idx, (orig_i, obj) in enumerate(valid_parsed):
            t = extract_text_from_a2a_response(obj)
            if t:
                text_parts.append(t)
                chunk_indices_with_text.append(orig_i)

        combined_text = "".join(text_parts)
        if not combined_text:
            return responses_so_far

        request_data: dict = {"responses_so_far": responses_so_far}
        user_metadata = self.transform_user_api_key_dict_to_metadata(user_api_key_dict)
        if user_metadata:
            request_data["litellm_metadata"] = user_metadata

        inputs = GenericGuardrailAPIInputs(texts=[combined_text])
        guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="response",
            logging_obj=litellm_logging_obj,
        )
        guardrailed_texts = guardrailed_inputs.get("texts", [])
        if not guardrailed_texts:
            return responses_so_far
        guardrailed_text = guardrailed_texts[0]

        # Find first chunk (by original index) that has text; put full guardrailed text there and clear rest
        first_chunk_with_text: Optional[int] = (
            chunk_indices_with_text[0] if chunk_indices_with_text else None
        )

        for orig_i, obj in valid_parsed:
            result = obj.get("result", {})
            if not isinstance(result, dict):
                continue
            texts_in_chunk: List[str] = []
            mappings: List[Tuple[Tuple[str, ...], int]] = []
            self._extract_texts_from_result(
                result=result,
                texts_to_check=texts_in_chunk,
                task_mappings=mappings,
            )
            if not mappings:
                continue
            if orig_i == first_chunk_with_text:
                # Put full guardrailed text in first text part; clear others
                for task_idx, (path, part_idx) in enumerate(mappings):
                    text = guardrailed_text if task_idx == 0 else ""
                    self._apply_text_to_path(
                        result=result,
                        path=path,
                        part_idx=part_idx,
                        text=text,
                    )
            else:
                for path, part_idx in mappings:
                    self._apply_text_to_path(
                        result=result,
                        path=path,
                        part_idx=part_idx,
                        text="",
                    )

        # Write back to responses_so_far where we had NDJSON strings
        for i, item in enumerate(responses_so_far):
            if isinstance(item, str) and parsed[i] is not None:
                responses_so_far[i] = json.dumps(parsed[i]) + "\n"

        return responses_so_far

    def _extract_texts_from_result(
        self,
        result: Dict[str, Any],
        texts_to_check: List[str],
        task_mappings: List[Tuple[Tuple[str, ...], int]],
    ) -> None:
        """
        Extract text from all possible locations in an A2A result.

        Handles multiple response formats:
        1. Direct message with parts: {"parts": [...]}
        2. Nested message: {"message": {"parts": [...]}}
        3. Task with artifacts: {"artifacts": [{"parts": [...]}]}
        4. Task with status message: {"status": {"message": {"parts": [...]}}}
        5. Streaming artifact-update: {"artifact": {"parts": [...]}}
        """
        # Case 1: Direct parts in result (direct message)
        if "parts" in result:
            self._extract_texts_from_parts(
                parts=result["parts"],
                path=("parts",),
                texts_to_check=texts_to_check,
                task_mappings=task_mappings,
            )

        # Case 2: Nested message
        message = result.get("message")
        if message and isinstance(message, dict) and "parts" in message:
            self._extract_texts_from_parts(
                parts=message["parts"],
                path=("message", "parts"),
                texts_to_check=texts_to_check,
                task_mappings=task_mappings,
            )

        # Case 3: Streaming artifact-update (singular artifact)
        artifact = result.get("artifact")
        if artifact and isinstance(artifact, dict) and "parts" in artifact:
            self._extract_texts_from_parts(
                parts=artifact["parts"],
                path=("artifact", "parts"),
                texts_to_check=texts_to_check,
                task_mappings=task_mappings,
            )

        # Case 4: Task with status message
        status = result.get("status", {})
        if isinstance(status, dict):
            status_message = status.get("message")
            if (
                status_message
                and isinstance(status_message, dict)
                and "parts" in status_message
            ):
                self._extract_texts_from_parts(
                    parts=status_message["parts"],
                    path=("status", "message", "parts"),
                    texts_to_check=texts_to_check,
                    task_mappings=task_mappings,
                )

        # Case 5: Task with artifacts (plural, array)
        artifacts = result.get("artifacts", [])
        if artifacts and isinstance(artifacts, list):
            for artifact_idx, art in enumerate(artifacts):
                if isinstance(art, dict) and "parts" in art:
                    self._extract_texts_from_parts(
                        parts=art["parts"],
                        path=("artifacts", str(artifact_idx), "parts"),
                        texts_to_check=texts_to_check,
                        task_mappings=task_mappings,
                    )

    def _extract_texts_from_parts(
        self,
        parts: List[Dict[str, Any]],
        path: Tuple[str, ...],
        texts_to_check: List[str],
        task_mappings: List[Tuple[Tuple[str, ...], int]],
    ) -> None:
        """Extract text from message parts."""
        for part_idx, part in enumerate(parts):
            if part.get("kind") == "text":
                text = part.get("text", "")
                if text:
                    texts_to_check.append(text)
                    task_mappings.append((path, part_idx))

    def _apply_text_to_path(
        self,
        result: Dict[Union[str, int], Any],
        path: Tuple[str, ...],
        part_idx: int,
        text: str,
    ) -> None:
        """Apply guardrailed text back to the specified path in the result."""
        # Navigate to the parts list
        current = result
        for key in path:
            if key.isdigit():
                # Array index
                current = current[int(key)]
            else:
                current = current[key]

        # Update the text in the part
        current[part_idx]["text"] = text
