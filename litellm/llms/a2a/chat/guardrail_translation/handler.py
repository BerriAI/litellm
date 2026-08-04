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
from typing import TYPE_CHECKING, Any, Optional

from litellm._logging import verbose_proxy_logger
from litellm.llms.a2a.common_utils import serialize_a2a_data_part
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
    - Input: params.message.parts[].text (where kind == "text") or
      params.message.parts[].data (where kind == "data")
    - Output: result.message.parts[].text or result.artifacts[].parts[].text,
      and the "data" equivalents of both
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

        def _scan_input_part(part_idx: int, part: dict[str, Any]) -> tuple[str, int, str] | None:
            kind = part.get("kind")
            if kind == "text":
                text = part.get("text", "")
                return (text, part_idx, "text") if text else None
            if kind == "data":
                part_data = part.get("data")
                return (serialize_a2a_data_part(part_data), part_idx, "data") if part_data is not None else None
            return None

        # Extract text from all text parts, and serialized data from all data parts
        scanned = tuple(
            entry for part_idx, part in enumerate(parts) if (entry := _scan_input_part(part_idx, part)) is not None
        )
        texts_to_check = tuple(text for text, _, _ in scanned)
        # Track which parts contain scannable content, and which field to write
        # the guardrailed value back to ("text" or "data")
        part_mappings = tuple((part_idx, field) for _, part_idx, field in scanned)

        # Step 2: Apply guardrail to all texts in batch
        if texts_to_check:
            inputs = GenericGuardrailAPIInputs(
                texts=list(texts_to_check)  # mutable-ok: GenericGuardrailAPIInputs.texts is typed List[str]
            )

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
            if guardrailed_texts and len(guardrailed_texts) == len(part_mappings):
                for task_idx, (part_idx, field) in enumerate(part_mappings):
                    parts[part_idx][field] = guardrailed_texts[task_idx]

        verbose_proxy_logger.debug("A2A: Processed input message: %s", message)

        return data

    async def process_output_response(
        self,
        response: Any,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
        user_api_key_dict: Optional["UserAPIKeyAuth"] = None,
        request_data: dict | None = None,
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
            verbose_proxy_logger.warning("A2A: Unknown response type %s, skipping guardrail", type(response))
            return response

        result = response_dict.get("result", {})
        if not result or not isinstance(result, dict):
            verbose_proxy_logger.debug("A2A: No result in response, skipping guardrail")
            return response

        # Find all text-containing parts in the response. Each scanned entry is
        # (text, path_to_parts_list, part_index, field); path_to_parts_list is a
        # tuple of keys to navigate to the parts list.
        scanned = self._extract_texts_from_result(result=result)
        texts_to_check = tuple(text for text, _, _, _ in scanned)
        task_mappings = tuple((path, part_idx, field) for _, path, part_idx, field in scanned)

        if not texts_to_check:
            verbose_proxy_logger.debug("A2A: No text content in response")
            return response

        # Step 2: Apply guardrail to all texts in batch
        # Use the real request_data if provided (proxy path), otherwise
        # create a standalone dict (SDK / direct-call path).
        if request_data is None:
            request_data = {"response": response_dict}
        else:
            if "response" not in request_data:
                request_data["response"] = response_dict

        # Add user API key metadata with prefixed keys
        if "litellm_metadata" not in request_data:
            user_metadata = self.transform_user_api_key_dict_to_metadata(user_api_key_dict)
            if user_metadata:
                request_data["litellm_metadata"] = user_metadata

        inputs = GenericGuardrailAPIInputs(
            texts=list(texts_to_check)  # mutable-ok: GenericGuardrailAPIInputs.texts is typed List[str]
        )

        guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="response",
            logging_obj=litellm_logging_obj,
        )

        guardrailed_texts = guardrailed_inputs.get("texts", [])

        # Step 3: Apply guardrailed text back to original response
        if guardrailed_texts and len(guardrailed_texts) == len(task_mappings):
            for task_idx, (path, part_idx, field) in enumerate(task_mappings):
                self._apply_text_to_path(
                    result=result,
                    path=path,
                    part_idx=part_idx,
                    field=field,
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
        responses_so_far: list[Any],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
        user_api_key_dict: Optional["UserAPIKeyAuth"] = None,
        request_data: dict | None = None,
    ) -> list[Any]:
        """
        Process A2A streaming output by applying guardrails to accumulated text.

        responses_so_far can be a list of JSON-RPC 2.0 objects (dict or NDJSON str), e.g.:
        - task with history, status-update, artifact-update (with result.artifact.parts),
        - then status-update (final). Text is extracted from result.artifact.parts,
        result.message.parts, result.parts, etc., concatenated in order, guardrailed once,
        then the combined guardrailed text is written into the first chunk that had text
        and all other text parts in other chunks are cleared (in-place).
        """
        parsed, valid_parsed = self._parse_streaming_responses(responses_so_far)
        if not valid_parsed:
            return responses_so_far

        combined_text, chunk_indices_with_text = self._collect_text_from_parsed_chunks(valid_parsed)
        if not combined_text:
            return responses_so_far

        if request_data is None:
            request_data = {"responses_so_far": responses_so_far}
        else:
            if "responses_so_far" not in request_data:
                request_data["responses_so_far"] = responses_so_far

        if "litellm_metadata" not in request_data:
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
        first_chunk_with_text: int | None = chunk_indices_with_text[0] if chunk_indices_with_text else None

        for orig_i, obj in valid_parsed:
            result = obj.get("result", {})
            if not isinstance(result, dict):
                continue
            scanned = self._extract_texts_from_result(result=result)
            if not scanned:
                continue
            if orig_i == first_chunk_with_text:
                # Put full guardrailed text in first text part; clear others
                for task_idx, (_, path, part_idx, field) in enumerate(scanned):
                    text = guardrailed_text if task_idx == 0 else ""
                    self._apply_text_to_path(
                        result=result,
                        path=path,
                        part_idx=part_idx,
                        field=field,
                        text=text,
                    )
            else:
                for _, path, part_idx, field in scanned:
                    self._apply_text_to_path(
                        result=result,
                        path=path,
                        part_idx=part_idx,
                        field=field,
                        text="",
                    )

        # Write back to responses_so_far where we had NDJSON strings
        for i, item in enumerate(responses_so_far):
            if isinstance(item, str) and parsed[i] is not None:
                responses_so_far[i] = json.dumps(parsed[i]) + "\n"

        return responses_so_far

    def _parse_streaming_responses(
        self,
        responses_so_far: list[Any],
    ) -> tuple[list[dict[str, Any] | None], list[tuple[int, dict[str, Any]]]]:
        """Parse JSON-RPC items, returning aligned parsed list and valid entries."""
        parsed: list[dict[str, Any] | None] = [None] * len(responses_so_far)
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
        return parsed, valid_parsed

    def _collect_text_from_parsed_chunks(
        self,
        valid_parsed: list[tuple[int, dict[str, Any]]],
    ) -> tuple[str, list[int]]:
        """Collect text from parsed chunks, returning combined text and indices."""
        from litellm.llms.a2a.common_utils import extract_text_from_a2a_response

        text_parts: list[str] = []
        chunk_indices_with_text: list[int] = []
        for _idx, (orig_i, obj) in enumerate(valid_parsed):
            t = extract_text_from_a2a_response(obj)
            if t:
                text_parts.append(t)
                chunk_indices_with_text.append(orig_i)
        return "".join(text_parts), chunk_indices_with_text

    def _extract_texts_from_result(
        self,
        result: dict[str, Any],
    ) -> tuple[tuple[str, tuple[str, ...], int, str], ...]:
        """
        Extract text from all possible locations in an A2A result.

        Handles multiple response formats:
        1. Direct message with parts: {"parts": [...]}
        2. Nested message: {"message": {"parts": [...]}}
        3. Task with artifacts: {"artifacts": [{"parts": [...]}]}
        4. Task with status message: {"status": {"message": {"parts": [...]}}}
        5. Streaming artifact-update: {"artifact": {"parts": [...]}}

        Returns a tuple of (text, path_to_parts_list, part_index, field) entries.
        """
        entries: tuple[tuple[str, tuple[str, ...], int, str], ...] = ()

        # Case 1: Direct parts in result (direct message)
        if "parts" in result:
            entries += self._extract_texts_from_parts(parts=result["parts"], path=("parts",))

        # Case 2: Nested message
        message = result.get("message")
        if message and isinstance(message, dict) and "parts" in message:
            entries += self._extract_texts_from_parts(parts=message["parts"], path=("message", "parts"))

        # Case 3: Streaming artifact-update (singular artifact)
        artifact = result.get("artifact")
        if artifact and isinstance(artifact, dict) and "parts" in artifact:
            entries += self._extract_texts_from_parts(parts=artifact["parts"], path=("artifact", "parts"))

        # Case 4: Task with status message
        status = result.get("status", {})
        if isinstance(status, dict):
            status_message = status.get("message")
            if status_message and isinstance(status_message, dict) and "parts" in status_message:
                entries += self._extract_texts_from_parts(
                    parts=status_message["parts"], path=("status", "message", "parts")
                )

        # Case 5: Task with artifacts (plural, array)
        artifacts = result.get("artifacts", [])
        if artifacts and isinstance(artifacts, list):
            for artifact_idx, art in enumerate(artifacts):
                if isinstance(art, dict) and "parts" in art:
                    entries += self._extract_texts_from_parts(
                        parts=art["parts"], path=("artifacts", str(artifact_idx), "parts")
                    )

        return entries

    def _extract_texts_from_parts(
        self,
        parts: list[dict[str, Any]],
        path: tuple[str, ...],
        depth: int = 0,
        max_depth: int = 10,
    ) -> tuple[tuple[str, tuple[str, ...], int, str], ...]:
        """
        Extract text from message parts, serialized data from data parts, and
        recurse into any part that itself carries a nested "parts" list.

        Mirrors `extract_text_from_a2a_message`'s handling (including the
        recursion depth guard) so the two stay in sync. Returns a tuple of
        (text, path_to_parts_list, part_index, field) entries.
        """
        if depth >= max_depth:
            return ()

        def _scan(part_idx: int, part: dict[str, Any]) -> tuple[tuple[str, tuple[str, ...], int, str], ...]:
            kind = part.get("kind")
            if kind == "text":
                text = part.get("text", "")
                return ((text, path, part_idx, "text"),) if text else ()
            if kind == "data":
                part_data = part.get("data")
                if part_data is None:
                    return ()
                return ((serialize_a2a_data_part(part_data), path, part_idx, "data"),)
            if "parts" in part:
                return self._extract_texts_from_parts(
                    parts=part["parts"],
                    path=path + (str(part_idx), "parts"),
                    depth=depth + 1,
                    max_depth=max_depth,
                )
            return ()

        return tuple(entry for part_idx, part in enumerate(parts) for entry in _scan(part_idx, part))

    def _apply_text_to_path(
        self,
        result: dict[str | int, Any],
        path: tuple[str, ...],
        part_idx: int,
        field: str,
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

        # Update the guardrailed value in the part
        current[part_idx][field] = text
