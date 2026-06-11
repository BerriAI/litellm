from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_skip_system_message_for_guardrail,
    effective_skip_tool_message_for_guardrail,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_CONVERSE_ACTIONS = frozenset({"converse", "converse-stream"})


def _is_converse_endpoint(endpoint: str) -> bool:
    parts = endpoint.rstrip("/").split("/")
    return bool(parts) and parts[-1] in _CONVERSE_ACTIONS


def _extract_converse_texts(
    body: dict,
    skip_system: bool,
    skip_tool: bool,
) -> Tuple[List[str], List[Tuple[str, int, int]]]:
    """
    Walk a Bedrock Converse request body and collect text content.

    Returns (texts, task_mappings) where each task_mapping is
    ("system", block_idx, -1) or ("message", msg_idx, content_idx).
    """
    texts: List[str] = []
    task_mappings: List[Tuple[str, int, int]] = []

    if not skip_system:
        for i, block in enumerate(body.get("system") or []):
            text = block.get("text") if isinstance(block, dict) else None
            if text:
                texts.append(text)
                task_mappings.append(("system", i, -1))

    for msg_idx, message in enumerate(body.get("messages") or []):
        if not isinstance(message, dict):
            continue
        for content_idx, block in enumerate(message.get("content") or []):
            if not isinstance(block, dict):
                continue
            if skip_tool and ("toolUse" in block or "toolResult" in block):
                continue
            text = block.get("text")
            if text:
                texts.append(text)
                task_mappings.append(("message", msg_idx, content_idx))

    return texts, task_mappings


def _write_back_texts(
    body: dict,
    guardrailed_texts: List[str],
    task_mappings: List[Tuple[str, int, int]],
) -> None:
    for idx, mapping in enumerate(task_mappings):
        if idx >= len(guardrailed_texts):
            break
        location, outer_idx, inner_idx = mapping
        if location == "system":
            system = body.get("system")
            if system and isinstance(system, list) and outer_idx < len(system):
                system[outer_idx]["text"] = guardrailed_texts[idx]
        else:
            messages = body.get("messages")
            if not (
                messages and isinstance(messages, list) and outer_idx < len(messages)
            ):
                continue
            content = messages[outer_idx].get("content")
            if content and isinstance(content, list) and inner_idx < len(content):
                content[inner_idx]["text"] = guardrailed_texts[idx]


class BedrockPassthroughGuardrailHandler(BaseTranslation):
    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> Any:
        endpoint = data.get("endpoint", "")
        body = data.get("data")

        if not _is_converse_endpoint(endpoint) or not isinstance(body, dict):
            verbose_proxy_logger.debug(
                "BedrockPassthroughGuardrailHandler: skipping non-converse endpoint %s",
                endpoint,
            )
            return data

        if not isinstance(body.get("messages"), list):
            return data

        skip_system = effective_skip_system_message_for_guardrail(guardrail_to_apply)
        skip_tool = effective_skip_tool_message_for_guardrail(guardrail_to_apply)

        texts, task_mappings = _extract_converse_texts(body, skip_system, skip_tool)

        if not texts:
            return data

        inputs = GenericGuardrailAPIInputs(texts=texts)
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
        if guardrailed_texts:
            _write_back_texts(body, guardrailed_texts, task_mappings)

        return data

    async def process_output_response(
        self,
        response: Any,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
        user_api_key_dict: Optional[Any] = None,
        request_data: Optional[dict] = None,
    ) -> Any:
        if not isinstance(response, dict):
            return response

        output_message = (
            response.get("output", {}).get("message", {})
            if isinstance(response.get("output"), dict)
            else {}
        )
        content_blocks = (
            output_message.get("content") if isinstance(output_message, dict) else None
        )

        if not isinstance(content_blocks, list):
            return response

        texts: List[str] = []
        text_indices: List[int] = []
        for i, block in enumerate(content_blocks):
            if isinstance(block, dict) and "text" in block:
                texts.append(block["text"])
                text_indices.append(i)

        if not texts:
            return response

        effective_request_data = request_data or {}
        if (
            "litellm_metadata" not in effective_request_data
            and user_api_key_dict is not None
        ):
            user_metadata = self.transform_user_api_key_dict_to_metadata(
                user_api_key_dict
            )
            if user_metadata:
                effective_request_data = {
                    **effective_request_data,
                    "litellm_metadata": user_metadata,
                }

        inputs = GenericGuardrailAPIInputs(texts=texts)
        model = effective_request_data.get("model") if effective_request_data else None
        if model:
            inputs["model"] = model

        guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
            inputs=inputs,
            request_data=effective_request_data,
            input_type="response",
            logging_obj=litellm_logging_obj,
        )

        guardrailed_texts = guardrailed_inputs.get("texts", [])
        for list_pos, block_idx in enumerate(text_indices):
            if list_pos < len(guardrailed_texts):
                content_blocks[block_idx]["text"] = guardrailed_texts[list_pos]

        return response
