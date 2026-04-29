"""
OCI Generative AI — Cohere V2 chat transformation helpers (apiFormat="COHEREV2").

Required for ``cohere.command-a-vision*``: the V1 ``COHERE`` apiFormat carries a
flat ``message: str`` and silently drops images, while ``GENERIC`` is rejected
by the server with "Chat request type does not match serving model." Only the
V2 schema is accepted by Cohere vision models on OCI.

Wire format (mirrors ``oci.generative_ai_inference.models.CohereChatRequestV2``):

  {
    "apiFormat": "COHEREV2",
    "messages": [
      {"role": "USER", "content": [
          {"type": "TEXT", "text": "..."},
          {"type": "IMAGE_URL", "imageUrl": {"url": "...", "detail": "AUTO"}}
      ]},
      ...
    ],
    "isStream": false,
    "maxTokens": 4000,
    ...
  }

Tool-use is intentionally not implemented here — the langchain-oci provider
also explicitly rejects tool messages on V2 (`cohere.py:445-450`). Image and
text content are sufficient for the vision use case.
"""

import datetime
from typing import Any, Dict, List

from litellm.llms.oci.common_utils import OCIError
from litellm.types.llms.oci import (
    CohereV2ChatResult,
    CohereV2ContentUnion,
    CohereV2ImageContent,
    CohereV2ImageUrl,
    CohereV2Message,
    CohereV2TextContent,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, ModelResponse, Usage


_OPENAI_TO_V2_ROLE: Dict[str, str] = {
    "system": "SYSTEM",
    "user": "USER",
    "assistant": "ASSISTANT",
    # OCI's V2 schema also has "TOOL", but tool-use is out of scope here.
}


def _content_to_v2(content: Any) -> List[CohereV2ContentUnion]:
    """Adapt an OpenAI-format message ``content`` to a V2 content-block list."""
    if content is None:
        return []
    if isinstance(content, str):
        return [CohereV2TextContent(text=content)]
    if not isinstance(content, list):
        return [CohereV2TextContent(text=str(content))]

    parts: List[CohereV2ContentUnion] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text", "")
            if isinstance(text, str):
                parts.append(CohereV2TextContent(text=text))
        elif item_type == "image_url":
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
                detail = image_url.get("detail")
            else:
                url = image_url
                detail = None
            if not isinstance(url, str):
                raise OCIError(
                    status_code=400,
                    message="image_url must be a string or {'url': str, 'detail'?: str}",
                )
            parts.append(
                CohereV2ImageContent(
                    imageUrl=CohereV2ImageUrl(
                        url=url,
                        detail=detail if detail in ("AUTO", "HIGH", "LOW") else None,
                    )
                )
            )
        # Unknown content types are silently skipped — V2 only models text/image_url.
    return parts


def adapt_messages_to_cohere_v2_standard(
    messages: List[AllMessageValues],
) -> List[CohereV2Message]:
    """Build the V2 ``messages`` list from an OpenAI-format message array."""
    v2_messages: List[CohereV2Message] = []
    for msg in messages:
        role = msg.get("role")
        if role not in _OPENAI_TO_V2_ROLE:
            # Skip tool / function / unsupported roles.
            continue
        v2_messages.append(
            CohereV2Message(
                role=_OPENAI_TO_V2_ROLE[role],  # type: ignore[arg-type]
                content=_content_to_v2(msg.get("content")),
            )
        )
    return v2_messages


def _v2_finish_reason(reason: Any) -> Any:
    """Translate OCI V2 finishReason to OpenAI finish_reason."""
    if reason == "COMPLETE":
        return "stop"
    if reason == "MAX_TOKENS":
        return "length"
    if reason == "TOOL_CALL":
        return "tool_calls"
    if reason == "STOP_SEQUENCE":
        return "stop"
    return reason


def _join_text_content(content: Any) -> str:
    """Concatenate the text from a V2 assistant content list."""
    if not content:
        return ""
    out = []
    for part in content:
        # part may be a Pydantic model OR a plain dict (when parsed from response_json).
        ptype = getattr(part, "type", None) or (
            part.get("type") if isinstance(part, dict) else None
        )
        if ptype == "TEXT":
            text = getattr(part, "text", None) or (
                part.get("text") if isinstance(part, dict) else ""
            )
            out.append(text or "")
    return "".join(out)


def handle_cohere_v2_response(
    json_response: dict,
    model: str,
    model_response: ModelResponse,
) -> ModelResponse:
    """Parse a non-streaming Cohere V2 response into a LiteLLM ModelResponse."""
    parsed = CohereV2ChatResult(**json_response)

    model_response.model = model
    model_response.created = int(datetime.datetime.now().timestamp())

    chat = parsed.chatResponse
    text = _join_text_content(chat.message.content if chat.message else None)
    finish_reason = _v2_finish_reason(chat.finishReason)

    model_response.choices = [
        Choices(
            index=0,
            message={
                "role": "assistant",
                "content": text,
                "tool_calls": None,
            },
            finish_reason=finish_reason,
        )
    ]

    if chat.usage is not None:
        model_response.usage = Usage(  # type: ignore[attr-defined]
            prompt_tokens=chat.usage.promptTokens,
            completion_tokens=chat.usage.completionTokens,
            total_tokens=chat.usage.totalTokens,
        )
    else:
        model_response.usage = Usage(  # type: ignore[attr-defined]
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

    return model_response
