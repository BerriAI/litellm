"""Image scanning for the standalone guardrail endpoint."""

from collections.abc import Sequence
from typing import Final, Literal

from fastapi import HTTPException

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.bedrock.guardrail_attachments import find_request_attachments
from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrail
from litellm.types.guardrails import GuardrailEventHooks


async def scan_images(
    guardrail: CustomGuardrail,
    images: Sequence[str],
    request_data: dict[str, object],  # mutable-ok: the guardrail records its verdict on the request dict
    input_type: Literal["request", "response"],
) -> None:
    """Raise unless the guardrail evaluated every image and blocked none of them."""
    if not isinstance(guardrail, BedrockGuardrail) or guardrail.checks is not None:
        raise _not_scanned(f"Guardrail '{guardrail.guardrail_name}' does not support image scanning.")
    message: Final = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": image}} for image in images],
    }
    attachments: Final = find_request_attachments(
        {"messages": [message]}, "acompletion", skip_tool_messages=False, latest_user_message_only=False
    )
    if attachments.unscannable:
        raise _not_scanned(
            f"Guardrail '{guardrail.guardrail_name}' cannot scan this image: "
            f"{', '.join(sorted(frozenset(attachments.unscannable)))}."
        )
    is_response: Final = input_type == "response"
    answer: Final = await guardrail.async_scan_images(
        attachments.images,
        request_data=request_data,
        source="OUTPUT" if is_response else "INPUT",
        event_type=GuardrailEventHooks.post_call if is_response else GuardrailEventHooks.pre_call,
    )
    coverage: Final = answer.get("guardrailCoverage") or {}
    if coverage.get("images", {}).get("guarded", 0) < len(images):
        raise _not_scanned(
            f"Guardrail '{guardrail.guardrail_name}' did not report evaluating the image. "
            "Bedrock evaluates images only for content filters that have the Image modality enabled."
        )


def _not_scanned(detail: str) -> HTTPException:
    return HTTPException(status_code=422, detail=detail)
