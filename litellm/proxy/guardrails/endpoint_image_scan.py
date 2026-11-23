"""Image scanning for the standalone guardrail endpoint."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, NoReturn, TypeAlias

from fastapi import HTTPException
from typing_extensions import assert_never

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.bedrock.guardrail_attachments import find_request_attachments
from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
    BedrockGuardrail,
    bedrock_error_type,
    overrides_bedrock_call,
)
from litellm.types.guardrails import GuardrailEventHooks


@dataclass(frozen=True, slots=True)
class ImageNotScanned:
    detail: str


@dataclass(frozen=True, slots=True)
class GuardrailErrorBody:
    detail: str


ImageScanFailure: TypeAlias = ImageNotScanned | GuardrailErrorBody


async def scan_images(
    guardrail: CustomGuardrail,
    images: Sequence[str],
    request_data: dict[str, object],  # mutable-ok: the guardrail records its verdict on the request dict
    input_type: Literal["request", "response"],
) -> None:
    """Raise unless the guardrail evaluated every image and blocked none of them."""
    failure: Final = await _image_scan_failure(guardrail, images, request_data, input_type)
    if failure is not None:
        _raise_public(failure)


async def _image_scan_failure(
    guardrail: CustomGuardrail,
    images: Sequence[str],
    request_data: dict[str, object],  # mutable-ok: the guardrail records its verdict on the request dict
    input_type: Literal["request", "response"],
) -> ImageScanFailure | None:
    name: Final = guardrail.guardrail_name
    if not isinstance(guardrail, BedrockGuardrail) or guardrail.checks is not None or overrides_bedrock_call(guardrail):
        return ImageNotScanned(f"Guardrail '{name}' does not support image scanning.")
    message: Final = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": image}} for image in images],
    }
    attachments: Final = find_request_attachments(
        {"messages": [message]}, "acompletion", skip_tool_messages=False, latest_user_message_only=False
    )
    if attachments.unscannable:
        return ImageNotScanned(
            f"Guardrail '{name}' cannot scan this image: {', '.join(sorted(frozenset(attachments.unscannable)))}. "
            "Send it inline as a base64 PNG or JPEG data URL under 4 MB."
        )
    is_response: Final = input_type == "response"
    try:
        answer: Final = await guardrail.async_scan_images(
            attachments.images,
            request_data=request_data,
            source="OUTPUT" if is_response else "INPUT",
            event_type=GuardrailEventHooks.post_call if is_response else GuardrailEventHooks.pre_call,
        )
    except HTTPException as error:
        if error.status_code != 400 or isinstance(error.detail, dict):
            raise
        return ImageNotScanned(f"Guardrail '{name}' could not evaluate the image: {error.detail}")
    error_type: Final = bedrock_error_type(answer)
    if error_type is not None:
        return GuardrailErrorBody(f"Guardrail '{name}' returned {error_type} instead of a verdict on the image.")
    coverage: Final = answer.get("guardrailCoverage") or {}
    if coverage.get("images", {}).get("guarded", 0) < len(images):
        return ImageNotScanned(
            f"Guardrail '{name}' did not report evaluating the image. "
            "Bedrock evaluates images only for content filters that have the Image modality enabled."
        )
    return None


def _raise_public(failure: ImageScanFailure) -> NoReturn:
    match failure:
        case ImageNotScanned(detail=detail):
            raise HTTPException(status_code=422, detail=detail)
        case GuardrailErrorBody(detail=detail):
            raise HTTPException(status_code=502, detail=detail)
        case _:
            assert_never(failure)


def ignores_messages(guardrail: CustomGuardrail) -> bool:
    return isinstance(guardrail, BedrockGuardrail) and not guardrail.experimental_use_latest_role_message_only
