import json
from typing import Any, List, Optional, Tuple

import httpx

from litellm.constants import STREAM_SSE_DONE_STRING
from litellm.llms.openai.common_utils import OpenAIError
from litellm.types.llms.openai import ResponsesAPIStreamEvents
from litellm.types.utils import ImageUsage, ImageUsageInputTokensDetails
from litellm.utils import CustomStreamWrapper


def extract_image_payloads(raw_response: httpx.Response) -> List[str]:
    content_type = raw_response.headers.get("content-type", "")
    body_text = raw_response.text or ""
    parsed_payloads: List[dict] = []

    if "text/event-stream" in content_type.lower() or looks_like_sse(body_text):
        parsed_payloads = parse_sse_payloads(body_text)
    else:
        try:
            response_json = raw_response.json()
        except Exception:
            response_json = {}
        if isinstance(response_json, dict):
            parsed_payloads = [response_json]

    images: List[str] = []
    partial_images: List[str] = []
    for payload in parsed_payloads:
        extracted_images, extracted_partial_images = extract_images_from_payload(
            payload
        )
        images.extend(extracted_images)
        partial_images.extend(extracted_partial_images)
    return dedupe(images) or dedupe(partial_images)


def extract_image_usage(raw_response: httpx.Response) -> Optional[ImageUsage]:
    parsed_payloads = get_parsed_payloads(raw_response)

    for payload in parsed_payloads:
        if payload.get("type") != ResponsesAPIStreamEvents.RESPONSE_COMPLETED:
            continue
        image_gen_usage = get_image_generation_usage(payload)
        if image_gen_usage is not None:
            return transform_image_usage(image_gen_usage)

    for payload in reversed(parsed_payloads):
        image_gen_usage = get_image_generation_usage(payload)
        if image_gen_usage is not None and not is_zero_image_usage(image_gen_usage):
            return transform_image_usage(image_gen_usage)
    return None


def get_parsed_payloads(raw_response: httpx.Response) -> List[dict]:
    content_type = raw_response.headers.get("content-type", "")
    body_text = raw_response.text or ""

    if "text/event-stream" in content_type.lower() or looks_like_sse(body_text):
        return parse_sse_payloads(body_text)

    try:
        response_json = raw_response.json()
    except Exception:
        response_json = {}
    if isinstance(response_json, dict):
        return [response_json]
    return []


def transform_image_usage(usage: dict) -> ImageUsage:
    input_tokens_details = usage.get("input_tokens_details") or {}
    return ImageUsage(
        input_tokens=usage.get("input_tokens", 0),
        input_tokens_details=ImageUsageInputTokensDetails(
            image_tokens=input_tokens_details.get("image_tokens", 0),
            text_tokens=input_tokens_details.get("text_tokens", 0),
        ),
        output_tokens=usage.get("output_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
    )


def get_image_generation_usage(response_payload: Any) -> Optional[dict]:
    if not isinstance(response_payload, dict):
        return None

    payloads_to_check: List[dict] = []
    current_payload = response_payload
    visited_container_ids = set()
    while isinstance(current_payload, dict):
        container_id = id(current_payload)
        if container_id in visited_container_ids:
            break
        visited_container_ids.add(container_id)
        payloads_to_check.append(current_payload)

        next_payload = current_payload.get("response")
        if not isinstance(next_payload, dict):
            break
        current_payload = next_payload

    for payload in reversed(payloads_to_check):
        image_gen_usage = _get_image_generation_usage_from_payload(payload)
        if image_gen_usage is not None:
            return image_gen_usage
    return None


def _get_image_generation_usage_from_payload(response_payload: dict) -> Optional[dict]:
    tool_usage = response_payload.get("tool_usage")
    if not isinstance(tool_usage, dict):
        return None

    image_gen_usage = tool_usage.get("image_gen")
    if not isinstance(image_gen_usage, dict):
        return None

    input_tokens = image_gen_usage.get("input_tokens")
    output_tokens = image_gen_usage.get("output_tokens")
    if input_tokens is None or output_tokens is None:
        return None

    normalized_usage = dict(image_gen_usage)
    if normalized_usage.get("total_tokens") is None:
        normalized_usage["total_tokens"] = input_tokens + output_tokens
    return normalized_usage


def is_zero_image_usage(usage: dict) -> bool:
    return (
        (usage.get("input_tokens") or 0) == 0
        and (usage.get("output_tokens") or 0) == 0
        and (usage.get("total_tokens") or 0) == 0
    )


def looks_like_sse(body_text: str) -> bool:
    trimmed_body = body_text.lstrip()
    return (
        trimmed_body.startswith("event:")
        or trimmed_body.startswith("data:")
        or "\nevent:" in body_text
        or "\ndata:" in body_text
    )


def parse_sse_payloads(body_text: str) -> List[dict]:
    payloads: List[dict] = []
    for line in body_text.splitlines():
        stripped_line = CustomStreamWrapper._strip_sse_data_from_chunk(line)
        if not stripped_line:
            continue
        stripped_line = stripped_line.strip()
        if not stripped_line or stripped_line == STREAM_SSE_DONE_STRING:
            continue
        try:
            parsed = json.loads(stripped_line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            payloads.append(parsed)
    return payloads


def extract_images_from_payload(payload: dict) -> Tuple[List[str], List[str]]:
    event_type = payload.get("type")
    if event_type in (
        ResponsesAPIStreamEvents.RESPONSE_FAILED,
        ResponsesAPIStreamEvents.ERROR,
    ):
        error_obj = payload.get("error") or (payload.get("response") or {}).get("error")
        raise OpenAIError(message=str(error_obj or payload), status_code=400)

    partial_images: List[str] = []
    if event_type in (
        ResponsesAPIStreamEvents.IMAGE_GENERATION_PARTIAL_IMAGE,
        "response.image_generation_call.partial_image",
    ):
        partial_image_b64 = payload.get("partial_image_b64")
        b64_json = payload.get("b64_json")
        if isinstance(partial_image_b64, str):
            partial_images.append(partial_image_b64)
        if isinstance(b64_json, str):
            partial_images.append(b64_json)
        return [], partial_images

    candidates: List[str] = []
    if event_type == "image_generation.completed":
        b64_json = payload.get("b64_json")
        if isinstance(b64_json, str):
            candidates.append(b64_json)

    response_payload = payload.get("response")
    if isinstance(response_payload, dict):
        candidates.extend(extract_images_from_nested_value(response_payload))

    candidates.extend(extract_images_from_nested_value(payload))
    return dedupe(candidates), dedupe(partial_images)


def extract_images_from_nested_value(value: Any) -> List[str]:
    images: List[str] = []
    values_to_visit = [value]
    visited_container_ids = set()

    while values_to_visit:
        current_value = values_to_visit.pop()
        if isinstance(current_value, dict):
            container_id = id(current_value)
            if container_id in visited_container_ids:
                continue
            visited_container_ids.add(container_id)

            value_type = current_value.get("type")
            if value_type in ("image_generation_call", "image_generation"):
                images.extend(get_image_strings_from_dict(current_value))
            elif isinstance(current_value.get("b64_json"), str):
                images.append(current_value["b64_json"])

            values_to_visit.extend(reversed(list(current_value.values())))
        elif isinstance(current_value, list):
            container_id = id(current_value)
            if container_id in visited_container_ids:
                continue
            visited_container_ids.add(container_id)

            values_to_visit.extend(reversed(current_value))
    return dedupe(images)


def get_image_strings_from_dict(value: dict) -> List[str]:
    images: List[str] = []
    for key in ("result", "b64_json", "image"):
        candidate = value.get(key)
        if isinstance(candidate, str):
            images.append(candidate)
        elif isinstance(candidate, list):
            images.extend(item for item in candidate if isinstance(item, str))
    return images


def dedupe(values: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
