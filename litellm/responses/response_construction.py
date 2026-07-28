"""
Best-effort construction of Responses API objects from raw provider payloads.

Providers deviate from the OpenAI Responses spec often enough that strict
pydantic validation fails (missing ``output``, a ``usage`` block with unexpected
fields, ...). Callers used to fall back to ``model_construct``, which skips
validation *and* nested model coercion, so ``response.usage`` stayed a plain
dict and ``response.output`` could be missing entirely. Everything downstream
reads those attributes as declared (``ResponsesAPIResponse.usage`` is a
``ResponseAPIUsage``, ``.output`` is a list), so the fallback produced objects
that violate their own type and blew up with ``AttributeError`` mid-stream.

These helpers keep the declared shape intact even when validation fails, so the
fallback stays a degraded-data path instead of a crash path.
"""

from collections.abc import Callable, Mapping

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.types.llms.base import BaseLiteLLMOpenAIResponseObject
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse

_FIELD_MAPPING_ADAPTER = TypeAdapter(dict[str, object])


def construct_responses_api_response(payload: Mapping[str, object]) -> ResponsesAPIResponse:
    """Build a ``ResponsesAPIResponse``, validating when possible.

    On validation failure, fields required by the model are defaulted and
    ``usage`` is coerced, so attribute access on the result behaves as declared.
    """
    try:
        return ResponsesAPIResponse.model_validate(dict(payload))
    except ValidationError:
        verbose_logger.debug(
            "Responses API: validation failed for payload %s, falling back to model_construct",
            payload,
        )

    output = payload.get("output")
    construct: Callable[..., ResponsesAPIResponse] = ResponsesAPIResponse.model_construct
    return construct(
        **{
            **payload,
            "output": output if isinstance(output, list) else [],
            "usage": _construct_usage(payload.get("usage")),
        }
    )


def construct_responses_api_stream_event(
    event_model: type[BaseLiteLLMOpenAIResponseObject],
    payload: Mapping[str, object],
) -> BaseLiteLLMOpenAIResponseObject:
    """Build a Responses API streaming event without validation.

    Terminal events (``response.completed`` / ``.failed`` / ``.incomplete``)
    declare a ``ResponsesAPIResponse``; ``model_construct`` alone would leave it
    as the raw dict, so construct that nested object explicitly.
    """
    construct: Callable[..., BaseLiteLLMOpenAIResponseObject] = event_model.model_construct
    response_payload = _as_field_mapping(payload.get("response"))
    if response_payload is None:
        return construct(**payload)
    return construct(**{**payload, "response": construct_responses_api_response(response_payload)})


def _construct_usage(usage: object) -> ResponseAPIUsage | None:
    """Coerce a raw ``usage`` payload into ``ResponseAPIUsage``, or drop it."""
    if usage is None or isinstance(usage, ResponseAPIUsage):
        return usage
    usage_fields = _as_field_mapping(usage)
    if usage_fields is None:
        return None

    input_tokens = _as_token_count(usage_fields.get("input_tokens"))
    output_tokens = _as_token_count(usage_fields.get("output_tokens"))
    try:
        return ResponseAPIUsage.model_validate(
            {
                **usage_fields,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": _as_token_count(usage_fields.get("total_tokens")) or input_tokens + output_tokens,
            }
        )
    except ValidationError:
        verbose_logger.debug("Responses API: unparseable usage payload %s, dropping it", usage_fields)
        return None


def _as_field_mapping(value: object) -> Mapping[str, object] | None:
    """Validate an untyped payload into a string-keyed field mapping."""
    try:
        return _FIELD_MAPPING_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _as_token_count(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0
