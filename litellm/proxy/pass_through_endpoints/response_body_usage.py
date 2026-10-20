"""Token usage reported in the body of a generic pass-through response.

A pass-through target LiteLLM has no provider handler for still, most of the
time, returns a ``usage`` object next to its answer, in one of the two shapes
the industry settled on: OpenAI's ``prompt_tokens``/``completion_tokens`` or
Anthropic's ``input_tokens``/``output_tokens``. Without a handler those counts
were dropped, so every spend row and every trace for such a route showed 0
tokens even though the body said otherwise.
"""

from collections.abc import Mapping
from typing import Final

from pydantic import ConfigDict, TypeAdapter, ValidationError

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.utils import Usage

_PROMPT_TOKEN_KEYS: Final = ("prompt_tokens", "input_tokens")
_COMPLETION_TOKEN_KEYS: Final = ("completion_tokens", "output_tokens")
_TOTAL_TOKEN_KEYS: Final = ("total_tokens",)
_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))


def _json_object(value: object) -> Mapping[str, object] | None:
    try:
        return _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return None


def _token_count(usage: Mapping[str, object], keys: tuple[str, ...]) -> int | None:
    """The first usable count under ``keys``; ``None`` when the body reports none.

    ``bool`` is an ``int`` subclass, so it is ruled out explicitly: a target that
    answers ``"input_tokens": true`` is not reporting one token.
    """
    for key in keys:
        if isinstance(value := usage.get(key), int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def parse_response_body_usage(response_body: object) -> Usage | None:
    """Read the token counts a pass-through target reported in its JSON body.

    Returns ``None`` when the body is not a JSON object, carries no ``usage``
    object, or none of the known keys holds a non-negative integer. A missing
    side counts as 0 (a decisions API, for instance, legitimately reports
    ``output_tokens: 0``). An explicit ``total_tokens`` is kept as sent;
    otherwise the total is the sum of both sides.
    """
    body: Final = _json_object(response_body)
    if body is None:
        return None
    usage: Final = _json_object(body.get("usage"))
    if usage is None:
        return None
    prompt_tokens: Final = _token_count(usage, _PROMPT_TOKEN_KEYS)
    completion_tokens: Final = _token_count(usage, _COMPLETION_TOKEN_KEYS)
    if prompt_tokens is None and completion_tokens is None:
        return None
    reported_total: Final = _token_count(usage, _TOTAL_TOKEN_KEYS)
    prompt: Final = prompt_tokens or 0
    completion: Final = completion_tokens or 0
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=reported_total if reported_total is not None else prompt + completion,
    )


def apply_response_body_usage(logging_obj: LiteLLMLoggingObj, response_body: object) -> Usage | None:
    """Record the body-reported counts on the request's logging object.

    Runs after the upstream header contract has had its say: a
    ``combined_usage_object`` already on the logging object (an upstream that
    reported ``x-litellm-total-tokens``) is left untouched, since a target that
    reports its own totals owns them.
    """
    if logging_obj.model_call_details.get("combined_usage_object") is not None:
        return None
    usage: Final = parse_response_body_usage(response_body)
    if usage is None:
        return None
    logging_obj.model_call_details["combined_usage_object"] = usage  # rebind-ok: shared per-request logging record
    return usage
