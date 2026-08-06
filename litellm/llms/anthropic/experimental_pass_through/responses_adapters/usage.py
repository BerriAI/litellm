from typing import Final

from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicUsage
from litellm.types.llms.openai import ResponseAPIUsage


def _positive_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value > 0:
        return value
    return 0


def _field(usage: ResponseAPIUsage | dict[str, object] | None, field_name: str) -> object:
    if isinstance(usage, dict):
        return usage.get(field_name)
    return getattr(usage, field_name, None)


def _cached_input_tokens(usage: ResponseAPIUsage | dict[str, object] | None) -> int:
    details: Final = _field(usage, "input_tokens_details")
    if details is None:
        return 0
    if isinstance(details, dict):
        return _positive_int(details.get("cached_tokens"))
    return _positive_int(getattr(details, "cached_tokens", None))


def anthropic_usage_from_responses_usage(usage: ResponseAPIUsage | dict[str, object] | None) -> AnthropicUsage:
    """
    The Responses API counts cache hits inside ``input_tokens`` and reports them in
    ``input_tokens_details.cached_tokens``, while Anthropic clients read
    ``cache_read_input_tokens`` and expect ``input_tokens`` to exclude it.
    """
    input_tokens: Final = _positive_int(_field(usage, "input_tokens"))
    output_tokens: Final = _positive_int(_field(usage, "output_tokens"))
    cache_read_input_tokens: Final = min(_cached_input_tokens(usage), input_tokens)

    if cache_read_input_tokens > 0:
        return AnthropicUsage(
            input_tokens=input_tokens - cache_read_input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )
    return AnthropicUsage(input_tokens=input_tokens, output_tokens=output_tokens)
