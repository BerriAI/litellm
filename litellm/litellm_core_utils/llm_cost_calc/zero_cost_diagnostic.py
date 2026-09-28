from collections.abc import Mapping
from functools import reduce
from typing import Final

from pydantic import TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    PromptTokensDetailsWrapper,
    StandardLoggingZeroCostDiagnostic,
    Usage,
)

ZERO_COST_COUNTER_NAME: Final = "litellm_zero_cost_requests_total"

_TEXT_INPUT_RATE: Final = "input_cost_per_token"
_AUDIO_INPUT_RATE: Final = "input_cost_per_audio_token"
_TEXT_OUTPUT_RATE: Final = "output_cost_per_token"
_AUDIO_OUTPUT_RATE: Final = "output_cost_per_audio_token"
_RATE_KEY_MARKERS: Final = ("cost", "pricing")
_NESTED_PRICING: Final = TypeAdapter(Mapping[str, object] | tuple[object, ...])
_MAX_PRICING_DEPTH: Final = 4


def _audio_tokens(details: PromptTokensDetailsWrapper | CompletionTokensDetailsWrapper | None) -> int:
    audio_tokens: Final = details.audio_tokens if details is not None else None
    return audio_tokens if isinstance(audio_tokens, int) and audio_tokens > 0 else 0


def _tokens(value: object) -> int:
    return value if isinstance(value, int) and value > 0 else 0


def used_pricing_keys(usage: Usage) -> tuple[str, ...]:
    prompt_audio: Final = _audio_tokens(usage.prompt_tokens_details)
    completion_audio: Final = _audio_tokens(usage.completion_tokens_details)
    prompt_text: Final = _tokens(usage.prompt_tokens) - prompt_audio
    completion_text: Final = _tokens(usage.completion_tokens) - completion_audio
    components: Final = (
        (_TEXT_INPUT_RATE, prompt_text),
        (_AUDIO_INPUT_RATE, prompt_audio),
        (_TEXT_OUTPUT_RATE, completion_text),
        (_AUDIO_OUTPUT_RATE, completion_audio),
    )
    return tuple(key for key, count in components if count > 0)


def _nested_pricing(value: object) -> Mapping[str, object] | tuple[object, ...] | None:
    try:
        return _NESTED_PRICING.validate_python(value)
    except ValidationError:
        return None


def _is_rate_key(key: str) -> bool:
    return any(marker in key for marker in _RATE_KEY_MARKERS)


def _rate_values(value: object) -> tuple[object, ...]:
    nested: Final = _nested_pricing(value)
    if isinstance(nested, Mapping):
        return tuple(child for key, child in nested.items() if _is_rate_key(key))
    if nested is None:
        return (value,)
    return nested


def _expand_rate_values(values: tuple[object, ...], _depth: int) -> tuple[object, ...]:
    return tuple(nested for value in values for nested in _rate_values(value))


def _is_positive_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value > 0


def _declares_a_rate(pricing_entry: Mapping[str, object]) -> bool:
    leaves: Final = reduce(_expand_rate_values, range(_MAX_PRICING_DEPTH), (pricing_entry,))
    return any(_is_positive_number(leaf) for leaf in leaves)


def _is_explicit_zero(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value == 0


def diagnose_zero_cost(
    usage: Usage,
    pricing_model: str,
    pricing_entry: Mapping[str, object],
    calculation_failed: bool,
) -> StandardLoggingZeroCostDiagnostic | None:
    used_keys: Final = used_pricing_keys(usage)
    if not used_keys:
        return None
    missing_keys: Final = tuple(key for key in used_keys if pricing_entry.get(key) is None)
    if not missing_keys and all(_is_explicit_zero(pricing_entry[key]) for key in used_keys):
        return None
    if not _declares_a_rate(pricing_entry):
        return None
    if calculation_failed:
        return StandardLoggingZeroCostDiagnostic(
            reason="cost_calculation_error", pricing_model=pricing_model, missing_pricing_keys=()
        )
    if missing_keys:
        return StandardLoggingZeroCostDiagnostic(
            reason="missing_pricing_key", pricing_model=pricing_model, missing_pricing_keys=missing_keys
        )
    return StandardLoggingZeroCostDiagnostic(
        reason="pricing_not_applied", pricing_model=pricing_model, missing_pricing_keys=()
    )


def _cause(diagnostic: StandardLoggingZeroCostDiagnostic) -> str:
    reason: Final = diagnostic["reason"]
    match reason:
        case "missing_pricing_key":
            return (
                f"pricing entry '{diagnostic['pricing_model']}' has no {', '.join(diagnostic['missing_pricing_keys'])}. "
                "Set the missing rate in the deployment's model_info or in the model cost map, "
                "or set every rate to 0 to mark the model free"
            )
        case "pricing_not_applied":
            return (
                f"pricing entry '{diagnostic['pricing_model']}' declares non-zero rates for this usage, "
                "but the cost calculator returned $0"
            )
        case "cost_calculation_error":
            return (
                f"cost calculation raised for pricing entry '{diagnostic['pricing_model']}', "
                "see response_cost_failure_debug_information"
            )
        case _:
            return assert_never(reason)


def zero_cost_warning(
    diagnostic: StandardLoggingZeroCostDiagnostic,
    *,
    model_group: str | None,
    model: str,
    custom_llm_provider: str | None,
    usage: Usage,
) -> str:
    request: Final = (
        f"model_group={model_group or model} model={model} provider={custom_llm_provider or 'unknown'} "
        f"prompt_tokens={_tokens(usage.prompt_tokens)} completion_tokens={_tokens(usage.completion_tokens)}"
    )
    return (
        f"Billable request priced at $0 and logged as such ({request}): {_cause(diagnostic)}. "
        f'Counted in {ZERO_COST_COUNTER_NAME}{{reason="{diagnostic["reason"]}"}}'
    )
