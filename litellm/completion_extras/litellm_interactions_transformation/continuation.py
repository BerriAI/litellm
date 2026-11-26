from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import chain
from typing import Final, TypeAlias

import httpx
from pydantic import JsonValue, TypeAdapter

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import as_object_list
from litellm.litellm_core_utils.env_utils import get_env_int_or_none
from litellm.llms.vertex_ai.gemini.long_decode import DEFAULT_OUTPUT_BUDGET_ENV
from litellm.responses.bridge_check import str_keyed_mapping

JsonMapping: TypeAlias = Mapping[str, JsonValue]

HOP_END_STATUS: Final = "requires_continuation"
CUT_OFF_STATUSES: Final = frozenset({"incomplete", HOP_END_STATUS})


def _token_of(interaction: JsonMapping) -> str | None:
    token: Final = interaction.get("continuation_token")
    return token if isinstance(token, str) and token else None


def client_continuation_token(interaction: JsonMapping) -> str | None:
    return _token_of(interaction) if interaction.get("status") in CUT_OFF_STATUSES else None


def resends_input_on_resume(custom_llm_provider: str) -> bool:
    return custom_llm_provider == "gemini"


def resumable_body(
    request_body: Mapping[str, object], *, resends_input: bool
) -> dict[str, object]:  # mutable-ok: JSON request body passed to HTTPHandler.post
    resumes_stored_turn: Final = bool(request_body.get("continuation_token")) and bool(
        request_body.get("previous_interaction_id")
    )
    if resends_input or not resumes_stored_turn:
        return dict(request_body)
    return {key: value for key, value in request_body.items() if key != "input"}


def _asks_for_media(response_format: object) -> bool:
    return any(
        (str_keyed_mapping(entry) or {}).get("type") in ("image", "audio")
        for entry in as_object_list(response_format) or (response_format,)
    )


def with_default_output_budget(
    request_body: Mapping[str, object], model_max_output_tokens: Callable[[], int | None]
) -> Mapping[str, object]:
    budget: Final = get_env_int_or_none(DEFAULT_OUTPUT_BUDGET_ENV)
    generation_config: Final = str_keyed_mapping(request_body.get("generation_config", {}))
    if budget is None or budget <= 0 or generation_config is None:
        return request_body
    if "max_output_tokens" in generation_config or request_body.get("continuation_token"):
        return request_body
    if request_body.get("response_modalities") not in (None, ["text"], ["TEXT"]):
        return request_body
    if _asks_for_media(request_body.get("response_format")):
        return request_body
    model_max: Final = model_max_output_tokens()
    if model_max is None:
        return request_body
    return {**request_body, "generation_config": {**generation_config, "max_output_tokens": min(budget, model_max)}}


def continuation_cap_of(request_body: Mapping[str, object]) -> int | None:
    generation_config: Final = str_keyed_mapping(request_body.get("generation_config"))
    cap: Final = generation_config.get("max_output_tokens") if generation_config is not None else None
    return cap if isinstance(cap, int) else None


def output_tokens_of(interaction: JsonMapping) -> int:
    usage: Final = interaction.get("usage")
    counts: Final = (
        tuple(usage.get(key) for key in ("total_output_tokens", "total_thought_tokens"))
        if isinstance(usage, dict)
        else ()
    )
    return sum(count for count in counts if isinstance(count, int) and not isinstance(count, bool))


def next_hop_body(
    request_body: Mapping[str, object],
    interaction: JsonMapping,
    produced_before: int,
    *,
    resends_input: bool,
) -> dict[str, object] | None:  # mutable-ok: JSON request body passed to HTTPHandler.post
    token: Final = _token_of(interaction)
    output_cap: Final = continuation_cap_of(request_body)
    hop_tokens: Final = output_tokens_of(interaction)
    if interaction.get("status") != HOP_END_STATUS or token is None or output_cap is None:
        return None
    if hop_tokens <= 0 or produced_before + hop_tokens >= output_cap:
        return None
    interaction_id: Final = interaction.get("id")
    return resumable_body(
        {
            **request_body,
            "continuation_token": token,
            **(
                {"previous_interaction_id": interaction_id}
                if isinstance(interaction_id, str) and interaction_id and request_body.get("store") is not False
                else {}
            ),
        },
        resends_input=resends_input,
    )


def _summed(values: Sequence[JsonValue]) -> JsonValue:
    if all(isinstance(value, list) for value in values):
        return list(chain.from_iterable(value for value in values if isinstance(value, list)))
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return sum(value for value in values if isinstance(value, int))
    return values[-1]


def merge_hop_usage(usages: Sequence[JsonValue]) -> JsonValue:
    hop_usages: Final = tuple(usage for usage in usages if isinstance(usage, dict))
    if len(hop_usages) <= 1:
        return hop_usages[0] if hop_usages else None
    keys: Final = tuple(dict.fromkeys(chain.from_iterable(hop_usages)))
    summed: Final = {key: _summed(tuple(usage[key] for usage in hop_usages if key in usage)) for key in keys}
    first: Final = hop_usages[0]
    return {
        **summed,
        **({"initial_input_tokens": first["initial_input_tokens"]} if "initial_input_tokens" in first else {}),
    }


def merge_hops(hops: Sequence[JsonMapping]) -> JsonMapping:
    last: Final = hops[-1]
    if len(hops) == 1:
        return last
    hop_steps: Final = tuple(hop.get("steps") for hop in hops)
    steps: Final = list(chain.from_iterable(steps for steps in hop_steps if isinstance(steps, list)))
    merged: Final = {**last, "steps": steps, "usage": merge_hop_usage(tuple(hop.get("usage") for hop in hops))}
    first_id: Final = hops[0].get("id")
    return {**merged, "id": first_id} if first_id else merged


def initial_input_tokens_of(hops: Sequence[JsonMapping]) -> int | None:
    first_usage: Final = hops[0].get("usage")
    count: Final = first_usage.get("total_input_tokens") if isinstance(first_usage, dict) else None
    return count if len(hops) > 1 and isinstance(count, int) else None


_JSON_VALUE: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


class RecitationBlockedHop(ValueError):
    pass


def _is_recitation_body(body: str) -> bool:
    try:
        parsed: Final = _JSON_VALUE.validate_json(body)
    except ValueError:
        return False
    error: Final = parsed.get("error") if isinstance(parsed, dict) else None
    return isinstance(error, dict) and error.get("code") in ("recitation", "safety")


def is_recitation_block(error: Exception) -> bool:
    if isinstance(error, RecitationBlockedHop):
        return True
    if not isinstance(error, httpx.HTTPStatusError):
        return False
    try:
        return _is_recitation_body(error.response.text)
    except httpx.StreamError:
        return False


def log_failed_hop(error: Exception | str) -> None:
    verbose_logger.warning(
        "Gemini Interactions long decode: a continuation hop failed, returning the answer so far with its continuation_token: %s",
        error,
    )
