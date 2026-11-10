"""One SDK session per case, in a fresh interpreter booted from the cost map at LITELLM_MODEL_COST_MAP_URL.

Each case does what SDK code does to ``litellm.model_cost`` and bills a real ``completion`` after every step, then
prints the charges as one JSON line. The steps use only names main exposes, so a case runs on main and the branch.
"""

import asyncio
import json
import os
import sys
from collections.abc import Callable
from typing import Final

import litellm
from litellm.litellm_core_utils.get_model_cost_map import (
    ModelCostMapReloadUnavailable,
    _finalize_model_cost_map,  # pyright: ignore[reportPrivateUsage]  # downstream code calls main's finalizer
    adopt_model_cost_map,
    refetch_model_cost_map,
)
from litellm.utils import (
    _cached_get_model_info_helper,  # pyright: ignore[reportPrivateUsage]  # downstream code clears it
    _invalidate_model_cost_lowercase_map,  # pyright: ignore[reportPrivateUsage]  # downstream code calls it
)

MODEL: Final = "lit-sdk-chat"
ALIAS: Final = "lit-sdk-chat-alias"
ADDED: Final = "lit-sdk-added"
FINETUNE: Final = f"ft:{MODEL}:org:custom:id"
CUSTOM_INPUT: Final = 9e-06
CUSTOM_OUTPUT: Final = 7e-06
BASE: Final = os.environ["MODEL_COST_SDK_BASE_URL"]
DOCUMENT_P2: Final = os.environ["MODEL_COST_SDK_DOCUMENT"]


def bill(model: str) -> float | None:
    response: Final = litellm.completion(
        model=f"openai/{model}",
        api_base=f"{BASE}/v1",
        api_key="sk-sdk-probe",
        messages=[{"role": "user", "content": "bill"}],
    )
    try:
        return float(litellm.completion_cost(completion_response=response))
    except Exception:  # noqa: BLE001  # an unpriced model is an observation here, not a failure
        return None


def reload(document: str) -> bool:
    result: Final = asyncio.run(refetch_model_cost_map(url=f"{BASE}/{document}"))
    if isinstance(result, ModelCostMapReloadUnavailable):
        return False
    adopt_model_cost_map(result.model_cost_map)
    return True


def row(scale: float) -> dict[str, object]:
    return {
        "litellm_provider": "openai",
        "mode": "chat",
        "input_cost_per_token": 1e-06 * scale,
        "output_cost_per_token": 2e-06 * scale,
        "cache_read_input_token_cost": 5e-07 * scale,
    }


def reload_reprices() -> list[object]:
    return [bill(MODEL), reload("p2.json"), bill(MODEL)]


def failed_reload_keeps_prices() -> list[object]:
    return [bill(MODEL), reload("missing.json"), bill(MODEL)]


def billed_and_listed(model: str) -> list[object]:
    info: Final = litellm.get_model_info(f"openai/{model}")
    return [bill(model), info["input_cost_per_token"], info["output_cost_per_token"]]


def every_name_bills_and_lists_the_row_across_a_reload() -> list[object]:
    names: Final = (MODEL, ALIAS, MODEL.upper())
    before: Final = [value for name in names for value in billed_and_listed(name)]
    return [*before, reload("p2.json"), *(value for name in names for value in billed_and_listed(name))]


def registration_survives_reload() -> list[object]:
    litellm.register_model({ADDED: {**row(3.0)}})
    litellm.register_model({"lit-sdk-request-scoped": row(3.0)}, persist_across_reloads=False)
    before: Final = [bill(ADDED), bill("lit-sdk-request-scoped")]
    return [*before, reload("p2.json"), bill(ADDED), bill("lit-sdk-request-scoped")]


def partial_registration_follows_reload() -> list[object]:
    litellm.register_model({MODEL: {"input_cost_per_token": CUSTOM_INPUT}})
    return [bill(MODEL), reload("p2.json"), bill(MODEL)]


def partial_registrations_merge_and_follow_reload() -> list[object]:
    litellm.register_model({MODEL: {"input_cost_per_token": CUSTOM_INPUT}})
    litellm.register_model({MODEL: {"output_cost_per_token": CUSTOM_OUTPUT}})
    return [bill(MODEL), reload("p2.json"), bill(MODEL)]


def finetune_registration_bills_the_same_after_a_reload() -> list[object]:
    litellm.register_model({FINETUNE: {"input_cost_per_token": CUSTOM_INPUT, "litellm_provider": "openai"}})
    before: Final = bill(FINETUNE)
    return [before is not None, reload("p1.json"), bill(FINETUNE) == before]


def finetune_registration_inherits_the_base_prices() -> list[object]:
    litellm.register_model({FINETUNE: {"input_cost_per_token": CUSTOM_INPUT, "litellm_provider": "openai"}})
    return [bill(FINETUNE)]


def new_row_after_a_miss_without_invalidating() -> list[object]:
    before: Final = bill(ADDED)
    litellm.model_cost[ADDED] = row(3.0)
    return [before, bill(ADDED)]


def value_edited_in_place_then_invalidate() -> list[object]:
    before: Final = bill(MODEL)
    litellm.model_cost[MODEL]["input_cost_per_token"] = CUSTOM_INPUT
    _invalidate_model_cost_lowercase_map()
    return [before, bill(MODEL)]


def table_swapped_then_invalidate() -> list[object]:
    before: Final = bill(MODEL)
    litellm.model_cost = {**litellm.model_cost, MODEL: row(2.0)}
    _invalidate_model_cost_lowercase_map()
    return [before, bill(MODEL)]


def table_swapped_then_helper_cache_cleared() -> list[object]:
    before: Final = bill(MODEL)
    litellm.model_cost = {**litellm.model_cost, MODEL: row(2.0)}
    _cached_get_model_info_helper.cache_clear()
    litellm.get_model_info.cache_clear()
    return [before, bill(MODEL)]


def contents_replaced_then_invalidate() -> list[object]:
    before: Final = bill(MODEL)
    saved: Final = {key: dict(value) for key, value in litellm.model_cost.items()}
    litellm.model_cost.clear()
    litellm.model_cost.update({**saved, MODEL: row(2.0)})
    _invalidate_model_cost_lowercase_map()
    return [before, bill(MODEL)]


def finalized_snapshot_replaces_contents() -> list[object]:
    with open(DOCUMENT_P2) as handle:
        finalized: Final = _finalize_model_cost_map(json.load(handle))
    litellm.model_cost.clear()
    litellm.model_cost.update(finalized)
    litellm.add_known_models(finalized)
    return [bill(MODEL)]


def alias_registration_after_an_adopted_reload() -> list[object]:
    reloaded: Final = reload("p2.json")
    litellm.register_model({ALIAS: {"input_cost_per_token": CUSTOM_INPUT}})
    return [reloaded, bill(MODEL), bill(ALIAS)]


def small_map_finalizes_and_leaves_the_live_map() -> list[object]:
    finalized: Final = _finalize_model_cost_map({ADDED: row(3.0)})
    return [list(finalized) == [ADDED], bill(MODEL)]


def popped_row_then_registered() -> list[object]:
    before: Final = bill(MODEL)
    litellm.model_cost.pop(MODEL)
    litellm.register_model({MODEL: row(3.0)})
    return [before, bill(MODEL)]


CASES: Final[dict[str, Callable[[], list[object]]]] = {
    case.__name__: case
    for case in (
        reload_reprices,
        failed_reload_keeps_prices,
        every_name_bills_and_lists_the_row_across_a_reload,
        registration_survives_reload,
        partial_registration_follows_reload,
        partial_registrations_merge_and_follow_reload,
        finetune_registration_bills_the_same_after_a_reload,
        finetune_registration_inherits_the_base_prices,
        new_row_after_a_miss_without_invalidating,
        value_edited_in_place_then_invalidate,
        table_swapped_then_invalidate,
        table_swapped_then_helper_cache_cleared,
        contents_replaced_then_invalidate,
        finalized_snapshot_replaces_contents,
        alias_registration_after_an_adopted_reload,
        small_map_finalizes_and_leaves_the_live_map,
        popped_row_then_registered,
    )
}

if __name__ == "__main__":
    sys.stdout.write(json.dumps(CASES[sys.argv[1]]()) + "\n")
