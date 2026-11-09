import itertools
import json
import re
import subprocess
from pathlib import Path
from typing import Final

from litellm.litellm_core_utils.llm_cost_calc.utils import _get_token_base_cost
from litellm.types.utils import Usage

REPO_ROOT: Final = Path(__file__).resolve().parents[4]
TOKEN_RATE_KEY: Final = re.compile(
    r"^(input_cost_per_token|output_cost_per_token|cache_read_input_token_cost"
    r"|cache_creation_input_token_cost(_above_1hr)?)"
    r"(_above_(\d+)(k?)_tokens)?(_(priority|flex|ultrafast|balanced|batches|decisions))?$"
)
SERVICE_TIERS: Final = ("priority", "flex", "ultrafast", "balanced")
LARGEST_PROMPT: Final = 10_000_000
INCLUSIVE_PROVIDERS: Final = frozenset({"xai"})


def token_rates(entry: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in entry.items()
        if value is not None and (key == "tiered_pricing" or TOKEN_RATE_KEY.match(key))
    }


def band_floors(info: dict[str, object]) -> frozenset[int]:
    matches: Final = (TOKEN_RATE_KEY.match(key) for key in info)
    flat: Final = frozenset(
        int(match.group(4)) * (1000 if match.group(5) else 1)
        for match in matches
        if match is not None and match.group(4) is not None
    )
    rows: Final = info.get("tiered_pricing")
    tiers: Final = (
        frozenset(int(row["range"][0]) for row in rows if isinstance(row, dict) and "range" in row)
        if isinstance(rows, list)
        else frozenset()
    )
    return flat | tiers


def tiers_in(info: dict[str, object]) -> tuple[str | None, ...]:
    return (None, *(tier for tier in SERVICE_TIERS if any(key.endswith(f"_{tier}") for key in info)))


def prompts(info: dict[str, object]) -> tuple[int, ...]:
    floors: Final = band_floors(info)
    return tuple(sorted(({1, LARGEST_PROMPT} | floors | {floor + 1 for floor in floors}) - {0}))


def check(info: dict[str, object], tier: str | None, prompt: int, inclusive: bool) -> list[object]:
    usage: Final = Usage(prompt_tokens=prompt, completion_tokens=1, total_tokens=prompt + 1)
    rates: Final = _get_token_base_cost(info, usage, service_tier=tier, threshold_is_inclusive=inclusive)
    one_hour: Final = rates.cache_creation_rates_by_ttl.get("1hr", rates.cache_creation_rate)
    return [tier, prompt, [rates.input_rate, rates.output_rate, rates.cache_creation_rate, one_hour, rates.cache_read_rate]]


def shape(info: dict[str, object], inclusive: bool) -> str:
    rows: Final = info.get("tiered_pricing")
    row_keys: Final = [sorted(row) for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    return json.dumps([sorted(info), row_keys, inclusive])


def checks(info: dict[str, object], inclusive: bool) -> list[list[object]]:
    return [check(info, tier, prompt, inclusive) for tier, prompt in itertools.product(tiers_in(info), prompts(info))]


def main() -> None:
    registry: Final = json.loads((REPO_ROOT / "model_prices_and_context_window.json").read_text())
    entries: Final = tuple(
        (token_rates(entry), entry.get("litellm_provider") in INCLUSIVE_PROVIDERS)
        for name, entry in registry.items()
        if name not in ("sample_spec", "fallback_generalizations") and isinstance(entry, dict)
    )
    by_shape: Final = {shape(info, inclusive): (info, inclusive) for info, inclusive in reversed(entries) if info}
    models: Final = tuple(
        json.dumps({"info": info, "inclusive": inclusive, "checks": checks(info, inclusive)})
        for info, inclusive in by_shape.values()
    )
    revision: Final = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True, cwd=REPO_ROOT).strip()
    Path(__file__).with_name("price_lookup_reference.json").write_text(
        f'{{"python_reference_commit": "{revision}", "models": [\n' + ",\n".join(models) + "\n]}\n"
    )


if __name__ == "__main__":
    main()
