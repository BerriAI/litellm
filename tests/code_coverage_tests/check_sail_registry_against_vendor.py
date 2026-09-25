"""
Compare the ``sail/`` rows in model_prices_and_context_window.json against the
live Sail vendor: model ids from GET /v1/models, context windows parsed from the
400 error of an oversized chat request, and tier prices parsed from the aria
labels on https://docs.sailresearch.com/pricing.

Requires SAIL_API_KEY in the environment; exits 0 with a message when unset.
Exits 1 with a per-model diff table on any mismatch.
"""

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Final

import httpx

BASE_URL: Final = "https://api.sailresearch.com/v1"
PRICING_URL: Final = "https://docs.sailresearch.com/pricing"
COST_MAP_PATH: Final = Path(__file__).resolve().parent.parent.parent / "model_prices_and_context_window.json"

CONTEXT_WINDOW_RE: Final = re.compile(r"model context window of (\d+) tokens")
PRICING_LABEL_RE: Final = re.compile(
    r"aria-label=\"(.+?) (Default \(ASAP\)|Balanced|Flex) pricing: "
    r"input \$([\d.]+), cached \$([\d.]+), output \$([\d.]+) per 1M tokens\.\""
)
ARIA_LABEL_RE: Final = re.compile(r'aria-label="([^"]*)"')
COPY_PREFIX: Final = "Copy "

TIER_TO_SUFFIX: Final = {"Default (ASAP)": "", "Balanced": "_balanced", "Flex": "_flex"}
TIER_FIELDS: Final = (
    ("input_cost_per_token", 2),
    ("cache_read_input_token_cost", 3),
    ("output_cost_per_token", 4),
)


def sail_rows() -> dict[str, dict]:
    cost_map: Final = json.loads(COST_MAP_PATH.read_text())
    return {key.split("sail/", 1)[1]: row for key, row in cost_map.items() if key.startswith("sail/")}


def vendor_model_ids(client: httpx.Client, api_key: str) -> tuple[set[str], list[str]]:
    response: Final = client.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {api_key}"})
    if response.status_code != 200:
        return set(), [f"GET /v1/models -> {response.status_code}: {response.text[:200]}"]
    data: Final = response.json().get("data", [])
    return {m["id"] for m in data if isinstance(m, dict) and isinstance(m.get("id"), str)}, []


def vendor_context_windows(
    client: httpx.Client, api_key: str, model_ids: list[str]
) -> tuple[dict[str, int], list[str]]:
    windows: dict[str, int] = {}
    unverified: list[str] = []
    for model_id in model_ids:
        response: Final = client.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 10_000_000,
            },
            timeout=120,
        )
        if response.status_code == 503:
            unverified.append(model_id)
            continue
        match: Final = CONTEXT_WINDOW_RE.search(response.text)
        if match is None:
            unverified.append(f"{model_id} (status {response.status_code})")
            continue
        windows[model_id] = int(match.group(1))
    return windows, unverified


def vendor_prices(client: httpx.Client) -> dict[str, dict[str, dict[str, float]]]:
    text: Final = client.get(PRICING_URL, timeout=60).text
    labels: Final = ARIA_LABEL_RE.findall(text)
    prices: dict[str, dict[str, dict[str, float]]] = {}
    copy_labels: Final = tuple(
        (index, label[len(COPY_PREFIX) :]) for index, label in enumerate(labels) if label.startswith(COPY_PREFIX)
    )
    for index, label in enumerate(labels):
        match: Final = PRICING_LABEL_RE.search(f'aria-label="{label}"')
        if match is None:
            continue
        if not copy_labels:
            continue
        model_id: Final = min(copy_labels, key=lambda entry: abs(entry[0] - index))[1]
        tiers: Final = prices.setdefault(model_id, {})
        tiers[TIER_TO_SUFFIX[match.group(2)]] = {
            "input_cost_per_token": float(match.group(3)) / 1e6,
            "cache_read_input_token_cost": float(match.group(4)) / 1e6,
            "output_cost_per_token": float(match.group(5)) / 1e6,
        }
    return prices


def main() -> int:
    api_key: Final = os.environ.get("SAIL_API_KEY")
    if not api_key:
        print("SAIL_API_KEY not set; skipping sail vendor registry check")
        return 0

    rows: Final = sail_rows()
    diffs: list[str] = []

    with httpx.Client() as client:
        vendor_ids, errors = vendor_model_ids(client, api_key)
        diffs.extend(errors)
        for model_id in sorted(set(rows) - vendor_ids):
            diffs.append(f"{model_id}: in cost map but not in GET /v1/models")
        for model_id in sorted(vendor_ids - set(rows)):
            diffs.append(f"{model_id}: in GET /v1/models but no sail/ row in cost map")

        windows, unverified = vendor_context_windows(client, api_key, sorted(set(rows) & vendor_ids))
        for model_id, window in sorted(windows.items()):
            expected: Final = rows[model_id].get("max_input_tokens")
            if expected != window:
                diffs.append(f"{model_id}: context window cost map={expected} vendor={window}")

        prices: Final = vendor_prices(client)
        if not prices:
            diffs.append(f"no tier prices parsed from {PRICING_URL}")
        for model_id in sorted(set(rows) - set(prices)):
            diffs.append(f"{model_id}: sail/ row in cost map but no tier prices on {PRICING_URL}")
        for model_id, tiers in sorted(prices.items()):
            row: Final = rows.get(model_id)
            if row is None:
                continue
            for suffix, vendor_fields in tiers.items():
                for field in ("input_cost_per_token", "cache_read_input_token_cost", "output_cost_per_token"):
                    ours: Final = row.get(f"{field}{suffix}")
                    theirs: Final = vendor_fields[field]
                    if ours is None or not math.isclose(ours, theirs, rel_tol=1e-9):
                        diffs.append(f"{model_id}: {field}{suffix or ' (asap)'} cost map={ours} vendor={theirs}")

    if unverified:
        print("unverified models (no context window error or 503):", ", ".join(unverified))
    if diffs:
        print(f"{len(diffs)} sail registry mismatch(es):")
        for diff in diffs:
            print(f"  {diff}")
        return 1
    print(f"sail registry matches vendor: {len(rows)} rows checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
