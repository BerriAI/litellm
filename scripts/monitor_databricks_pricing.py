#!/usr/bin/env python3
"""Monitor Databricks Foundation Model Serving pricing pages and update LiteLLM's
model_prices_and_context_window.json + packaged backup when rates change.

Triggered daily by .github/workflows/monitor_databricks_pricing.yml.

Behavior:
- Fetches the two official Databricks pricing pages (HTML, JS-rendered price
  table). Parses the embedded price data rows via regex extraction of the
  DBU table (works with the current page markup; fails loudly otherwise).
- Applies the LiteLLM convention: USD = DBU * 0.07 per token.
- Updates entries for the monitored model set (see MONITORED below).
- If any monitored rate changed, writes BOTH files, prints a diff summary and
  exits 0 (so the workflow can create the PR). If nothing changed, exits 0
  with "NO_CHANGE" marker so the workflow skips PR creation.
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_MAP = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_MAP = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

DBU_TO_USD = 0.07

# Databricks Foundation Model Serving page (open models, incl. DeepSeek V4)
FMS_PAGE = "https://www.databricks.com/product/pricing/foundation-model-serving"
# Proprietary page (GPT/Claude/Gemini) — fetched but not used for the monitored
# monitorset; kept for future expansion.
PROPRIETARY_PAGE = "https://www.databricks.com/product/pricing/proprietary-foundation-model-serving"

# model_map key -> (name pattern in the DBU table row, )
# name pattern is the model label as it appears on the pricing page table.
MONITORED = {
    "databricks/databricks-deepseek-v4-flash-0731": "Deepseek V4 Flash (0731)",
    "databricks/databricks-deepseek-v4-pro-0813": "Deepseek V4 Pro (0813)",
}

# Context windows / output caps from Databricks Foundation Model APIs limits doc
# (kept in sync with what we know; only rates are refreshed by this script).
MODEL_FIXTURE = {
    "databricks/databricks-deepseek-v4-flash-0731": {
        "max_input_tokens": 200000,
        "max_output_tokens": 10000,
    },
    "databricks/databricks-deepseek-v4-pro-0813": {
        "max_input_tokens": 200000,
        "max_output_tokens": 4000,
    },
}


def fetch(url: str, max_bytes: int = 5_000_000) -> str:
    """Fetch page HTML; returns text. Raises on non-200."""
    req = urllib.request.Request(url, headers={"User-Agent": "litellm-price-monitor/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} fetching {url}")
        return resp.read(max_bytes + 1).decode("utf-8", errors="replace")


def parse_dbu_table(html: str) -> dict[str, tuple[float, float]]:
    rows: dict[str, tuple[float, float]] = {}
    for tr in re.findall(r"<tr>(.*?)</tr>", html, flags=re.S):
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, flags=re.S)
        if not cells:
            continue
        label = re.sub(r"<[^>]+>", "", cells[0]).strip()
        nums = []
        for c in cells[1:]:
            txt = re.sub(r"<[^>]+>", "", c).strip()
            if re.fullmatch(r"\d+(?:\.\d+)?", txt):
                nums.append(float(txt))
        if label and len(nums) >= 2:
            rows[label] = (nums[0], nums[1])

    results: dict[str, tuple[float, float]] = {}
    for label in MONITORED.values():
        if label not in rows:
            raise RuntimeError(f"Could not locate pricing row for '{label}' on {FMS_PAGE}")
        results[label] = rows[label]
    return results


def build_entry(model_key: str, input_dbu: float, output_dbu: float) -> dict:
    fx = MODEL_FIXTURE[model_key]
    return {
        "input_cost_per_token": input_dbu / 1_000_000 * DBU_TO_USD,
        "input_dbu_cost_per_token": input_dbu,
        "litellm_provider": "databricks",
        "max_input_tokens": fx["max_input_tokens"],
        "max_output_tokens": fx["max_output_tokens"],
        "max_tokens": fx["max_output_tokens"],
        "metadata": {
            "notes": (
                f"Pricing derived from Databricks Foundation Model Serving DBU rates "
                f"({input_dbu:g} in / {output_dbu:g} out DBU per 1M tokens × ${DBU_TO_USD:.2f}/DBU "
                f"= ${input_dbu * DBU_TO_USD:.2f}/${output_dbu * DBU_TO_USD:.2f} per 1M). "
                f"Auto-refreshed daily by monitor_databricks_pricing workflow."
            )
        },
        "mode": "chat",
        "output_cost_per_token": output_dbu / 1_000_000 * DBU_TO_USD,
        "output_dbu_cost_per_token": output_dbu,
        "source": FMS_PAGE,
        "supports_function_calling": True,
        "supports_reasoning": True,
        "supports_tool_choice": True,
    }


def main() -> int:
    html = fetch(FMS_PAGE)
    rates = parse_dbu_table(html)

    with MAIN_MAP.open() as f:
        main_data = json.load(f)
    with BACKUP_MAP.open() as f:
        backup_data = json.load(f)

    changed = False
    for model_key, label in MONITORED.items():
        in_dbu, out_dbu = rates[label]
        entry = build_entry(model_key, in_dbu, out_dbu)
        old = main_data.get(model_key)
        if old != entry:
            main_data[model_key] = entry
            backup_data[model_key] = entry
            changed = True
            sys.stdout.write(
                f"CHANGED {model_key}: {old and old.get('input_cost_per_token')} -> {entry['input_cost_per_token']}\n"
            )

    if not changed:
        sys.stdout.write("NO_CHANGE\n")
        return 0

    with MAIN_MAP.open("w") as f:
        json.dump(main_data, f, indent=4)
        f.write("\n")
    with BACKUP_MAP.open("w") as f:
        json.dump(backup_data, f, indent=4)
        f.write("\n")
    sys.stdout.write("WROTE updated model map and backup\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
