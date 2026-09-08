#!/usr/bin/env python3
"""Monitor Databricks Foundation Model Serving pricing and update LiteLLM's
model_prices_and_context_window.json + packaged backup when rates change.

Triggered daily by .github/workflows/monitor_databricks_pricing.yml.

Behavior (registry-wide, not a fixed model list):
- Fetches both official pricing pages (open FMS + proprietary FMS) and parses
  every "Standard Pay Per Token" DBU table. Numeric columns are mapped by
  header text (Input / Output / Cache read / Cache write), so column
  reordering is safe. Priority/Batch/Provisioned tables are ignored.
- Compares page rates against every mapped registry entry and reports:
    UPDATED            - entry stores the published list rate and the page
                         value moved: rate fields refreshed in place.
    PROMO_SKIPPED      - entry stores the promotional rate (page list x 0.8);
                         left untouched, page value reported for a human.
    PROMO_ON_PAGE      - page displays a promotional price for an entry
                         storing the list rate; left untouched for a human.
    REVIEW             - stored rate matches neither pattern; manual check.
    RATES_AVAILABLE    - page publishes rates for an entry that has none.
    NOT_IN_REGISTRY    - page lists a mapped model the registry lacks.
    UNMAPPED_PAGE_MODEL- page lists a model with no mapping (new model?).
    MISSING_FROM_PAGE  - entry priced from these pages is no longer listed.
- Long-context tier rows and per-modality sub-rows (image/audio tokens) are
  skipped; only rate fields are touched, all other entry metadata is kept.
- If any entry was UPDATED, both JSON files are rewritten and a report is
  written to /tmp/dbx_monitor_pr_body.md for the workflow's PR body.
  Otherwise "NO_CHANGE" is printed so the workflow skips PR creation.
  Exit code is always 0.
"""

import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_MAP = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_MAP = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"
PR_BODY_PATH = Path("/tmp/dbx_monitor_pr_body.md")

DBU_TO_USD = 0.07
REL_TOL = 1.5e-3  # page rates carry 3-decimal rounding noise
PROMO_RATIO = 0.8
PROMO_TOL = 0.02

FMS_PAGE = "https://www.databricks.com/product/pricing/foundation-model-serving"
PROPRIETARY_PAGE = (
    "https://www.databricks.com/product/pricing/proprietary-foundation-model-serving"
)
PAGES = (FMS_PAGE, PROPRIETARY_PAGE)

# Header names (lowercased) of the numeric columns we consume. "cache write
# (1hr)" tiers are not modeled by LiteLLM and are dropped.
NUMERIC_HEADERS = ("input", "output", "cache read", "cache write")
DROPPED_HEADERS = ("cache write (1hr)",)

# Rowspan continuation rows on the proprietary page carry tier/modality
# qualifiers in the first cell; they never start a new model row.
QUALIFIER_LABELS = ("long context", "image tokens", "audio tokens", "in-geo", "global")
USABLE_QUALIFIERS = ("", "short context", "text tokens")

# Page label (markup stripped, lowercased) -> registry keys under
# "databricks/". Labels missing here are reported as UNMAPPED_PAGE_MODEL
# (new-model signal); keys missing from the registry are reported as
# NOT_IN_REGISTRY.
LABEL_TO_KEYS: Dict[str, List[str]] = {
    # Open Foundation Model Serving page
    "kimi k3": ["databricks-kimi-k3"],
    "glm-5.2, 5.3": ["databricks-glm-5-2", "databricks-glm-5-3"],
    "deepseek v4 pro": ["databricks-deepseek-v4-pro-0813"],
    "inkling": ["databricks-inkling"],
    "glm-5.3 flash": ["databricks-glm-5-3-flash"],
    "deepseek v4 flash": ["databricks-deepseek-v4-flash-0731"],
    "qwen 3.5 122b": ["databricks-qwen35-122b-a10b"],
    "llama 4 maverick": ["databricks-llama-4-maverick"],
    "llama 3.3 70b": ["databricks-meta-llama-3-3-70b-instruct"],
    "qwen 3 80b instruct": ["databricks-qwen3-next-80b-a3b-instruct"],
    "gpt-oss-120b": ["databricks-gpt-oss-120b"],
    "gemma 3 12b": ["databricks-gemma-3-12b"],
    "llama 3.1 8b": ["databricks-meta-llama-3-1-8b-instruct"],
    "gpt-oss-20b": ["databricks-gpt-oss-20b"],
    "gte": ["databricks-gte-large-en"],
    "bge large": ["databricks-bge-large-en"],
    "qwen 3 0.6b embedding": ["databricks-qwen3-embedding-0-6b"],
    # Proprietary Foundation Model Serving page
    "gpt-5.6 sol": ["databricks-gpt-5-6-sol"],
    "gpt-5.6 terra": ["databricks-gpt-5-6-terra"],
    "gpt-5.6 luna": ["databricks-gpt-5-6-luna"],
    "gpt-5.5": ["databricks-gpt-5-5"],
    "gpt-5.4 pro, 5.5 pro": ["databricks-gpt-5-5-pro"],
    "gpt-5.4": ["databricks-gpt-5-4"],
    "gpt-5.4 mini": ["databricks-gpt-5-4-mini"],
    "gpt-5.4 nano": ["databricks-gpt-5-4-nano"],
    "gpt-5.2 codex, 5.3 codex": ["databricks-gpt-5-2-codex", "databricks-gpt-5-3-codex"],
    "gpt-5.2": ["databricks-gpt-5-2"],
    "gpt-5, 5.1": ["databricks-gpt-5", "databricks-gpt-5-1"],
    "gpt-5.1 codex max": ["databricks-gpt-5-1-codex-max"],
    "gpt-5.1 codex mini": ["databricks-gpt-5-1-codex-mini"],
    "gpt-5 mini": ["databricks-gpt-5-mini"],
    "gpt-5 nano": ["databricks-gpt-5-nano"],
    "claude fable 5.1": ["databricks-claude-fable-5-1"],
    "claude fable 5": ["databricks-claude-fable-5"],
    "claude opus 4.5, 4.6, 4.7, 4.8, 5": [
        "databricks-claude-opus-4-5",
        "databricks-claude-opus-4-6",
        "databricks-claude-opus-4-7",
        "databricks-claude-opus-4-8",
        "databricks-claude-opus-5",
    ],
    "claude opus 4, 4.1": ["databricks-claude-opus-4", "databricks-claude-opus-4-1"],
    "claude sonnet 5": ["databricks-claude-sonnet-5"],
    "claude sonnet 4.5, 4.6": [
        "databricks-claude-sonnet-4-5",
        "databricks-claude-sonnet-4-6",
    ],
    "claude sonnet 4": ["databricks-claude-sonnet-4", "databricks-claude-sonnet-4-1"],
    "claude haiku 4.5": ["databricks-claude-haiku-4-5"],
    "gemini 3.0 pro, 3.1 pro": ["databricks-gemini-3-1-pro", "databricks-gemini-3-pro"],
    "gemini 2.5 pro": ["databricks-gemini-2-5-pro"],
    "gemini 3.7 flash, 3.8 flash": [
        "databricks-gemini-3-7-flash",
        "databricks-gemini-3-8-flash",
    ],
    "gemini 3.6 flash": ["databricks-gemini-3-6-flash"],
    "gemini 3.5 flash": ["databricks-gemini-3-5-flash"],
    "gemini 3.0 flash": ["databricks-gemini-3-flash"],
    "gemini 2.5 flash": ["databricks-gemini-2-5-flash"],
    "gemini 3.5 flash lite": ["databricks-gemini-3-5-flash-lite"],
    "gemini 3.1 flash lite": ["databricks-gemini-3-1-flash-lite"],
    "gemini 3 pro image": ["databricks-gemini-3-pro-image"],
    "gemini 3.1 flash image": ["databricks-gemini-3-1-flash-image"],
    "grok 4.6": ["databricks-grok-4-6"],
}

# Page labels with no text-token registry mapping today: image-generation
# models bill per-image/vendor pass-through, and Kimi K2.7 is not in the
# supported-models docs. Remove from here once entries exist.
IGNORED_LABELS = (
    "kimi k2.7",
    "gpt image 1",
    "gpt image 1 mini",
    "gpt image 1.5",
    "gpt image 2",
    "gemini 3.1 flash lite image",
)

# Rate fields an entry stores for a page-published column.
FIELD_BY_HEADER = {
    "input": ("input_cost_per_token", "input_dbu_cost_per_token"),
    "output": ("output_cost_per_token", "output_dbu_cost_per_token"),
    "cache read": ("cache_read_input_token_cost", None),
    "cache write": ("cache_creation_input_token_cost", None),
}


def fetch(url: str, max_bytes: int = 5_000_000) -> str:
    """Fetch page HTML; returns text. Raises on non-200."""
    req = urllib.request.Request(url, headers={"User-Agent": "litellm-price-monitor/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        if resp.status != 200:
            raise RuntimeError("HTTP {} fetching {}".format(resp.status, url))
        return resp.read(max_bytes + 1).decode("utf-8", errors="replace")


def _cell_text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html)).strip()


def _parse_number(cell: str) -> Optional[float]:
    txt = cell.replace(",", "").strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", txt):
        return float(txt)
    return None


def _is_placeholder(cell: str) -> bool:
    """True for dash/n/a cells meaning 'not published' - neither number nor label."""
    return cell.strip() in ("-", "—", "–", "n/a", "N/A", "na")


def parse_standard_pp_token_tables(
    html: str,
) -> Dict[str, List[Tuple[str, Tuple[Optional[float], ...], Tuple[str, ...]]]]:
    """Extract rows of every "Standard Pay Per Token" table.

    Returns label -> list of (qualifier, numbers, numeric_cols), where numbers
    align with that table's numeric header order.
    """
    parsed: Dict[str, List[Tuple[str, Tuple[Optional[float], ...], Tuple[str, ...]]]] = {}
    for tm in re.finditer(r"<table[^>]*>(.*?)</table>", html, re.S):
        table = tm.group(1)
        head = re.search(r"<thead>(.*?)</thead>", table, re.S)
        if head is None:
            continue
        header_texts = [
            _cell_text(th) for th in re.findall(r"<th[^>]*>(.*?)</th>", head.group(1), re.S)
        ]
        if not any("Standard Pay Per Token" in h for h in header_texts):
            continue
        numeric_cols = tuple(
            h.lower()
            for h in header_texts
            if h.lower() in NUMERIC_HEADERS and h.lower() not in DROPPED_HEADERS
        )
        for rm in re.finditer(r"<tr>(.*?)</tr>", table, re.S):
            cells = [
                _cell_text(c)
                for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", rm.group(1), re.S)
            ]
            if not cells:
                continue
            label = re.sub(r"[\*⌖]+", "", cells[0]).strip().lower()
            if not label or label == "model" or label in header_texts:
                continue
            if label in QUALIFIER_LABELS:
                continue  # rowspan continuation row (tier / modality rate)
            qualifier = ""
            numbers: List[Optional[float]] = []
            for cell in cells[1:]:
                if _is_placeholder(cell):
                    numbers.append(None)
                    continue
                num = _parse_number(cell)
                if num is not None:
                    numbers.append(num)
                elif cell and not qualifier:
                    qualifier = cell.lower()
            if len(numbers) < 1 or qualifier not in USABLE_QUALIFIERS:
                continue  # label-only rows / single-metric or modality rows
            numbers.extend([None] * (len(numeric_cols) - len(numbers)))
            parsed.setdefault(label, []).append(
                (qualifier, tuple(numbers[: len(numeric_cols)]), numeric_cols)
            )
    return parsed


def dbu_to_usd(dbu: Optional[float]) -> Optional[float]:
    if dbu is None:
        return None
    return dbu / 1_000_000 * DBU_TO_USD


def _approx(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= max(abs(b) * REL_TOL, 1e-12)


def classify(entry: Dict, page_in: float, page_out: Optional[float]) -> str:
    """How the entry's stored rates relate to the page row."""
    stored_in = entry.get("input_cost_per_token")
    if stored_in is None:
        return "unpriced"
    usd_in = dbu_to_usd(page_in)
    usd_out = dbu_to_usd(page_out) if page_out is not None else None
    if usd_in is None:
        return "no-input-column"
    if _approx(stored_in, usd_in) and (
        usd_out is None or _approx(entry.get("output_cost_per_token"), usd_out)
    ):
        return "list"
    if usd_in and abs(stored_in / usd_in - PROMO_RATIO) <= PROMO_TOL:
        stored_out = entry.get("output_cost_per_token")
        if (
            usd_out is None
            or stored_out is None
            or abs(stored_out / usd_out - PROMO_RATIO) <= PROMO_TOL
        ):
            return "promo"
    if usd_in and abs(usd_in / stored_in - PROMO_RATIO) <= PROMO_TOL:
        return "page-promo"
    return "mismatch"


def _refresh_cache_fields(
    entry: Dict, header_values: Dict[str, Optional[float]]
) -> Tuple[bool, List[str]]:
    """Sync cache rates with the page; keep n/a conventions tracking input."""
    notes: List[str] = []
    changed = False
    input_usd = entry.get("input_cost_per_token")
    for header, field in (
        ("cache read", "cache_read_input_token_cost"),
        ("cache write", "cache_creation_input_token_cost"),
    ):
        page_dbu = header_values.get(header)
        if page_dbu is not None:
            target = dbu_to_usd(page_dbu)
            note = "{} DBU {}".format(field, page_dbu)
        elif input_usd and (
            entry.get(field) is None or _approx(entry.get(field), input_usd)
        ):
            # not on the page: bill cache at input; custom conventions (gemini 0.1x) fall through
            target = input_usd
            note = "{}=input (not published)".format(field)
        else:
            continue
        if target is not None and not _approx(entry.get(field), target):
            notes.append("{}: {} -> {}".format(note, entry.get(field), target))
            entry[field] = target
            changed = True
    return changed, notes


def update_entry(
    entry: Dict,
    key: str,
    numbers: Tuple[Optional[float], ...],
    numeric_cols: Tuple[str, ...],
    page_url: str,
) -> Tuple[bool, str]:
    """Apply page rates to one registry entry. Returns (changed, report line)."""
    header_values = dict(zip(numeric_cols, numbers))
    page_in = header_values.get("input")
    page_out = header_values.get("output")
    if page_in is None:
        return False, "REVIEW {}: page row has no input column".format(key)
    status = classify(entry, page_in, page_out)
    if status == "promo":
        return False, (
            "PROMO_SKIPPED {}: entry stores the promotional rate; page list input={} output={}".format(
                key, page_in, page_out
            )
        )
    if status == "page-promo":
        return False, (
            "PROMO_ON_PAGE {}: page shows promotional pricing (input={}); "
            "entry keeps list rate {}".format(
                key, page_in, entry.get("input_cost_per_token")
            )
        )
    if status == "mismatch":
        stored = entry.get("input_cost_per_token")
        ratio = stored / dbu_to_usd(page_in) if stored and page_in else 0
        return False, (
            "REVIEW {}: stored input={} vs page list input={} DBU (ratio {:.3f}) "
            "- manual check".format(key, stored, page_in, ratio)
        )
    changed = False
    detail: List[str] = []
    for header in ("input", "output"):
        page_dbu = header_values.get(header)
        if page_dbu is None:
            continue  # e.g. embeddings: output not published - preserve stored
        usd_field, dbu_field = FIELD_BY_HEADER[header]
        target_usd = dbu_to_usd(page_dbu)
        if not _approx(entry.get(usd_field), target_usd):
            detail.append("{} {}->{}".format(header, entry.get(usd_field), target_usd))
            entry[usd_field] = target_usd
            changed = True
        if dbu_field is not None:
            target_dbu = page_dbu / 1_000_000
            if not _approx(entry.get(dbu_field), target_dbu):
                entry[dbu_field] = target_dbu
                changed = True
    cache_changed, cache_notes = _refresh_cache_fields(entry, header_values)
    detail.extend(cache_notes)
    changed = changed or cache_changed
    if changed and entry.get("source") != page_url:
        entry["source"] = page_url
    if changed:
        return True, "UPDATED {}: {}".format(key, "; ".join(detail))
    return False, ""


def main() -> int:
    pages = [(url, fetch(url)) for url in PAGES]

    with MAIN_MAP.open() as f:
        main_data = json.load(f)
    with BACKUP_MAP.open() as f:
        backup_data = json.load(f)

    tracked_keys: set = set()
    changed = False
    report_lines: List[str] = []

    for page_url, html in pages:
        tables = parse_standard_pp_token_tables(html)
        for label, rows in sorted(tables.items()):
            if label in IGNORED_LABELS:
                continue
            keys = LABEL_TO_KEYS.get(label)
            qualifier, numbers, numeric_cols = rows[0]
            if keys is None:
                report_lines.append(
                    "UNMAPPED_PAGE_MODEL: '{}' on {} lists input={} output={} DBU/1M "
                    "- add a mapping or a registry entry".format(
                        label, Path(page_url).name, numbers[0], numbers[1]
                    )
                )
                continue
            for key in keys:
                full_key = "databricks/" + key
                tracked_keys.add(full_key)
                entry = main_data.get(full_key)
                if entry is None:
                    report_lines.append(
                        "NOT_IN_REGISTRY {}: '{}' on {} lists rates ({} DBU in) "
                        "but the registry has no entry".format(
                            full_key, label, Path(page_url).name, numbers[0]
                        )
                    )
                    continue
                if not entry.get("input_cost_per_token"):
                    report_lines.append(
                        "RATES_AVAILABLE {}: '{}' now publishes rates ({} DBU in / {} out) "
                        "- entry currently unpriced".format(
                            full_key, label, numbers[0], numbers[1]
                        )
                    )
                    continue
                was_changed, line = update_entry(
                    entry, full_key, numbers, numeric_cols, page_url
                )
                changed = changed or was_changed
                if line:
                    report_lines.append(line)

    # Entries priced from these pages that vanished from them: retirement signal.
    for full_key, entry in main_data.items():
        if not full_key.startswith("databricks/") or full_key in tracked_keys:
            continue
        if entry.get("input_cost_per_token") and entry.get("source") in PAGES:
            report_lines.append(
                "MISSING_FROM_PAGE {}: priced entry no longer on the pricing pages "
                "- check for retirement".format(full_key)
            )

    if not changed:
        sys.stdout.write("NO_CHANGE\n")
        for line in report_lines:
            sys.stdout.write(line + "\n")
        return 0

    for full_key, entry in main_data.items():
        if full_key.startswith("databricks/") and full_key in backup_data:
            backup_data[full_key] = entry

    with MAIN_MAP.open("w") as f:
        json.dump(main_data, f, indent=4)
        f.write("\n")
    with BACKUP_MAP.open("w") as f:
        json.dump(backup_data, f, indent=4)
        f.write("\n")

    body = [
        "Automated daily check of the Databricks Foundation Model Serving pricing pages",
        "([open](https://www.databricks.com/product/pricing/foundation-model-serving),",
        "[proprietary](https://www.databricks.com/product/pricing/proprietary-foundation-model-serving))",
        "detected published-rate changes. Rate fields refreshed in place; metadata untouched.",
        "",
        "## Monitor report",
        "",
        "```",
    ]
    body.extend(report_lines)
    body.append("```")
    PR_BODY_PATH.write_text("\n".join(body) + "\n")

    sys.stdout.write("CHANGED\n")
    for line in report_lines:
        sys.stdout.write(line + "\n")
    sys.stdout.write(
        "WROTE updated model map, backup and PR body to {}\n".format(PR_BODY_PATH)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
