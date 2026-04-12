#!/usr/bin/env python3
"""
Generate a TypeScript compliance-prompts data file from a CSV eval file.

Usage example:
    python generate_compliance_prompts.py \
      --csv ../../litellm/proxy/guardrails/guardrail_hooks/litellm_content_filter/guardrail_benchmarks/evals/block_insults.csv \
      --framework "Insults & Abuse" \
      --framework-icon "alert-triangle" \
      --framework-description "Detects insults, name-calling, and personal attacks — blocks abuse while allowing legitimate complaints." \
      --category "Insults & Personal Attacks" \
      --category-icon "alert-triangle" \
      --category-description "Detects insults, name-calling, and personal attacks directed at the chatbot, staff, or other people" \
      --output ../src/data/insultsCompliancePrompts.ts
"""

import argparse
import csv
import os
import sys


def escape_ts_string(s: str) -> str:
    """Escape a string for use inside a TypeScript double-quoted string literal."""
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    return s


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a TypeScript CompliancePrompt[] file from a CSV eval file."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the input CSV file (columns: prompt, expected_result, framework, category).",
    )
    parser.add_argument(
        "--framework",
        required=True,
        help='Framework display name, e.g. "Insults & Abuse".',
    )
    parser.add_argument(
        "--framework-icon",
        required=True,
        help='Lucide icon name for the framework, e.g. "alert-triangle".',
    )
    parser.add_argument(
        "--framework-description",
        required=True,
        help="One-line description of the framework.",
    )
    parser.add_argument(
        "--category",
        required=True,
        help='Category display name, e.g. "Insults & Personal Attacks".',
    )
    parser.add_argument(
        "--category-icon",
        required=True,
        help='Lucide icon name for the category, e.g. "alert-triangle".',
    )
    parser.add_argument(
        "--category-description",
        required=True,
        help="One-line description of the category.",
    )
    parser.add_argument(
        "--var-prefix",
        required=True,
        help='Prefix for exported variable names, e.g. "insults" -> insultsCompliancePrompts.',
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output .ts file.",
    )

    args = parser.parse_args()

    # --- Read CSV ---
    csv_path = os.path.abspath(args.csv)
    if not os.path.isfile(csv_path):
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("Error: CSV file is empty.", file=sys.stderr)
        sys.exit(1)

    # --- Derive variable names from --var-prefix ---
    prefix = args.var_prefix
    array_name = f"{prefix}CompliancePrompts"
    meta_name = f"{prefix}FrameworkMeta"

    # --- Build TypeScript source ---
    lines: list[str] = []

    # Header comment
    csv_basename = os.path.basename(args.csv)
    lines.append(
        f"// Auto-generated from {csv_basename} — do not edit manually."
    )
    lines.append(
        f"// Regenerate: python scripts/generate_compliance_prompts.py --csv ... --output ..."
    )
    lines.append("")
    lines.append(
        'import type { CompliancePrompt, ComplianceFramework } from "./compliancePrompts";'
    )
    lines.append("")
    lines.append(f"export const {array_name}: CompliancePrompt[] = [")

    for idx, row in enumerate(rows, start=1):
        prompt_text = escape_ts_string(row["prompt"].strip())
        expected = row["expected_result"].strip().lower()
        csv_category = row.get("category", "unknown").strip()
        prompt_id = f"{csv_category}-{idx}"

        lines.append("  {")
        lines.append(f'    id: "{prompt_id}",')
        lines.append(f'    framework: "{escape_ts_string(args.framework)}",')
        lines.append(f'    category: "{escape_ts_string(args.category)}",')
        lines.append(f'    categoryIcon: "{escape_ts_string(args.category_icon)}",')
        lines.append(
            f'    categoryDescription: "{escape_ts_string(args.category_description)}",'
        )
        lines.append(f'    prompt: "{prompt_text}",')
        lines.append(f'    expectedResult: "{expected}",')
        lines.append("  },")

    lines.append("];")
    lines.append("")
    lines.append(f"export const {meta_name} = {{")
    lines.append(f'  name: "{escape_ts_string(args.framework)}",')
    lines.append(f'  icon: "{escape_ts_string(args.framework_icon)}",')
    lines.append(
        f'  description: "{escape_ts_string(args.framework_description)}",'
    )
    lines.append("};")
    lines.append("")

    # --- Write output ---
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Generated {len(rows)} prompts -> {output_path}")


if __name__ == "__main__":
    main()
