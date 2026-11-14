from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

Schema = Mapping[str, Mapping[str, str]]
MODELS: Final = re.compile(r"^[ \t]*model\s+(\w+)\s*\{(.*?)^[ \t]*\}", re.MULTILINE | re.DOTALL)
DROPPED_UNUSED_MODELS: Final = frozenset({"LiteLLM_AccountingDelivery", "LiteLLM_AccountingEvent"})
RENAMED_UNRELEASED_FIELDS: Final = frozenset(
    {
        ("LiteLLM_CustomerMemberTable", "budget_template_id"),
        ("LiteLLM_CustomerMemberTable", "budget_template"),
        ("LiteLLM_BudgetTable", "customer_member_templates"),
    }
)


def parse_model_fields(schema: str) -> Schema:
    def fields(body: str) -> Mapping[str, str]:
        parts: Final = tuple(line.split() for line in body.splitlines())
        return MappingProxyType({part[0]: " ".join(part[1:]) for part in parts if len(part) >= 2})

    return MappingProxyType({match[1]: fields(match[2]) for match in MODELS.finditer(schema)})


def _field_changes(model: str, field: str, old_type: str, new_fields: Mapping[str, str]) -> Iterator[str]:
    if field not in new_fields:
        if (model, field) in RENAMED_UNRELEASED_FIELDS:
            return
        yield f"Breaking: Field {model}.{field} was removed"
        return
    old_base: Final = old_type.split()[0]
    new_base: Final = new_fields[field].split()[0]
    if old_base.endswith("?") and not new_base.endswith("?"):
        yield f"Breaking: Field {model}.{field} changed from optional to required"
    if old_base.rstrip("?") != new_base.rstrip("?"):
        yield f"Breaking: Field {model}.{field} changed type from {old_base} to {new_base}"


def _model_changes(model: str, old_fields: Mapping[str, str], new_schema: Schema) -> Iterator[str]:
    if model not in new_schema:
        if model not in DROPPED_UNUSED_MODELS:
            yield f"Breaking: Model {model} was removed"
        return
    for field, old_type in old_fields.items():
        yield from _field_changes(model, field, old_type, new_schema[model])


def check_breaking_changes(old_schema: Schema, new_schema: Schema) -> tuple[str, ...]:
    def changes() -> Iterator[str]:
        for model, fields in old_schema.items():
            yield from _model_changes(model, fields, new_schema)

    return tuple(changes())


def _merge_base(base_ref: str) -> str:
    return subprocess.run(
        ("git", "merge-base", "HEAD", base_ref), capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> int:
    parser: Final = argparse.ArgumentParser()
    source: Final = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-ref", help="compare against the merge base of HEAD and this ref (needs history)")
    source.add_argument("--base-commit", help="compare against this commit as is (works in a shallow clone)")
    options: Final = TypeAdapter(dict[str, str | None]).validate_python(vars(parser.parse_args()))
    base_commit: Final = options["base_commit"]
    base: Final = base_commit if base_commit else _merge_base(options["base_ref"] or "")
    canonical: Final = subprocess.run(
        ("git", "show", f"{base}:litellm/schema.prisma"), capture_output=True, text=True, check=False
    )
    previous: Final = (
        canonical.stdout
        if canonical.returncode == 0
        else subprocess.run(("git", "show", f"{base}:schema.prisma"), capture_output=True, text=True, check=True).stdout
    )
    changes: Final = check_breaking_changes(
        parse_model_fields(previous), parse_model_fields(Path("litellm/schema.prisma").read_text())
    )
    if changes:
        sys.stderr.write("\n".join(changes) + "\n")
        return 1
    sys.stdout.write(f"Schema compatibility passed against merge base {base}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
