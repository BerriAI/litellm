from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import BaseModel, ConfigDict, TypeAdapter

Schema: TypeAlias = Mapping[str, Mapping[str, str]]
MODELS: Final = re.compile(r"^[ \t]*model\s+(\w+)\s*\{(.*?)^[ \t]*\}", re.MULTILINE | re.DOTALL)


class _Options(BaseModel):
    model_config = ConfigDict(frozen=True)
    base_ref: str | None
    base_commit: str | None


def parse_model_fields(schema: str) -> Schema:
    def fields(body: str) -> Mapping[str, str]:
        parts: Final = tuple(line.split() for line in body.splitlines())
        return MappingProxyType({part[0]: " ".join(part[1:]) for part in parts if len(part) >= 2})

    return MappingProxyType({match[1]: fields(match[2]) for match in MODELS.finditer(schema)})


def _field_changes(model: str, field: str, old_type: str, new_fields: Mapping[str, str]) -> Iterator[str]:
    if field not in new_fields:
        yield f"Breaking: Field {model}.{field} was removed"
        return
    new_type: Final = new_fields[field]
    if old_type == new_type:
        return
    if "?" in old_type and "?" not in new_type:
        yield f"Breaking: Field {model}.{field} changed from optional to required"
    if not old_type.startswith(new_type.split("?")[0]):
        yield f"Breaking: Field {model}.{field} changed type from {old_type} to {new_type}"


def _model_changes(model: str, old_fields: Mapping[str, str], new_schema: Schema) -> Iterator[str]:
    if model not in new_schema:
        yield f"Breaking: Model {model} was removed"
        return
    for field, old_type in old_fields.items():
        yield from _field_changes(model, field, old_type, new_schema[model])


def check_breaking_changes(old_schema: Schema, new_schema: Schema) -> tuple[str, ...]:
    def changes() -> Iterator[str]:
        for model, fields in old_schema.items():
            yield from _model_changes(model, fields, new_schema)

    return tuple(changes())


def main() -> int:
    parser: Final = argparse.ArgumentParser()
    baseline: Final = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--base-ref")
    baseline.add_argument("--base-commit")
    options: Final = _Options.model_validate(vars(parser.parse_args()))
    base: Final = (
        options.base_commit
        if options.base_commit is not None
        else subprocess.run(
            ("git", "merge-base", "HEAD", TypeAdapter(str).validate_python(options.base_ref)),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    previous: Final = subprocess.run(
        ("git", "show", f"{base}:schema.prisma"), capture_output=True, text=True, check=True
    ).stdout
    changes: Final = check_breaking_changes(
        parse_model_fields(previous), parse_model_fields(Path("schema.prisma").read_text())
    )
    if changes:
        sys.stderr.write("\n".join(changes) + "\n")
        return 1
    sys.stdout.write(f"Schema compatibility passed against baseline {base}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
