import argparse
import os
import sys
from functools import reduce
from pathlib import Path
from typing import Final
from xml.sax.saxutils import escape

from pydantic import JsonValue, TypeAdapter, ValidationError
from secrets_to_env import MIN_MASKED_LENGTH

REDACTED: Final = "***"
json_adapter: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


def string_leaves(node: JsonValue) -> tuple[str, ...]:
    match node:
        case str():
            return (node,)
        case list():
            return tuple(leaf for child in node for leaf in string_leaves(child))
        case dict():
            return tuple(leaf for child in node.values() for leaf in string_leaves(child))
    return ()


def field_lines(value: str) -> tuple[str, ...]:
    try:
        return tuple(line for leaf in string_leaves(json_adapter.validate_json(value)) for line in leaf.splitlines())
    except ValidationError:
        return ()


def masked_values(values_files: tuple[Path, ...]) -> tuple[str, ...]:
    values: Final = frozenset(
        line.split("=", 1)[1].strip().strip("'")
        for path in values_files
        for line in path.read_text().splitlines()
        if "=" in line
    )
    texts: Final = frozenset(text for value in values for text in (value, *field_lines(value)))
    renderings: Final = frozenset(
        rendering
        for text in texts
        if len(text) >= MIN_MASKED_LENGTH
        for rendering in (text, escape(text), escape(text, {'"': "&quot;"}))
    )
    return tuple(sorted(renderings, key=lambda rendering: (-len(rendering), rendering)))


def redact(text: str, values: tuple[str, ...]) -> str:
    return reduce(lambda redacted, value: redacted.replace(value, REDACTED), values, text)


def write_redacted(source: Path, out_dir: Path, values: tuple[str, ...]) -> None:
    target: Final = out_dir / source.name
    with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "w") as handle:
        _ = handle.write(redact(source.read_text(errors="replace"), values))


def main() -> int:
    parser: Final = argparse.ArgumentParser()
    _ = parser.add_argument("--values", action="append", type=Path, required=True)
    _ = parser.add_argument("--out", type=Path, required=True)
    _ = parser.add_argument("files", nargs="*", type=Path)
    args: Final = parser.parse_args()
    values_files: Final = tuple(args.values)
    out_dir: Final[Path] = args.out
    sources: Final = tuple(args.files)
    try:
        values: Final = masked_values(values_files)
        out_dir.mkdir(mode=0o700, exist_ok=True)
        for source in sources:
            write_redacted(source, out_dir, values)
    except OSError as error:
        _ = sys.stderr.write(f"could not redact {error.filename}\n")
        return 1
    _ = sys.stdout.write(f"redacted {len(sources)} file(s) into {out_dir}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
