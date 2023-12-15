from __future__ import annotations

from collections.abc import Generator, Mapping
from datetime import date, datetime, time
from decimal import Decimal
import string
from types import MappingProxyType
from typing import Any, BinaryIO, NamedTuple

ASCII_CTRL = frozenset(chr(i) for i in range(32)) | frozenset(chr(127))
ILLEGAL_BASIC_STR_CHARS = frozenset('"\\') | ASCII_CTRL - frozenset("\t")
BARE_KEY_CHARS = frozenset(string.ascii_letters + string.digits + "-_")
ARRAY_TYPES = (list, tuple)
ARRAY_INDENT = " " * 4
MAX_LINE_LENGTH = 100

COMPACT_ESCAPES = MappingProxyType(
    {
        "\u0008": "\\b",  # backspace
        "\u000A": "\\n",  # linefeed
        "\u000C": "\\f",  # form feed
        "\u000D": "\\r",  # carriage return
        "\u0022": '\\"',  # quote
        "\u005C": "\\\\",  # backslash
    }
)


def dump(
    __obj: dict[str, Any], __fp: BinaryIO, *, multiline_strings: bool = False
) -> None:
    ctx = Context(multiline_strings, {})
    for chunk in gen_table_chunks(__obj, ctx, name=""):
        __fp.write(chunk.encode())


def dumps(__obj: dict[str, Any], *, multiline_strings: bool = False) -> str:
    ctx = Context(multiline_strings, {})
    return "".join(gen_table_chunks(__obj, ctx, name=""))


class Context(NamedTuple):
    allow_multiline: bool
    # cache rendered inline tables (mapping from object id to rendered inline table)
    inline_table_cache: dict[int, str]


def gen_table_chunks(
    table: Mapping[str, Any],
    ctx: Context,
    *,
    name: str,
    inside_aot: bool = False,
) -> Generator[str, None, None]:
    yielded = False
    literals = []
    tables: list[tuple[str, Any, bool]] = []  # => [(key, value, inside_aot)]
    for k, v in table.items():
        if isinstance(v, dict):
            tables.append((k, v, False))
        elif is_aot(v) and not all(is_suitable_inline_table(t, ctx) for t in v):
            tables.extend((k, t, True) for t in v)
        else:
            literals.append((k, v))

    if inside_aot or name and (literals or not tables):
        yielded = True
        yield f"[[{name}]]\n" if inside_aot else f"[{name}]\n"

    if literals:
        yielded = True
        for k, v in literals:
            yield f"{format_key_part(k)} = {format_literal(v, ctx)}\n"

    for k, v, in_aot in tables:
        if yielded:
            yield "\n"
        else:
            yielded = True
        key_part = format_key_part(k)
        display_name = f"{name}.{key_part}" if name else key_part
        yield from gen_table_chunks(v, ctx, name=display_name, inside_aot=in_aot)


def format_literal(obj: object, ctx: Context, *, nest_level: int = 0) -> str:
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, (int, float, date, datetime)):
        return str(obj)
    if isinstance(obj, Decimal):
        return format_decimal(obj)
    if isinstance(obj, time):
        if obj.tzinfo:
            raise ValueError("TOML does not support offset times")
        return str(obj)
    if isinstance(obj, str):
        return format_string(obj, allow_multiline=ctx.allow_multiline)
    if isinstance(obj, ARRAY_TYPES):
        return format_inline_array(obj, ctx, nest_level)
    if isinstance(obj, dict):
        return format_inline_table(obj, ctx)
    raise TypeError(f"Object of type {type(obj)} is not TOML serializable")


def format_decimal(obj: Decimal) -> str:
    if obj.is_nan():
        return "nan"
    if obj == Decimal("inf"):
        return "inf"
    if obj == Decimal("-inf"):
        return "-inf"
    return str(obj)


def format_inline_table(obj: dict, ctx: Context) -> str:
    # check cache first
    obj_id = id(obj)
    if obj_id in ctx.inline_table_cache:
        return ctx.inline_table_cache[obj_id]

    if not obj:
        rendered = "{}"
    else:
        rendered = (
            "{ "
            + ", ".join(
                f"{format_key_part(k)} = {format_literal(v, ctx)}"
                for k, v in obj.items()
            )
            + " }"
        )
    ctx.inline_table_cache[obj_id] = rendered
    return rendered


def format_inline_array(obj: tuple | list, ctx: Context, nest_level: int) -> str:
    if not obj:
        return "[]"
    item_indent = ARRAY_INDENT * (1 + nest_level)
    closing_bracket_indent = ARRAY_INDENT * nest_level
    return (
        "[\n"
        + ",\n".join(
            item_indent + format_literal(item, ctx, nest_level=nest_level + 1)
            for item in obj
        )
        + f",\n{closing_bracket_indent}]"
    )


def format_key_part(part: str) -> str:
    if part and BARE_KEY_CHARS.issuperset(part):
        return part
    return format_string(part, allow_multiline=False)


def format_string(s: str, *, allow_multiline: bool) -> str:
    do_multiline = allow_multiline and "\n" in s
    if do_multiline:
        result = '"""\n'
        s = s.replace("\r\n", "\n")
    else:
        result = '"'

    pos = seq_start = 0
    while True:
        try:
            char = s[pos]
        except IndexError:
            result += s[seq_start:pos]
            if do_multiline:
                return result + '"""'
            return result + '"'
        if char in ILLEGAL_BASIC_STR_CHARS:
            result += s[seq_start:pos]
            if char in COMPACT_ESCAPES:
                if do_multiline and char == "\n":
                    result += "\n"
                else:
                    result += COMPACT_ESCAPES[char]
            else:
                result += "\\u" + hex(ord(char))[2:].rjust(4, "0")
            seq_start = pos + 1
        pos += 1


def is_aot(obj: Any) -> bool:
    """Decides if an object behaves as an array of tables (i.e. a nonempty list
    of dicts)."""
    return bool(
        isinstance(obj, ARRAY_TYPES) and obj and all(isinstance(v, dict) for v in obj)
    )


def is_suitable_inline_table(obj: dict, ctx: Context) -> bool:
    """Use heuristics to decide if the inline-style representation is a good
    choice for a given table."""
    rendered_inline = f"{ARRAY_INDENT}{format_inline_table(obj, ctx)},"
    return len(rendered_inline) <= MAX_LINE_LENGTH and "\n" not in rendered_inline
