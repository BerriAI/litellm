"""Regenerate generated/values.json: what CPython produces for each value in CORPUS.

    python scripts/generate_fixtures.py > generated/values.json

Each row records `repr`, `str`, `json.dumps` (or its error), `bool`, `int`, `float`, pydantic's
lax `int`, `float`, and `bool` validation, and `pickle.dumps` at every protocol. Conversions record the result's
`repr` or the exception class raised. Run it with the repository environment, which has pydantic. `literal` says whether `ast.literal_eval(repr(value))` gives the value back,
which is how Python reads `str(dict)` text back from a cache; the Rust tests reach the other
rows only through pickle. `view` is `repr` of the value as `pickle::loads` decodes it, with
tuples, sets, and frozensets rendered as lists. `sources` records `ast.literal_eval` on raw
source texts: its result, or the exception it raises.
"""

import ast
import json
import pickle
import sys
import warnings

from pydantic import TypeAdapter, ValidationError

LAX_INT = TypeAdapter(int)
LAX_FLOAT = TypeAdapter(float)
LAX_BOOL = TypeAdapter(bool)

# Entries are source texts, or `(name, source)` when the source is too long to read in a
# test report. `name` is what the Rust `KNOWN` table keys on.
CORPUS = [
    # Scalars
    "None",
    "True",
    "False",
    "0",
    "-7",
    "2**63 - 1",
    "-(2**63)",
    "2**64",
    "-(2**70)",
    # Floats around CPython's repr thresholds
    "0.0",
    "-0.0",
    "0.2",
    "1.0",
    "-1.5",
    "0.1 + 0.2",
    "123456789.123",
    "1e15",
    "1e16",
    "1.5e16",
    "9999999999999998.0",
    "0.0001",
    "1e-05",
    "1.25e-07",
    "5e-324",
    "1.7976931348623157e308",
    "1e22",
    "float('inf')",
    "float('-inf')",
    "float('nan')",
    # Complex
    "1j",
    "-1j",
    "complex(0, -1)",
    "1+2j",
    "-1.5-0.5j",
    "complex(0.0, 1e16)",
    # Strings: quote selection, escapes, printable and non-printable non-ASCII
    "''",
    "'plain'",
    '"it\'s"',
    "'say \"hi\"'",
    "'both \\' and \"'",
    "'back\\\\slash'",
    "'\\t\\n\\r'",
    "'\\x00\\x1f\\x7f'",
    "'\\x85\\xa0\\xad'",
    "'caf\\xe9'",
    "'\\u65e5\\u672c'",
    "'\\u200b\\u2028\\u3000'",
    "'\\U0001f600'",
    "'\\U000e0001\\U0010ffff'",
    "'\\b\\f'",
    # Bytes
    "b''",
    "b'abc'",
    'b"a\'b"',
    "b'a\"b\\'c'",
    "b'\\x00\\t\\n\\r\\x7f\\x80\\xff'",
    # Numeric text, as providers and config files send it
    "' 12 '",
    "'+5'",
    "'-4'",
    "'007'",
    "'1_000'",
    "'1__0'",
    "'_1'",
    "'1_'",
    "'3.0'",
    "'3.00'",
    "'3.5'",
    "' 0.5 '",
    "'true'",
    "' Yes '",
    "'off'",
    "'2'",
    "1",
    "'.5'",
    "'5.'",
    "'1e3'",
    "'1_0.5'",
    "'1e1_0'",
    "'inf'",
    "'-Infinity'",
    "'nan'",
    "'NaN'",
    "'0x10'",
    "'abc'",
    "b'12'",
    "b' 1.5 '",
    "3.9",
    "-3.9",
    "2.5",
    # Containers
    "[]",
    "[1, 'a', None, True]",
    "()",
    "(1,)",
    "(1, (2, 3))",
    "{}",
    "{'a': 1, 'b': [1.0, 2.5]}",
    "{'z': 1, 'a': 2, 'm': 3}",
    "{1: 'int', 2.5: 'float', True: 'bool', None: 'none'}",
    "{(1, 2): 'tuple key'}",
    "{'nested': {'deeper': {'deepest': [{}]}}}",
    "{1}",
    "set()",
    "frozenset({1})",
    "[[[[[[[[[[1]]]]]]]]]]",
    # Deeper than the Rust decoders allow: CPython's parser accepts ~200 nested brackets
    # and its unpickler has no limit, so this row records a deliberate divergence.
    ("nested_150", "[" * 150 + "1" + "]" * 150),
    # The shape LiteLLM caches
    '{\'timestamp\': 1726000000.123, \'response\': \'{"id": "chatcmpl-1", "object": "chat.completion"}\'}',
    "{'model': 'gpt-4o', 'messages': [{'role': 'user', 'content': 'hi'}], 'temperature': 0.2, 'stream': False}",
]

# Source texts for `literal_eval` itself: tokenizer and evaluator edge cases, recorded with
# CPython's result or the exception it raises. Raw strings keep backslashes literal.
SOURCES = [
    # Layout: leading/trailing whitespace, comments, newlines, continuations
    "1",
    "  1",
    "\t1",
    "\n1",
    " \n 1",
    "1\n",
    "1 # comment",
    "# c\n1",
    "1 \\\n",
    "1,",
    "1, 2",
    "1,\n2",
    "(1,\n2)",
    "[1,\n 2,\n]",
    # Containers and grouping
    "()",
    "(1)",
    "((1,))",
    "(,)",
    "[,]",
    "{,}",
    "[1,]",
    "{'a': 1,}",
    "{1,}",
    "{'a': 1 'b': 2}",
    "{1: 'a', True: 'b'}",
    "{1, True, 1.0}",
    "{(1, 2): 'x', (1.0, 2): 'y'}",
    "{[1]: 2}",
    "{{1}}",
    "{(1, [2])}",
    "set()",
    "set( )",
    "set([1])",
    "frozenset()",
    # Names
    "True",
    "False",
    "None",
    "Truex",
    "true",
    "...",
    # Integers and floats
    "0",
    "00",
    "0_0",
    "01",
    "007",
    "1_000",
    "1_",
    "1__0",
    "_1",
    "0x1F",
    "0X_1f",
    "0o17",
    "0b101",
    "0b102",
    "0x",
    "1e3",
    "1E-3",
    "1e",
    "1.e5",
    ".5",
    "5.",
    "1..",
    "1.5.2",
    "1_0.0_1e1_0",
    "1e999",
    "-1e999",
    "1j",
    "1.5J",
    "010j",
    "010.5",
    "1a",
    "0x1g",
    # Signs and complex sums
    "-1",
    "+1",
    "- 1",
    "--1",
    "-+1",
    "-(1)",
    "-(-1)",
    "-(1+2j)",
    "-True",
    "-'a'",
    "-[1]",
    "1+2j",
    "1-2j",
    "1 + 2j",
    "(1)+(2j)",
    "1+2",
    "1+2+3j",
    "1+-2j",
    "2j+1",
    "True+1j",
    "1-0j",
    "0.0-0j",
    "-0.0+1j",
    "-0.0",
    "-0",
    "(-0.0)",
    "[-0.0, (-0.0)]",
    "2**3",
    "1*2",
    "1 if 1 else 2",
    "(1,)(2)",
    # String prefixes, quoting, and concatenation
    "''",
    '""',
    "'a' 'b'",
    "'a' \"b\" '''c'''",
    "'a' b'b'",
    "b'a' b'b'",
    "u'x'",
    "U'x'",
    "r'x'",
    "R'x'",
    "b'x'",
    "B'x'",
    "br'x'",
    "Rb'x'",
    "rB'x'",
    "ur'x'",
    "bu'x'",
    "f'x'",
    "rf'x'",
    "'''a\nb'''",
    '"""a\\""""',
    "'a\nb'",
    "'a\\\nb'",
    "r'a\\\nb'",
    "'unterminated",
    # Escapes
    r"'\a\b\f\n\r\t\v'",
    r"'\0\12\101\1011'",
    r"'\777'",
    r"b'\777'",
    r"b'\400'",
    r"'\x41'",
    r"'\x4'",
    r"'\u00e9'",
    r"'\u00e'",
    r"'\U0001F600'",
    r"'\U00110000'",
    r"'\ud800'",
    r"'\N{BULLET}'",
    r"'\q'",
    r"'\\'",
    r"'\''",
    r'"\""',
    r"b'\u0041'",
    r"b'\x41\xff'",
    "b'café'",
    "'café'",
    r"r'\d'",
    r"r'\''",
    r"rb'\d'",
    r"r'\'",
    "'日\U0001f600'",
]


def named(entry):
    """Split a corpus entry into its report name and its source text."""
    if isinstance(entry, tuple):
        return entry
    return entry, entry


def evaluate(entry):
    name, source = named(entry)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return {"name": name, "source": source, "repr": repr(ast.literal_eval(source))}
        except Exception as error:  # noqa: BLE001 - recorded, not raised
            return {"name": name, "source": source, "error": type(error).__name__}


def view(value):
    if isinstance(value, (list, tuple, set, frozenset)):
        return "[" + ", ".join(view(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{view(key)}: {view(item)}" for key, item in value.items()) + "}"
    return repr(value)


def plain(value):
    """Whether pickle can encode the value without a class reference such as `complex`."""
    if isinstance(value, complex):
        return False
    if isinstance(value, (list, tuple, set, frozenset)):
        return all(plain(item) for item in value)
    if isinstance(value, dict):
        return all(plain(key) and plain(item) for key, item in value.items())
    return True


def is_literal(value):
    try:
        parsed = ast.literal_eval(repr(value))
    except (ValueError, SyntaxError):
        return False
    return repr(parsed) == repr(value)


def converted(convert, value):
    try:
        return {"value": repr(convert(value))}
    except (TypeError, ValueError, OverflowError, ValidationError) as error:
        return {"error": "ValueError" if isinstance(error, ValidationError) else type(error).__name__}


def row(entry):
    name, source = named(entry)
    value = eval(source)
    entry = {
        "name": name,
        "source": source,
        "literal": is_literal(value),
        "plain": plain(value),
        "repr": repr(value),
        "str": str(value),
        "truthy": bool(value),
        "int": converted(int, value),
        "float": converted(float, value),
        "lax_int": converted(LAX_INT.validate_python, value),
        "lax_float": converted(LAX_FLOAT.validate_python, value),
        "lax_bool": converted(LAX_BOOL.validate_python, value),
    }
    try:
        entry["json"] = json.dumps(value)
    except (TypeError, ValueError) as error:
        entry["json_error"] = str(error)
    try:
        entry["pickle"] = {str(protocol): pickle.dumps(value, protocol=protocol).hex() for protocol in range(6)}
        entry["view"] = view(value)
    except Exception as error:  # noqa: BLE001 - recorded, not raised
        entry["pickle_error"] = f"{type(error).__name__}: {error}"
    return entry


if __name__ == "__main__":
    json.dump(
        {
            "python": sys.version.split()[0],
            "rows": [row(entry) for entry in CORPUS],
            "sources": [evaluate(entry) for entry in SOURCES],
        },
        sys.stdout,
        indent=2,
        ensure_ascii=True,
    )
    sys.stdout.write("\n")
