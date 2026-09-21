"""Quoting for command lines that cmd.exe reads before handing them to a program."""

from typing import Final

_CMD_PERCENT_GUARD: Final = "%%cd:~,%"


def _double_trailing_backslashes(segment: str) -> str:
    bare: Final = segment.rstrip("\\")
    return bare + "\\" * 2 * (len(segment) - len(bare))


def quote_for_cmd(token: str) -> str:
    """Quote one token so both parsers that read it see the original text.

    Follows the algorithm the Rust standard library settled on for batch files
    after CVE-2024-24576. Two parsers see this token: cmd.exe, which ends a
    quoted string on a lone `"` and so wants an embedded one doubled, and the
    program's own C runtime argv split, where a backslash escapes the quote that
    follows it, so every backslash run standing before a quote is doubled.
    Quoting cannot stop cmd expanding `%VAR%`, so each `%` is prefixed with
    `%%cd:~,`: the zero-length substring of the always defined `cd` expands to
    nothing and leaves no `%` pair for cmd to match.
    """
    escaped: Final = '""'.join(_double_trailing_backslashes(part) for part in token.split('"'))
    return '"' + escaped.replace("%", _CMD_PERCENT_GUARD) + '"'
