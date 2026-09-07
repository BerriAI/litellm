"""
Decides which tool_use blocks shunt rewrites, and builds the Bash command that replaces them.

Ports two decisions from Spotify's shunt plugin (see auto_router_shunt.py's module docstring):
`check-file-size`'s "is this an untargeted Read on a large file" gate, and
`check-bash-read`'s "is this cat/head/tail/less/more on a bare file path" parser, including its
documented parser bug (an option-value token like the `5` in `head -n 5 file` is mistaken for
the path). shunt's version tolerates that bug because its hook runs on the file's own machine
and checks the guessed path exists before blocking; a misparse then falls through to `allow`,
leaving the original command untouched. This module has no filesystem access to make that same
check, so it must never rewrite a command it isn't sure it parsed correctly — see
`extract_bare_read_path`'s docstring for how the port preserves shunt's fail-safe direction.
"""

import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

_READ_COMMANDS: Final = ("cat", "head", "tail", "less", "more")
_READ_COMMAND_PATTERN: Final = re.compile(r"^(?:" + "|".join(_READ_COMMANDS) + r")\s")
_PIPE_OR_REDIRECT_PATTERN: Final = re.compile(r"[|>]")


def is_targeted_read(offset: object, limit: object) -> bool:
    """Whether a `Read`-shaped tool call already names a bounded section.

    Mirrors `check-file-size`'s "offset or limit set" gate, including its documented bypass:
    `offset: 0` or `limit: 0` count as targeted, matching shunt's own behavior exactly.
    """
    return offset is not None or limit is not None


def extract_bare_read_path(command: str) -> str | None:
    """The file path a bare `cat`/`head`/`tail`/`less`/`more` command reads, or None.

    None means "do not rewrite this command": either it isn't a bulk read (piped, redirected,
    not one of the five commands), or no non-flag argument was found.

    shunt's own parser walks flags as if none of them take a value, which misreads a flag's
    value token as the path — e.g. the `5` in `head -n 5 file`. shunt tolerates this because its
    hook runs on the file's own machine and checks the guessed path actually exists before
    blocking; a nonexistent guess falls through to `allow`, leaving the original command
    untouched. This port has no filesystem to make that same check, so it cannot rely on a
    downstream existence check to catch a bad guess — treating a wrong guess as the real path
    would rewrite the command against a file the model never named. A bare numeric token (as
    every flag value in the five commands' own option sets happens to be — `-n`, `head -c`,
    `tail -n +N`) is skipped as a likely flag value rather than returned, which fixes shunt's
    documented bug for exactly the case it names rather than reproducing it. A path made only
    of digits (or `tail`'s `+N` follow-from-line syntax) is stripped either way, matching
    shunt's own scope: neither version claims to handle a bare-numeric filename correctly.

    Like shunt's own bash word-splitting, this does not handle a quoted path containing spaces
    (`cat "my file.txt"` returns `"my`) — matching parity, not a regression.
    """
    if _PIPE_OR_REDIRECT_PATTERN.search(command):
        return None
    match: Final = _READ_COMMAND_PATTERN.match(command)
    if match is None:
        return None
    args: Final = command[match.end() :].split()
    for arg in args:
        if arg.startswith("-") or arg.lstrip("+-").isdigit():
            continue
        return arg.strip("\"'")
    return None


@dataclass(frozen=True, slots=True)
class ShuntBashRewrite:
    """A Bash command that replaces an intercepted tool_use, and why."""

    command: str
    note: str


# The generated command reads the client's own credential out of its environment at run time
# instead of carrying it. Embedding the value would copy the caller's key into the model's
# response, the conversation history, and the next upstream turn, which is exactly what keeping
# it in `secret_fields` is meant to prevent. `${VAR:-$OTHER}` also covers both header styles:
# Claude Code sets ANTHROPIC_AUTH_TOKEN, while an x-api-key client sets ANTHROPIC_API_KEY.
_AUTH_ENV_EXPR: Final = "${ANTHROPIC_AUTH_TOKEN:-$ANTHROPIC_API_KEY}"
# Deliberately double-quoted, not shlex.quote'd: this is shell syntax to evaluate, not data.
_AUTH_FLAG: Final = f'-H "Authorization: Bearer {_AUTH_ENV_EXPR}"'


def build_bounded_read_command(
    *, path: str, question: str, min_lines: int, bulk_read_endpoint: str
) -> ShuntBashRewrite:
    """The shunt conditional: read small files directly, delegate large ones.

    One template for both rewrite sources (an untargeted `Read` tool call, and a bare
    `cat`/`head`/`tail`/`less`/`more` command) since both made the identical decision in shunt.
    `-sS` on curl mirrors shunt's own `-s` plus fails loudly on a server error rather than
    silently returning an HTML error page as if it were the answer.

    Every interpolated value goes through `shlex.quote`, and the path is reported with `printf`
    rather than inside a double-quoted `echo`, because these values come from the model and the
    command runs a shell on the developer's own machine: inside double quotes a `$(...)` in a
    path would still be command-substituted. `min_lines` is an int, so it needs no quoting.
    """
    quoted_path: Final = shlex.quote(path)
    command: Final = (
        f"L=$(wc -l < {quoted_path} 2>/dev/null || echo 0); "
        f'if [ "$L" -gt {min_lines} ]; then '
        f"printf '[shunt] %s: %s lines, bounded read delegated\\n' {quoted_path} \"$L\" >&2; "
        f"curl -sS -F {shlex.quote(f'question={question}')} "
        f"-F {shlex.quote(f'paths=@{path}')} "
        f"{_AUTH_FLAG} {shlex.quote(bulk_read_endpoint)}; "
        f"else cat {quoted_path}; fi"
    )
    return ShuntBashRewrite(
        command=command,
        note=f"Bounded read: files over {min_lines} lines are delegated to a cheaper model.",
    )


def build_bulk_read_command(*, question: str, paths: Sequence[str], bulk_read_endpoint: str) -> ShuntBashRewrite:
    """The curl a model's own explicit `bulk_read(question, paths)` tool call becomes.

    Unconditional (no size check): the model chose to delegate, unlike the automatic bounding
    `build_bounded_read_command` applies to a plain `Read`/`cat`/`head`/`tail` call.
    """
    path_flags: Final = " ".join(f"-F {shlex.quote(f'paths=@{path}')}" for path in paths)
    command: Final = (
        f"curl -sS -F {shlex.quote(f'question={question}')} {path_flags} {_AUTH_FLAG} {shlex.quote(bulk_read_endpoint)}"
    )
    return ShuntBashRewrite(command=command, note="Delegated to a cheaper model via bulk_read.")


def build_code_write_command(
    *, spec: str, reference: str, target: str | None, code_write_endpoint: str
) -> ShuntBashRewrite:
    """The curl a model's own explicit `code_write(spec, reference, target)` tool call becomes.

    shunt's own script writes straight to disk since it runs on the file's own machine; this
    has no local filesystem, so the generated code returns to the client, which writes it
    itself when `target` is given. Either way the generated code enters the client's context
    only as a file write, never as text the routed model has to hold or repeat.
    """
    request: Final = (
        f"curl -sS -F {shlex.quote(f'spec={spec}')} "
        f"-F {shlex.quote(f'reference=@{reference}')} "
        f"{_AUTH_FLAG} {shlex.quote(code_write_endpoint)}"
    )
    if target is None:
        return ShuntBashRewrite(command=request, note="Delegated to a cheaper model via code_write.")
    command: Final = f"{request} > {shlex.quote(target)}"
    return ShuntBashRewrite(command=command, note=f"Delegated to a cheaper model via code_write, written to {target}.")
