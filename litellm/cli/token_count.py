"""``litellm token-count`` — offline token counting for a single prompt.

Counts the tokens a model would see for a given text, file, or message
list, using the public ``litellm.token_counter`` helper. Reuses the
same tokenizer selection that ``litellm.completion`` uses, so the
count matches what a real call would produce. No network calls are
made and no provider credentials are required.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from typing import Any, Literal

import click

Mode = Literal["text", "file", "messages"]


@dataclass(frozen=True)
class TokenCount:
    """Result of a token-count run, suitable for both human and JSON output."""

    model: str
    mode: Mode
    count: int

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def _read_stdin() -> str:
    return sys.stdin.read()


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _resolve_messages_payload(value: str) -> list[Any]:
    """Parse the --messages argument (raw JSON string or '-' for stdin) into a list.

    The downstream ``token_counter(messages=...)`` call accepts a
    list of OpenAI-style message dicts; the JSON-decoded list is
    structurally compatible without further validation.
    """
    raw = _read_stdin() if value == "-" else value
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise click.UsageError(f"--messages must be valid JSON: {exc}") from exc
    if not isinstance(parsed, list):
        raise click.UsageError("--messages must be a JSON list of message objects.")
    return parsed


def _resolve_input(
    *,
    text: str | None,
    file: str | None,
    messages: str | None,
) -> tuple[Mode, Any]:
    """Pick the right input-source option and return (mode, payload).

    Exactly one of ``text``, ``file``, or ``messages`` must be provided.
    The payload is whatever the chosen mode needs:

    - ``text``     -> the raw string
    - ``file``     -> the file contents (already read)
    - ``messages`` -> the parsed JSON list

    Raises ``click.UsageError`` on invalid combinations so the CLI
    surfaces exit 2 instead of a Python traceback.
    """
    provided = [name for name, val in (("text", text), ("file", file), ("messages", messages)) if val is not None]
    if len(provided) == 0:
        raise click.UsageError(
            "Provide one of --text, --file, or --messages (as a JSON list, or '-' to read from stdin)."
        )
    if len(provided) > 1:
        raise click.UsageError(f"--{provided[0]} and --{provided[1]} are mutually exclusive; pass only one.")
    if text is not None:
        return "text", text
    if file is not None:
        if file == "-":
            return "file", _read_stdin()
        try:
            return "file", _read_text_file(file)
        except FileNotFoundError as exc:
            raise click.UsageError(f"--file {file!r} does not exist.") from exc
        except OSError as exc:
            raise click.UsageError(f"--file {file!r} could not be read: {exc}") from exc
    return "messages", _resolve_messages_payload(messages)  # type: ignore[arg-type]


def count_tokens(
    model: str,
    *,
    text: str | None = None,
    file: str | None = None,
    messages: str | None = None,
) -> TokenCount:
    """Public helper for programmatic use. Returns a ``TokenCount``.

    Raises :class:`ValueError` for invalid input (no input source, two
    input sources, malformed messages JSON) so callers can catch input
    validation separately from a missing model. The model is required;
    ``litellm.token_counter`` falls back to a generic encoder for
    unregistered model names, so unknown models do not raise here.
    """
    if not model:
        raise ValueError("model is required")
    try:
        mode, payload = _resolve_input(text=text, file=file, messages=messages)
    except click.UsageError as exc:
        raise ValueError(str(exc)) from exc

    from litellm import token_counter

    try:
        if mode == "text":
            count = int(token_counter(model=model, text=payload))
        elif mode == "file":
            count = int(token_counter(model=model, text=payload))
        else:
            count = int(token_counter(model=model, messages=payload))
    except Exception as exc:
        raise RuntimeError(f"token_counter raised {type(exc).__name__}: {exc}") from exc

    return TokenCount(model=model, mode=mode, count=count)


def _render_line(result: TokenCount) -> str:
    return f"model={result.model} mode={result.mode} count={result.count}"


@click.command()
@click.option(
    "--model",
    "model",
    required=True,
    help="Model name to use for tokenizer selection (e.g. gpt-4o, claude-sonnet-4-6, ollama/llama3).",
)
@click.option(
    "--text",
    "text",
    default=None,
    help="Raw text string to count tokens for. Mutually exclusive with --file and --messages.",
)
@click.option(
    "--file",
    "file_path",
    default=None,
    help="Path to a UTF-8 text file to read and count tokens for. Pass '-' to read from stdin. Mutually exclusive with --text and --messages.",
)
@click.option(
    "--messages",
    "messages_json",
    default=None,
    help="JSON list of OpenAI-style message objects to count tokens for. Pass '-' to read from stdin. Mutually exclusive with --text and --file.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Emit a JSON object instead of a single-line result.",
)
def cli(
    model: str,
    text: str | None,
    file_path: str | None,
    messages_json: str | None,
    output_json: bool,
) -> None:
    """Count tokens for a single prompt against a model tokenizer."""
    try:
        result = count_tokens(
            model,
            text=text,
            file=file_path,
            messages=messages_json,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    except click.UsageError:
        raise
    except Exception as exc:
        click.echo(f"token-count failed: {type(exc).__name__}: {exc}", err=True)
        sys.exit(1)

    if output_json:
        click.echo(json.dumps(result.to_jsonable()))
    else:
        click.echo(_render_line(result))
