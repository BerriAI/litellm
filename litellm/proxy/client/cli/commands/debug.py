"""`lite debug claude`: one-shot debug report for a Claude Code session routed through the proxy.

Claude Code puts its session id in `metadata.user_id`, which the proxy lifts into
`LiteLLM_SpendLogs.session_id`. This command pulls every turn of that session, plus
the request / response bodies for failures and the most recent turns, and renders a
single markdown report that can be pasted into a bug report or handed to another agent.
"""

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Final

import click
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError, field_validator

from ...http_client import HTTPClient
from ._cli_context import cli_context_values

CLAUDE_DIR: Final = Path.home() / ".claude"
REPORT_DIR: Final = Path.home() / ".litellm" / "debug"
SESSION_ID_ENV: Final = "CLAUDE_SESSION_ID"
SLASH_COMMAND_NAME: Final = "debug-lite"
SLASH_COMMAND_BODY: Final = """---
description: Pull the LiteLLM debug report (spend, request, response, error) for this Claude Code session
allowed-tools: Bash(lite debug claude:*)
---
Below is the LiteLLM debug report for this Claude Code session. Summarize the failing
request(s) in a few sentences (model, error, request id) and tell me the path the full
report was saved to so I can hand it off. If nothing failed, say so.

!`lite debug claude $ARGUMENTS`
"""


class DebugError(Exception):
    """Raised for any user-actionable failure while building the report."""


class ErrorInformation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    error_code: str | None = None
    error_class: str | None = None
    error_message: str | None = None
    llm_provider: str | None = None


class SpendLogMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    status: str | None = None
    error_information: ErrorInformation | None = None


class SpendLogRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    request_id: str
    start_time: str | None = Field(default=None, alias="startTime")
    end_time: str | None = Field(default=None, alias="endTime")
    model: str | None = None
    model_group: str | None = None
    custom_llm_provider: str | None = None
    api_base: str | None = None
    call_type: str | None = None
    status: str | None = None
    spend: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    metadata: SpendLogMetadata = SpendLogMetadata()

    @field_validator("metadata", mode="before")
    @classmethod
    def _parse_metadata(cls, value: object) -> object:
        if value is None:
            return SpendLogMetadata()
        if isinstance(value, str):
            return json.loads(value) if value else SpendLogMetadata()
        return value

    @property
    def failed(self) -> bool:
        return (self.status or self.metadata.status) == "failure"

    @property
    def error(self) -> ErrorInformation | None:
        return self.metadata.error_information


class SessionLogsPage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    data: tuple[SpendLogRow, ...]
    total: int
    total_pages: int


class RequestResponsePayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    proxy_server_request: JsonValue = None
    response: JsonValue = None
    messages: JsonValue = None


_SESSION_PAGE: Final = TypeAdapter(SessionLogsPage)
_PAYLOAD: Final[TypeAdapter[RequestResponsePayload | None]] = TypeAdapter(RequestResponsePayload | None)
_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)

_SESSION_PAGE_SIZE: Final = 100


def detect_claude_session_id(env: Mapping[str, str], claude_dir: Path) -> str | None:
    """Explicit env var first, else the transcript Claude Code touched most recently."""
    explicit: Final = env.get(SESSION_ID_ENV)
    if explicit:
        return explicit
    transcripts: Final = tuple(claude_dir.glob("projects/*/*.jsonl"))
    if not transcripts:
        return None
    newest: Final = max(transcripts, key=lambda p: p.stat().st_mtime)
    return newest.stem


class SpendLogsFetcher:
    """Thin typed wrapper over the two spend-log endpoints the report needs."""

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    def session_rows(self, session_id: str) -> tuple[SpendLogRow, ...]:
        first: Final = self._page(session_id, 1)
        rest: Final = tuple(
            row for page in range(2, first.total_pages + 1) for row in self._page(session_id, page).data
        )
        rows: Final = first.data + rest
        return tuple(sorted(rows, key=lambda r: r.start_time or ""))

    def _get(self, uri: str, params: Mapping[str, str | int] | None = None) -> JsonValue:
        return _JSON.validate_python(self._http.request("GET", uri, params=params))  # pyright: ignore[reportUnknownMemberType]  # HTTPClient.request is untyped

    def _page(self, session_id: str, page: int) -> SessionLogsPage:
        raw: Final = self._get(
            "/spend/logs/session/ui",
            MappingProxyType({"session_id": session_id, "page": page, "page_size": _SESSION_PAGE_SIZE}),
        )
        try:
            return _SESSION_PAGE.validate_python(raw)
        except ValidationError as e:
            raise DebugError(f"Unexpected /spend/logs/session/ui response: {e}") from e

    def payload(self, request_id: str) -> RequestResponsePayload | None:
        raw: Final = self._get(f"/spend/logs/ui/{request_id}")
        try:
            return _PAYLOAD.validate_python(raw)
        except ValidationError as e:
            raise DebugError(f"Unexpected /spend/logs/ui/{request_id} response: {e}") from e


def _fmt_json(value: JsonValue, max_chars: int) -> str:
    text: Final = value if isinstance(value, str) else json.dumps(value, indent=2, default=str)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n... (truncated, {len(text) - max_chars} more chars)"


def _row_section(row: SpendLogRow, index: int, payload: RequestResponsePayload | None, max_chars: int) -> str:
    err: Final = row.error
    error_lines: Final = (
        (
            f"- error: `{err.error_code or '?'}` {err.error_class or ''}".rstrip(),
            f"\n```\n{err.error_message or ''}\n```",
        )
        if err is not None and row.failed
        else ()
    )
    body_lines: Final = (
        (
            "",
            "<details><summary>request body</summary>",
            "",
            "```json",
            _fmt_json(payload.proxy_server_request, max_chars),
            "```",
            "</details>",
            "",
            "<details><summary>response</summary>",
            "",
            "```json",
            _fmt_json(payload.response, max_chars),
            "```",
            "</details>",
        )
        if payload is not None
        else ()
    )
    header: Final = f"### {index}. {'FAILED' if row.failed else 'ok'} {row.model or row.model_group or '?'}"
    facts: Final = (
        f"- request_id: `{row.request_id}`",
        f"- time: {row.start_time} -> {row.end_time}",
        f"- provider: {row.custom_llm_provider or '?'} ({row.api_base or 'n/a'}), call_type: {row.call_type or '?'}",
        f"- spend: ${row.spend:.6f}, tokens: {row.prompt_tokens} in / {row.completion_tokens} out",
    )
    return "\n".join((header, *facts, *error_lines, *body_lines))


def render_report(
    *,
    session_id: str,
    base_url: str,
    rows: Sequence[SpendLogRow],
    payloads: Mapping[str, RequestResponsePayload | None],
    max_chars: int,
) -> str:
    failures: Final = tuple(r for r in rows if r.failed)
    summary: Final = (
        f"# LiteLLM debug report: Claude Code session `{session_id}`",
        "",
        f"- proxy: {base_url}",
        f"- generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- turns: {len(rows)}, failed: {len(failures)}",
        f"- total spend: ${sum(r.spend for r in rows):.6f}",
        f"- models: {', '.join(sorted(frozenset(r.model or r.model_group or '?' for r in rows))) or 'n/a'}",
        "",
        "Bodies are included for failed turns and the most recent turns. "
        "Bodies are empty unless the proxy runs with `general_settings.store_prompts_in_spend_logs: true`.",
        "",
        "## Turns",
        "",
    )
    sections: Final = tuple(
        _row_section(row, i, payloads.get(row.request_id), max_chars) for i, row in enumerate(rows, start=1)
    )
    return "\n".join(summary) + "\n\n".join(sections) + "\n"


def build_report(
    *,
    fetcher: SpendLogsFetcher,
    session_id: str,
    base_url: str,
    recent_bodies: int,
    max_chars: int,
) -> str:
    rows: Final = fetcher.session_rows(session_id)
    if not rows:
        raise DebugError(
            f"No spend logs found for session {session_id!r} on {base_url}. "
            "Is Claude Code routed through this proxy (`lite up`), and does your key have log access?"
        )
    wanted: Final = frozenset(r.request_id for r in rows if r.failed) | frozenset(
        r.request_id for r in rows[-recent_bodies:] if recent_bodies > 0
    )
    payloads: Final = MappingProxyType({rid: fetcher.payload(rid) for rid in wanted})
    return render_report(session_id=session_id, base_url=base_url, rows=rows, payloads=payloads, max_chars=max_chars)


def write_report(report: str, session_id: str, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path: Final = report_dir / f"claude-{session_id}.md"
    path.write_text(report, encoding="utf-8")
    path.chmod(0o600)
    return path


def install_slash_command(claude_dir: Path) -> Path:
    commands_dir: Final = claude_dir / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)
    path: Final = commands_dir / f"{SLASH_COMMAND_NAME}.md"
    path.write_text(SLASH_COMMAND_BODY, encoding="utf-8")
    return path


@click.group()
def debug() -> None:
    """Pull debug reports (spend, request, response, error) for coding-agent sessions"""


@debug.command("claude")
@click.option(
    "--session-id",
    default=None,
    help=f"Claude Code session id. Defaults to ${SESSION_ID_ENV}, else the most recently used transcript in ~/.claude",
)
@click.option(
    "--recent-bodies",
    default=3,
    show_default=True,
    type=click.IntRange(min=0),
    help="Also include request/response bodies for the N most recent turns (failed turns always get bodies)",
)
@click.option(
    "--max-body-chars",
    default=20_000,
    show_default=True,
    type=click.IntRange(min=100),
    help="Truncate each request/response body to this many characters",
)
@click.option("--no-save", is_flag=True, help="Print only, do not write the report under ~/.litellm/debug")
@click.pass_context
def debug_claude(
    ctx: click.Context, session_id: str | None, recent_bodies: int, max_body_chars: int, no_save: bool
) -> None:
    """Render a markdown debug report for one Claude Code session routed through the proxy

    Examples:
        lite debug claude
        lite debug claude --session-id e96634a3-fa28-4083-b354-55542e2dca01
    """
    resolved: Final = session_id or detect_claude_session_id(os.environ, CLAUDE_DIR)
    if resolved is None:
        raise click.ClickException(f"Could not find a Claude Code session. Pass --session-id or set ${SESSION_ID_ENV}.")
    values: Final = cli_context_values(ctx)
    base_url: Final = values["base_url"]
    fetcher: Final = SpendLogsFetcher(HTTPClient(base_url, values["api_key"]))
    try:
        report: Final = build_report(
            fetcher=fetcher,
            session_id=resolved,
            base_url=base_url,
            recent_bodies=recent_bodies,
            max_chars=max_body_chars,
        )
    except DebugError as e:
        raise click.ClickException(str(e)) from e
    click.echo(report)
    if not no_save:
        path: Final = write_report(report, resolved, REPORT_DIR)
        click.echo(f"Saved to {path}", err=True)


@debug.command("install-claude-command")
def debug_install_claude_command() -> None:
    """Install the /debug-lite slash command into ~/.claude/commands so Claude Code can run `lite debug claude`"""
    path: Final = install_slash_command(CLAUDE_DIR)
    click.echo(f"Installed /{SLASH_COMMAND_NAME}: {path}")
    click.echo("Restart Claude Code (or start a new session), then type /debug-lite.")
