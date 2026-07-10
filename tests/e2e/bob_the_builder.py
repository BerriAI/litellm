"""Bob the builder: on a red e2e run, ask Devin to fix the failing tests.

Wired as a ``pytest_sessionfinish`` step (see ``conftest.py``). When the run went
red and remediation is enabled, it hands the failing tests plus their captured
tracebacks to Devin *through the LiteLLM proxy's own MCP gateway* -- the same
gateway + master key the suite already uses -- so Devin files a Linear ticket per
failure and opens fix PRs. Nothing new ships in the runner pod: the proxy already
registers the ``devin`` MCP server and holds ``DEVIN_API_KEY``, injecting it
upstream, so this process only needs the proxy key it always has.

Opt-in via ``E2E_DEVIN_REMEDIATION=1`` so a normal local ``pytest tests/e2e`` run
never spawns a Devin session. ``DEVIN_DRY_RUN=1`` prints the prompt it would send
and makes no call. Everything is best-effort: any error here is logged and
swallowed so the run's exit status still reflects the tests, not remediation.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import pytest
from pydantic import BaseModel, ConfigDict

from e2e_config import MASTER_KEY, PROXY_BASE_URL
from e2e_http import Success
from transport import HttpTransport

REMEDIATION_ENV = "E2E_DEVIN_REMEDIATION"
_LIST_PATH = "/mcp-rest/tools/list"
_CALL_PATH = "/mcp-rest/tools/call"


@dataclass(frozen=True, slots=True)
class Failure:
    """One failed test: its pytest node id and the captured failure text."""

    nodeid: str
    detail: str


@dataclass(frozen=True, slots=True)
class Config:
    server: str
    create_tool: str
    linear_team: str
    target_repo: str
    target_ref: str
    max_failures: int
    max_detail_chars: int
    tags: tuple[str, ...]
    dry_run: bool


class _NoParams(BaseModel):
    pass


class _McpToolInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    server_name: str | None = None
    alias: str | None = None


class _McpTool(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    mcp_info: _McpToolInfo | None = None


class _McpToolsList(BaseModel):
    model_config = ConfigDict(extra="allow")
    tools: tuple[_McpTool, ...] = ()


class _DevinSessionArgs(BaseModel):
    prompt: str
    title: str
    tags: list[str]


class _ToolCallBody(BaseModel):
    name: str
    arguments: _DevinSessionArgs


class _ToolCallResult(BaseModel):
    model_config = ConfigDict(extra="allow")


class _Report(Protocol):
    @property
    def nodeid(self) -> str: ...

    @property
    def longreprtext(self) -> str: ...


class _TerminalReporter(Protocol):
    stats: Mapping[str, Sequence[_Report]]


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def load_config() -> Config:
    raw_tags = _env("DEVIN_TAGS", "e2e,stage")
    return Config(
        server=_env("DEVIN_MCP_SERVER", "devin"),
        create_tool=_env("DEVIN_SESSION_TOOL", "devin_session_create"),
        linear_team=_env("DEVIN_LINEAR_TEAM", "LIT"),
        target_repo=_env("DEVIN_TARGET_REPO", "BerriAI/litellm"),
        target_ref=_env("DEVIN_TARGET_REF", "litellm_internal_staging"),
        max_failures=int(_env("DEVIN_MAX_FAILURES", "50")),
        max_detail_chars=int(_env("DEVIN_MAX_DETAIL_CHARS", "3000")),
        tags=tuple(t.strip() for t in raw_tags.split(",") if t.strip()),
        dry_run=_env("DEVIN_DRY_RUN", "0") == "1",
    )


def collect_failures(session: pytest.Session, max_detail_chars: int) -> tuple[Failure, ...]:
    """Pull the failed and errored tests (with their tracebacks) off the run's
    terminal reporter. Returns empty when nothing failed or the reporter is
    absent (e.g. a skipped, proxy-less session)."""
    plugin: object = session.config.pluginmanager.getplugin("terminalreporter")
    if plugin is None:
        return ()
    reporter = cast(_TerminalReporter, plugin)
    reports = (*reporter.stats.get("failed", ()), *reporter.stats.get("error", ()))
    return tuple(
        Failure(nodeid=r.nodeid, detail=r.longreprtext.strip()[-max_detail_chars:]) for r in reports
    )


def dedup_tag(failures: tuple[Failure, ...]) -> str:
    """Stable short tag identifying this exact set of failing tests, so repeated
    nightly runs on the same failures reference one body of work."""
    joined = "\n".join(sorted(f.nodeid for f in failures))
    return "e2e-fail-" + hashlib.sha256(joined.encode()).hexdigest()[:12]


def _revision() -> str:
    for candidate in (Path(__file__).parent / ".litellm-revision", Path("/app/e2e/.litellm-revision")):
        try:
            return candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return _env("E2E_REVISION", "unknown")


def build_prompt(cfg: Config, failures: tuple[Failure, ...], tag: str) -> str:
    shown = failures[: cfg.max_failures]
    header = (
        f"The LiteLLM end-to-end suite failed on the "
        f"{_env('E2E_ENVIRONMENT', 'stage')} proxy. Source repo {cfg.target_repo} "
        f"at revision {_revision()} (branch {cfg.target_ref}). {len(failures)} "
        f"test(s) failed"
        + (f"; the first {len(shown)} are shown" if len(shown) < len(failures) else "")
        + ".\n\n"
    )
    task = (
        "For each failing test below:\n"
        f"1. Open a Linear ticket under the {cfg.linear_team} team describing the "
        "failure (test id, the assertion/error, likely cause), unless an open "
        "ticket for that same test already exists -- do not create duplicates.\n"
        f"2. Fix it in {cfg.target_repo}, branching off {cfg.target_ref} and "
        "following the repo's CONTRIBUTING and CLAUDE.md conventions (meaningful "
        "regression coverage, conventional commits, run the suite locally), then "
        "open a PR that references the Linear ticket.\n"
        "3. Prefer one focused PR per failing test; if several share a root cause, "
        "group them and say so.\n"
        f"Before starting, search existing sessions/PRs tagged '{tag}' or "
        "referencing these test ids and continue that work instead of restarting.\n\n"
        "Failing tests and their captured output:\n"
    )
    blocks = [f"### {i}. {f.nodeid}\n```\n{f.detail}\n```\n" for i, f in enumerate(shown, start=1)]
    return header + task + "\n".join(blocks)


def _resolve_tool_name(transport: HttpTransport, cfg: Config) -> str | None:
    """Find Devin's create-session tool on the gateway. The proxy prefixes tools
    with the server alias, so match by suffix and (when present) the owning
    server."""
    result = transport.get(
        _LIST_PATH, headers=transport.master, params=_NoParams(), response_type=_McpToolsList
    )
    if not isinstance(result, Success):
        print(f"bob_the_builder: could not list gateway MCP tools: {result}")
        return None
    for tool in result.data.tools:
        owner = tool.mcp_info.server_name or tool.mcp_info.alias if tool.mcp_info else None
        if (owner is None or owner == cfg.server) and (
            tool.name == cfg.create_tool or tool.name.endswith(cfg.create_tool)
        ):
            return tool.name
    print(
        f"bob_the_builder: no '{cfg.create_tool}' tool for server '{cfg.server}' on the gateway; "
        f"saw {[t.name for t in result.data.tools]}"
    )
    return None


def remediate(session: pytest.Session) -> None:
    """Entry point called from ``pytest_sessionfinish``. No-op unless remediation
    is enabled and the run actually had failures."""
    if os.environ.get(REMEDIATION_ENV) != "1":
        return
    cfg = load_config()
    failures = collect_failures(session, cfg.max_detail_chars)
    if not failures:
        return

    tag = dedup_tag(failures)
    title = f"Fix {len(failures)} failing LiteLLM e2e test(s) [{tag}]"
    prompt = build_prompt(cfg, failures, tag)
    args = _DevinSessionArgs(prompt=prompt, title=title, tags=[*cfg.tags, tag])

    if cfg.dry_run:
        print("bob_the_builder: DRY RUN -- would create a Devin session:")
        print(f"  server : {cfg.server}\n  tool   : {cfg.create_tool}\n  title  : {title}")
        print(f"  tags   : {args.tags}\n---- prompt ----\n{prompt}")
        return

    try:
        transport = HttpTransport(base_url=PROXY_BASE_URL, master_key=MASTER_KEY)
        tool_name = _resolve_tool_name(transport, cfg)
        if tool_name is None:
            return
        result = transport.post(
            _CALL_PATH,
            headers=transport.master,
            json=_ToolCallBody(name=tool_name, arguments=args),
            response_type=_ToolCallResult,
        )
        if isinstance(result, Success):
            print(f"bob_the_builder: created Devin session for {len(failures)} failure(s) [{tag}]")
            print(result.data.model_dump_json())
        else:
            print(f"bob_the_builder: Devin session call failed: {result}")
    except Exception as exc:  # noqa: BLE001 - remediation must never fail the run
        print(f"bob_the_builder: remediation error (ignored): {exc}")
