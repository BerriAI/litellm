import argparse
import os
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
from langchain_anthropic import ChatAnthropic

from sdk_watch.demo import run_demo
from sdk_watch.github import RestGitHub, already_handled, github_client
from sdk_watch.graph import Exact, OnDay, Selector, WatchDeps, run_watch
from sdk_watch.implement import agent_implement
from sdk_watch.report import DemoVerdict, PrDraft, pr_draft, summary
from sdk_watch.ship import Opened, Skipped, ping, ship
from sdk_watch.sources import WATCHED_SDKS, http_fetcher
from sdk_watch.triage import llm_triage

DEFAULT_REPOSITORY: Final = "BerriAI/litellm"


def build_parser() -> argparse.ArgumentParser:
    parser: Final = argparse.ArgumentParser(prog="sdk_watch")
    commands: Final = parser.add_subparsers(dest="command", required=True)
    watch: Final = commands.add_parser("watch", help="find SDK changes LiteLLM needs and write PR drafts")
    watch.add_argument("--repo", type=Path, required=True)
    watch.add_argument("--out", type=Path, required=True)
    watch.add_argument("--day", type=date.fromisoformat, help="release day to check, default yesterday in UTC")
    watch.add_argument("--sdk", help="check one release instead of a day, with --version")
    watch.add_argument("--version")
    watch.add_argument("--max-implementations", type=int, default=2)
    watch.add_argument("--skip-dedupe", action="store_true", help="replay an old release on an old checkout")
    watch.add_argument("--repository", default=DEFAULT_REPOSITORY)
    demo: Final = commands.add_parser("demo", help="run a PR draft's tests without and with its fix")
    demo.add_argument("--repo", type=Path, required=True)
    demo.add_argument("--change", type=Path, required=True)
    demo.add_argument("--python", type=Path, required=True)
    demo.add_argument("--verdict", type=Path, required=True)
    ship_parser: Final = commands.add_parser("ship", help="open the PR for a verified PR draft")
    ship_parser.add_argument("--repo", type=Path, required=True)
    ship_parser.add_argument("--change", type=Path, required=True)
    ship_parser.add_argument("--verdict", type=Path, required=True)
    ship_parser.add_argument("--gif", type=Path, required=True)
    ship_parser.add_argument("--base-sha", required=True, help="the commit the watch run checked out")
    ship_parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    return parser


def say(text: str) -> None:
    sys.stdout.write(f"{text}\n")


def selector(day: date | None, sdk: str | None, version: str | None) -> Selector:
    if sdk is not None and version is not None:
        return Exact(sdk=sdk, version=version)
    return OnDay(day=day or datetime.now(UTC).date() - timedelta(days=1))


def watch(args: argparse.Namespace) -> int:
    repo: Final = args.repo.resolve()
    model: Final = ChatAnthropic(
        model=os.environ.get("SDK_WATCH_MODEL", "sdk-watch"),
        anthropic_api_url=os.environ["LITELLM_PROXY_URL"],
        anthropic_api_key=os.environ["LITELLM_PROXY_KEY"],
        max_tokens=16000,
        max_retries=3,
        default_request_timeout=600,
    )
    with httpx.Client(timeout=60) as web, github_client(os.environ.get("GITHUB_TOKEN") or None) as api:
        fetch: Final = http_fetcher(web)
        deps: Final = WatchDeps(
            fetch_text=fetch,
            triage=llm_triage(model),
            already_handled=(lambda finding: False)
            if args.skip_dedupe
            else partial(already_handled, RestGitHub(client=api, repo=args.repository)),
            implement=agent_implement(model, repo, repo / ".venv" / "bin" / "python", fetch),
            sdks=WATCHED_SDKS,
            selector=selector(args.day, args.sdk, args.version),
            max_implementations=args.max_implementations,
        )
        state: Final = run_watch(deps)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.md").write_text(summary(state))
    for implementation in filter(lambda implementation: implementation.ships, state.implementations):
        draft = pr_draft(implementation)
        change_dir = args.out / draft.branch
        change_dir.mkdir(exist_ok=True)
        (change_dir / "change.patch").write_text(implementation.patch)
        (change_dir / "pr.json").write_text(draft.model_dump_json(indent=2))
    say(summary(state))
    return 0


def demo(args: argparse.Namespace) -> int:
    draft: Final = PrDraft.model_validate_json((args.change / "pr.json").read_text())
    verdict: Final = run_demo(args.repo.resolve(), (args.change / "change.patch").resolve(), draft, args.python, say)
    args.verdict.write_text(verdict.model_dump_json())
    return 0


def ship_command(args: argparse.Namespace) -> int:
    repo: Final = args.repo.resolve()
    draft: Final = PrDraft.model_validate_json((args.change / "pr.json").read_text())
    verdict: Final = DemoVerdict.model_validate_json(args.verdict.read_text())
    token: Final = os.environ["GITHUB_TOKEN"]
    with github_client(token) as api:
        result: Final = ship(
            repo=repo,
            repository=args.repository,
            base_sha=args.base_sha,
            patch=(args.change / "change.patch").resolve(),
            draft=draft,
            verdict=verdict,
            gif=args.gif,
            github=RestGitHub(client=api, repo=args.repository),
            token=token,
            run_url=os.environ.get("RUN_URL", ""),
        )
    match result:
        case Opened(url=url):
            webhook: Final = os.environ.get("SDK_WATCH_SLACK_WEBHOOK_URL")
            if webhook:
                ping(webhook, draft, result)
            say(f"Opened {url}" + ("" if webhook else " (no SDK_WATCH_SLACK_WEBHOOK_URL, so no Slack ping)"))
        case Skipped(reason=reason):
            say(f"Skipped {draft.branch}: {reason}")
    return 0


COMMANDS: Final = MappingProxyType({"watch": watch, "demo": demo, "ship": ship_command})


def main(argv: Sequence[str] | None = None) -> int:
    args: Final = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)
