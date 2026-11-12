"""Optional live traffic for the Task Insights demo: `python -m litellm.task_insights.demo_traffic`.

`send --batch discovery` and `send --batch heldout` send short synthetic coding requests through the gateway, so they
are billed and logged like real traffic. `export` writes the held-out requests without any assignment so a person can
label them, and `score` compares those labels with the persisted Jev assignments. The prompts are made up, and the
category each was written for is the author's intent, not a human label.
"""

import argparse
import asyncio
import csv
import sys
from collections.abc import Mapping
from typing import Final

from openai import AsyncOpenAI
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LIVE_PREFIX: Final = "task-insights-live-"

DISCOVERY_PROMPTS: Final = (
    "pytest fails with AssertionError in test_invoice_totals after I changed the rounding, why",
    "This jest test times out only in CI, how do I find the flaky part",
    "My integration test says connection refused to the test database, debug it",
    "Snapshot tests broke after a dependency upgrade, are the snapshots or the code wrong",
    "A unit test for the date helper fails on the first day of each month, find the bug",
    "test_user_signup passes alone but fails when the whole suite runs, why",
    "The e2e checkout test clicks the wrong button since the redesign, fix the selector",
    "Mocked HTTP calls in my tests return None instead of the fixture, what is wrong",
    "Add a REST endpoint that returns a user's last ten orders with pagination",
    "Implement a password reset flow that emails a one time link",
    "Add CSV export to the admin reports page",
    "Build a feature flag check so new billing UI only shows for beta accounts",
    "Add rate limiting of 100 requests per minute per API key",
    "Implement soft delete for projects so they can be restored within 30 days",
    "Add a websocket notification when a background job finishes",
    "Create a settings page where users can change their notification preferences",
    "Split this 800 line controller into smaller modules without changing behavior",
    "Replace these nested callbacks with async and await",
    "Rename the confusing variables in this pricing module and add type hints",
    "Deduplicate the three nearly identical validation functions",
    "Extract the repeated retry logic into one helper",
    "Convert this class component to a function component with hooks, same behavior",
    "Move hard coded config values into a typed settings object",
    "Explain what this regular expression matches and why it has a lookahead",
    "Walk me through how a request moves from the router to the database here",
    "What does this decorator do to the function it wraps",
    "Why does this function take a lock before reading the cache",
    "Explain the difference between these two SQL queries and which is faster",
    "Summarize what this module is responsible for",
    "How does this recursive tree walk avoid visiting a node twice",
    "Write unit tests for the discount calculator including edge cases",
    "Add tests for the CSV parser covering malformed rows",
    "Write an integration test for the login flow with a wrong password",
    "Increase test coverage for the permissions helper",
    "Add property based tests for the money rounding functions",
    "npm install fails with a peer dependency conflict after upgrading react",
    "The Docker build fails at pip install with gcc not found",
    "TypeScript cannot find module after I moved files, fix the imports",
    "CI lint step fails with an unknown rule after the eslint upgrade",
    "cargo build fails with a version conflict between two crates",
    "Write a commit message for these staged changes",
    "Update the README with setup steps for the new environment variables",
    "Add docstrings to the public functions in this module",
    "Draft release notes for the changes merged this week",
)

HELDOUT_PROMPTS: Final = (
    ("debug", "The login test fails with a 401 after the token change, find out why"),
    ("debug", "Why does this async test sometimes finish before the assertion runs"),
    ("debug", "Our cypress test fails because the modal is not visible yet, debug"),
    ("debug", "A test that compares floats fails on ARM machines only"),
    ("debug", "pytest says fixture not found after I renamed the conftest, fix it"),
    ("feature", "Add an endpoint to invite a teammate by email"),
    ("feature", "Implement dark mode with a toggle in the header"),
    ("feature", "Add two factor authentication with TOTP codes"),
    ("feature", "Build an audit log page that lists admin actions"),
    ("feature", "Add a bulk import for products from a spreadsheet"),
    ("refactor", "Break this long function into smaller pure functions"),
    ("refactor", "Replace the global variables in this module with dependency injection"),
    ("refactor", "Simplify these nested if statements without changing the results"),
    ("refactor", "Turn this switch statement into a lookup table"),
    ("refactor", "Remove the dead code paths in this handler"),
    ("explain", "What does this generator function yield and when does it stop"),
    ("explain", "Explain how this middleware decides whether to cache a response"),
    ("explain", "Why is this list comprehension slower than the loop"),
    ("explain", "How do these two services talk to each other"),
    ("tests", "Write tests for the shopping cart total including coupons"),
    ("tests", "Add tests that cover the error branches of the upload handler"),
    ("tests", "Create a test for the scheduler that runs at midnight"),
    ("build", "Webpack fails with out of memory during the production build"),
    ("build", "poetry lock fails with a resolver conflict on numpy"),
    ("build", "The GitHub Actions job fails because node 18 is no longer available"),
    ("docs", "Write a pull request description for this change"),
    ("docs", "Add a section to the docs explaining the new config options"),
    ("off_topic", "Write a haiku about the ocean"),
    ("off_topic", "What should I cook for dinner tonight"),
    ("off_topic", "Translate good morning into French and German"),
)


class TrafficSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LITELLM_TASK_INSIGHTS_", frozen=True, extra="ignore")

    gateway_url: str = "http://litellm:4000/v1"
    gateway_key: SecretStr | None = None
    traffic_model: str = "claude-haiku-5-5"
    traffic_max_tokens: int = Field(default=48, ge=1, le=512)


def batch_prompts(batch: str) -> tuple[tuple[str, str], ...]:
    """Request id and prompt for each request of a batch. Ids are stable so a rerun is recognizable."""
    if batch == "discovery":
        return tuple((f"{LIVE_PREFIX}discovery-{index:03d}", prompt) for index, prompt in enumerate(DISCOVERY_PROMPTS))
    return tuple((f"{LIVE_PREFIX}heldout-{index:03d}", prompt) for index, (_, prompt) in enumerate(HELDOUT_PROMPTS))


async def send(batch: str, settings: TrafficSettings) -> int:
    if settings.gateway_key is None:
        print("Set LITELLM_TASK_INSIGHTS_GATEWAY_KEY to send live traffic", file=sys.stderr)  # noqa: T201  # CLI output
        return 1
    client: Final = AsyncOpenAI(base_url=settings.gateway_url, api_key=settings.gateway_key.get_secret_value())
    for request_id, prompt in batch_prompts(batch):
        await client.chat.completions.create(
            model=settings.traffic_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=settings.traffic_max_tokens,
            extra_headers={"x-litellm-session-id": request_id},
        )
    print(f"Sent {len(batch_prompts(batch))} {batch} requests to {settings.traffic_model}")  # noqa: T201  # CLI output
    return 0


def export_heldout(path: str) -> None:
    """The file has no assignment column, so whoever labels it does so without seeing Jev's answer."""
    with open(path, "w", newline="") as handle:
        writer: Final = csv.writer(handle)
        writer.writerow(("prompt", "label"))
        writer.writerows((prompt, "") for _, prompt in HELDOUT_PROMPTS)


def agreement(labels: Mapping[str, str], assigned: Mapping[str, str | None]) -> tuple[int, int, int]:
    """Labeled prompts, prompts Jev assigned to a task, and assignments matching the label, where a label of
    `unknown` matches an unknown outcome."""
    labeled: Final = tuple(prompt for prompt, label in labels.items() if label)
    covered: Final = sum(1 for prompt in labeled if assigned.get(prompt) is not None)
    agreed: Final = sum(1 for prompt in labeled if (assigned.get(prompt) or "unknown") == labels[prompt])
    return len(labeled), covered, agreed


def read_labels(path: str) -> Mapping[str, str]:
    with open(path, newline="") as handle:
        return {row["prompt"]: row["label"].strip() for row in csv.DictReader(handle)}


async def score(labels: Mapping[str, str]) -> int:
    from litellm.task_insights.database import connected_database

    async with connected_database() as database:
        rows: Final = await database.query_raw(
            """SELECT DISTINCT ON (o.request_text) o.request_text, d.name
            FROM "LiteLLM_TaskObservation" o
            LEFT JOIN "LiteLLM_TaskAssignment" a ON a.request_id = o.request_id
            LEFT JOIN "LiteLLM_TaskDefinition" d ON d.task_id = a.task_id AND d.status = 'frozen'
            WHERE o.request_text = ANY($1::text[]) ORDER BY o.request_text, o.start_time DESC""",
            [prompt for _, prompt in HELDOUT_PROMPTS],
        )
    assigned: Final = {str(row["request_text"]): row["name"] if isinstance(row["name"], str) else None for row in rows}
    total, covered, agreed = agreement(labels, assigned)
    print(  # noqa: T201  # CLI output
        f"labeled={total} assigned={covered} coverage={covered / total if total else 0:.2f} "
        f"agreement={agreed / total if total else 0:.2f}"
    )
    return 0


def main() -> None:
    parser: Final = argparse.ArgumentParser(description="Optional live traffic for the Task Insights demo")
    commands: Final = parser.add_subparsers(dest="command", required=True)
    send_command: Final = commands.add_parser("send", help="Send one batch through the gateway")
    send_command.add_argument("--batch", choices=("discovery", "heldout"), required=True)
    commands.add_parser("export", help="Write the held-out prompts for labeling").add_argument("path")
    commands.add_parser("score", help="Score labels against the persisted assignments").add_argument("path")
    args: Final = vars(parser.parse_args())
    command: Final = str(args["command"])
    if command == "send":
        raise SystemExit(asyncio.run(send(str(args["batch"]), TrafficSettings())))
    if command == "export":
        export_heldout(str(args["path"]))
        return
    raise SystemExit(asyncio.run(score(read_labels(str(args["path"])))))


if __name__ == "__main__":
    main()
