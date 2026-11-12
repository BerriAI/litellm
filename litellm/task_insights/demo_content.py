"""Synthetic sample data for the Task Insights demo. Every request, cost and assignment here is made up."""

from dataclasses import dataclass
from typing import Final

from litellm.models.task_insights import TaskCapability

DEMO_PREFIX: Final = "task-insights-demo-"


@dataclass(frozen=True, slots=True)
class DemoTask:
    key: str
    name: str
    description: str
    inclusion: str
    exclusion: str
    capabilities: tuple[TaskCapability, ...]
    center: tuple[float, float]
    weight: float
    cost_factor: float
    model_mix: tuple[float, float, float]
    instructions: tuple[str, ...]
    follow_ups: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DemoModel:
    key: str
    model_name: str
    provider_model: str
    base_cost: float
    base_latency_ms: float


DEMO_MODELS: Final = (
    DemoModel("opus", "claude-opus-5-5", "anthropic/claude-opus-5-5", 0.042, 9_800),
    DemoModel("sonnet", "claude-sonnet-5-5", "anthropic/claude-sonnet-5-5", 0.016, 5_400),
    DemoModel("haiku", "claude-haiku-5-5", "anthropic/claude-haiku-5-5", 0.0035, 1_900),
)

DEMO_TEAMS: Final = (
    (f"{DEMO_PREFIX}team-platform", "Sample: Platform"),
    (f"{DEMO_PREFIX}team-mobile", "Sample: Mobile"),
    (f"{DEMO_PREFIX}team-data", "Sample: Data"),
)

DEMO_TASKS: Final = (
    DemoTask(
        key="debug-tests",
        name="Debug failing tests",
        description="Find out why a test or test suite fails and fix the cause.",
        inclusion="The request reports a failing, flaky or erroring test and asks for the cause or a fix.",
        exclusion="Writing new tests for working code, or build and dependency failures outside the tests.",
        capabilities=("function_calling",),
        center=(24.0, 30.0),
        weight=0.22,
        cost_factor=1.3,
        model_mix=(0.35, 0.5, 0.15),
        instructions=(
            "The checkout test fails with an assertion error on the order total, find out why and fix it",
            "This integration test passes locally but fails in CI about one run in five, figure out the flake",
            "pytest reports a KeyError in the fixture setup for the payments tests, please debug",
            "After my last change three snapshot tests fail, check whether the snapshots or the code are wrong",
            "The date parsing test fails only in the UTC+13 timezone, find the bug",
            "Our end to end login test times out waiting for the redirect, investigate",
        ),
        follow_ups=(
            "That fixed one case but the second test still fails with the same error",
            "Can you also make the test deterministic so it stops flaking",
        ),
    ),
    DemoTask(
        key="implement-feature",
        name="Implement a feature from a spec",
        description="Build new behavior described in a ticket, spec or plain-language request.",
        inclusion="The request asks for new user-facing or API behavior that does not exist yet.",
        exclusion="Restructuring existing behavior without changing it, or fixing a reported bug.",
        capabilities=("function_calling",),
        center=(70.0, 26.0),
        weight=0.24,
        cost_factor=2.1,
        model_mix=(0.55, 0.4, 0.05),
        instructions=(
            "Add pagination to the orders endpoint with a cursor and a page size limit of 100",
            "Implement CSV export for the reports page following the spec in the ticket",
            "Add a dark mode toggle to the settings screen and persist the choice",
            "Build a retry with exponential backoff for the webhook sender, max five attempts",
            "Add an endpoint that lets admins deactivate a user and revokes their sessions",
            "Implement rate limiting per API key using the existing redis client",
        ),
        follow_ups=(
            "Looks good, now add validation for the page size parameter",
            "Please also update the OpenAPI schema for the new endpoint",
        ),
    ),
    DemoTask(
        key="refactor",
        name="Refactor and clean up code",
        description="Restructure existing code for readability or maintainability without changing behavior.",
        inclusion="The request asks to simplify, split, rename, deduplicate or modernize code that already works.",
        exclusion="Adding new behavior or fixing a bug, even when the code is touched along the way.",
        capabilities=("function_calling",),
        center=(80.0, 62.0),
        weight=0.14,
        cost_factor=1.6,
        model_mix=(0.3, 0.6, 0.1),
        instructions=(
            "This 600 line service class does too much, split it into smaller modules without changing behavior",
            "Replace the nested callbacks in the upload handler with async and await",
            "Deduplicate the three nearly identical validation helpers into one",
            "Rename the confusing variables in the billing module and add type hints",
            "Move the hard coded configuration values into a settings object",
        ),
        follow_ups=("Keep the public function names the same so callers do not break",),
    ),
    DemoTask(
        key="explain-code",
        name="Explain unfamiliar code",
        description="Describe what existing code does, how parts connect, or why it was written a certain way.",
        inclusion="The request asks a question about how code works and expects an explanation, not a change.",
        exclusion="Requests that ask for any modification to the code.",
        capabilities=(),
        center=(46.0, 78.0),
        weight=0.16,
        cost_factor=0.55,
        model_mix=(0.1, 0.35, 0.55),
        instructions=(
            "Explain how the authentication middleware decides which routes need a token",
            "What does this regular expression in the log parser actually match",
            "Walk me through how a request flows from the router to the database layer",
            "Why does this cache invalidation function take a lock before reading",
            "Summarize what the migration scripts folder does and in which order they run",
        ),
        follow_ups=("And what happens when the token is expired",),
    ),
    DemoTask(
        key="write-tests",
        name="Write or extend tests",
        description="Add new tests or extend coverage for code that already works.",
        inclusion="The request asks for new unit, integration or end to end tests, or more coverage.",
        exclusion="Fixing tests that already fail, which is debugging.",
        capabilities=("function_calling",),
        center=(16.0, 62.0),
        weight=0.11,
        cost_factor=1.1,
        model_mix=(0.15, 0.55, 0.3),
        instructions=(
            "Write unit tests for the discount calculator covering the edge cases",
            "Add an integration test for the password reset flow",
            "Increase coverage of the CSV parser, it has no tests for malformed rows",
            "Write property based tests for the money rounding helpers",
        ),
        follow_ups=("Also cover the case where the input list is empty",),
    ),
    DemoTask(
        key="build-errors",
        name="Fix build and dependency errors",
        description="Get a project compiling, installing or deploying again after a toolchain or dependency problem.",
        inclusion="The request shows a compiler, bundler, package manager or CI pipeline error outside the tests.",
        exclusion="Test assertion failures, and runtime bugs in application logic.",
        capabilities=("function_calling",),
        center=(50.0, 48.0),
        weight=0.08,
        cost_factor=0.9,
        model_mix=(0.2, 0.5, 0.3),
        instructions=(
            "npm install fails with a peer dependency conflict after upgrading the router package",
            "The Docker build fails at the pip install step with a missing compiler error",
            "TypeScript build broke with cannot find module after moving files, fix the imports",
            "CI fails on the lint step with an unknown rule after the config update",
        ),
        follow_ups=("Now the build passes but there is a deprecation warning, can we fix that too",),
    ),
    DemoTask(
        key="docs-commits",
        name="Write docs and commit messages",
        description="Produce prose about code: READMEs, docstrings, changelogs, commit or pull request text.",
        inclusion="The request asks for written documentation or a message describing a change.",
        exclusion="Explaining code to the requester in conversation, or changing the code itself.",
        capabilities=(),
        center=(84.0, 88.0),
        weight=0.05,
        cost_factor=0.35,
        model_mix=(0.05, 0.25, 0.7),
        instructions=(
            "Write a commit message for the staged changes",
            "Update the README with setup steps for the new environment variables",
            "Add docstrings to the public functions in the export module",
            "Draft release notes for this week's merged changes",
        ),
        follow_ups=("Make it shorter, one line summary and a short body",),
    ),
)

OFF_TOPIC_REQUESTS: Final = (
    "Write a short poem about autumn for my team newsletter",
    "What is a good recipe for a quick vegetarian dinner",
    "Translate this paragraph about our holiday schedule into Spanish",
    "Summarize the plot of a famous novel in three sentences",
)
