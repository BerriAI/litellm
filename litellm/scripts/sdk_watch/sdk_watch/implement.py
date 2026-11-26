import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Final

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field

from sdk_watch.sources import Sdk
from sdk_watch.triage import Finding

OUTPUT_TAIL_CHARS: Final = 6000
DIFF_CHARS: Final = 40000
TEST_TIMEOUT_SECONDS: Final = 900

PERMISSIONS: Final = (
    FilesystemPermission(operations=["read", "write"], paths=["/**/.env*"], mode="deny"),
    FilesystemPermission(operations=["write"], paths=["/litellm/scripts/**"], mode="deny"),
    FilesystemPermission(
        operations=["write"],
        paths=["/litellm/**", "/tests/**", "/model_prices_and_context_window.json"],
        mode="allow",
    ),
    FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
)

IMPLEMENT_PROMPT: Final = """You work in a checkout of BerriAI/litellm, mounted at /. A provider SDK release added \
something LiteLLM may need to support. Decide whether LiteLLM already handles it, and when it does not, make the \
smallest correct change with a regression test.

1. Read /AGENTS.md and the AGENTS.md of every directory you edit, and follow them.
2. Find how LiteLLM handles the closest existing feature and whether this change is already supported. When it \
is, stop with outcome not_needed and say where it is handled.
3. Use fetch_sdk_diff on the commit linked in the release note to read the exact field names and types.
4. Never invent a fact you cannot see in the repo or the SDK diff, such as a model price or a context window. \
When the change needs one, stop with outcome not_needed and say what is missing.
5. Change only files under /litellm, /tests and /model_prices_and_context_window.json, never /litellm/scripts.
6. Add or extend a unit test under /tests/unit, in the test file that already covers the code you change, that \
fails without your change and passes with it. Tests never call the network. Run them with run_tests until \
they pass.
7. Return the report. pr_title is a conventional commit title. walkthrough is markdown for a reviewer: the SDK \
change with its link, what LiteLLM did before, what it does now, then each changed file and why."""


class Outcome(StrEnum):
    CHANGED = "changed"
    NOT_NEEDED = "not_needed"
    FAILED = "failed"


class ImplementationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    outcome: Outcome
    reason: str = Field(description="Why this outcome, naming the files that show it")
    pr_title: str = Field(description="Conventional commit title, e.g. fix(anthropic): keep mid-conversation system")
    walkthrough: str = Field(description="Markdown walkthrough of the change for a reviewer")
    test_files: tuple[str, ...] = Field(description="Repo-relative paths of the test files that prove the change")


@dataclass(frozen=True, slots=True)
class Implementation:
    finding: Finding
    report: ImplementationReport
    patch: str

    @property
    def ships(self) -> bool:
        return self.report.outcome is Outcome.CHANGED and bool(self.patch) and bool(self.report.test_files)


def is_test_path(path: str) -> bool:
    parts: Final = PurePosixPath(path).parts
    return len(parts) > 2 and parts[0] == "tests" and path.endswith(".py") and ".." not in parts


def pytest_command(python: Path, test_files: tuple[str, ...]) -> tuple[str, ...]:
    return (
        str(python),
        "-m",
        "pytest",
        *test_files,
        "-q",
        "-p",
        "no:cacheprovider",
        "--allow-hosts=127.0.0.1,::1,localhost",
    )


def test_env() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }


def run_tests_tool(repo: Path, python: Path) -> BaseTool:
    @tool
    def run_tests(test_files: list[str]) -> str:
        """Run pytest on repo-relative test files under tests/ and return the tail of the output."""
        rejected: Final = tuple(path for path in test_files if not is_test_path(path))
        if rejected or not test_files:
            return f"Refused: pass one or more .py files under tests/, got {test_files}"
        completed: Final = subprocess.run(
            pytest_command(python, tuple(test_files)),
            cwd=repo,
            env=test_env(),
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT_SECONDS,
            check=False,
        )
        output: Final = completed.stdout + completed.stderr
        return f"exit code {completed.returncode}\n{output[-OUTPUT_TAIL_CHARS:]}"

    return run_tests


def fetch_sdk_diff_tool(sdk: Sdk, fetch_text: Callable[[str], str]) -> BaseTool:
    @tool
    def fetch_sdk_diff(commit_sha: str, path_contains: str = "") -> str:
        """Fetch the diff of one commit in the provider SDK repo, optionally only files whose path contains a string."""
        if re.fullmatch(r"[0-9a-f]{7,40}", commit_sha) is None:
            return "Refused: commit_sha must be a hex commit sha"
        diff: Final = fetch_text(f"https://github.com/{sdk.repo}/commit/{commit_sha}.diff")
        chunks: Final = tuple(part for part in re.split(r"(?m)^(?=diff --git )", diff) if part.startswith("diff --git"))
        headers: Final = tuple(chunk.partition("\n")[0] for chunk in chunks)
        picked: Final = "".join(chunk for chunk, header in zip(chunks, headers, strict=True) if path_contains in header)
        return f"{len(headers)} files changed:\n" + "\n".join(headers) + "\n\n" + picked[:DIFF_CHARS]

    return fetch_sdk_diff


def is_clean(repo: Path) -> bool:
    return _git(repo, "status", "--porcelain") == ""


def collect_patch(repo: Path) -> str:
    _git(repo, "add", "--intent-to-add", "--all")
    patch: Final = _git(repo, "diff", "--binary")
    _git(repo, "reset", "--quiet")
    _git(repo, "checkout", "--quiet", "--", ".")
    _git(repo, "clean", "-fdq")
    return patch


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(("git", "-C", str(repo), *args), capture_output=True, text=True, check=True).stdout


def task_prompt(finding: Finding) -> str:
    candidate: Final = finding.candidate
    release: Final = candidate.release
    return (
        f"SDK: {candidate.sdk.name} ({candidate.sdk.repo}), release {release.version} on {release.released}: "
        f"{candidate.sdk.release_url(release.version)}\n"
        f"Release note: {finding.change.sdk_change}\n"
        f"Triage says LiteLLM needs ({finding.change.kind}): {finding.change.litellm_change}"
    )


def agent_implement(
    model: BaseChatModel, repo: Path, python: Path, fetch_text: Callable[[str], str]
) -> Callable[[Finding], Implementation]:
    def implement(finding: Finding) -> Implementation:
        agent: Final = create_deep_agent(
            model=model,
            tools=[run_tests_tool(repo, python), fetch_sdk_diff_tool(finding.candidate.sdk, fetch_text)],
            system_prompt=IMPLEMENT_PROMPT,
            backend=FilesystemBackend(root_dir=repo, virtual_mode=True),
            permissions=list(PERMISSIONS),
            response_format=ProviderStrategy(ImplementationReport),
            name="sdk-watch-implementer",
        )
        report: Final = _run_agent(agent, finding)
        patch: Final = collect_patch(repo)
        return Implementation(finding=finding, report=report, patch=patch if report.outcome is Outcome.CHANGED else "")

    return implement


def _run_agent(agent: CompiledStateGraph, finding: Finding) -> ImplementationReport:
    try:
        result: Final = agent.invoke(
            {"messages": [{"role": "user", "content": task_prompt(finding)}]}, config={"recursion_limit": 250}
        )
    except Exception as error:  # noqa: BLE001  # one failed change must not stop the others in the run
        return ImplementationReport(
            outcome=Outcome.FAILED,
            reason=f"{type(error).__name__}: {error}",
            pr_title="",
            walkthrough="",
            test_files=(),
        )
    return ImplementationReport.model_validate(result["structured_response"])
