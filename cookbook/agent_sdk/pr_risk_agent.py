import argparse
import asyncio
import json
import sys
from typing import Final

from pydantic import BaseModel

from litellm.agent_sdk import PRRiskAgent, PRRiskAssessment, PullRequest


class _Arguments(BaseModel):
    title: str
    body: str
    changed_files: int
    additions: int
    deletions: int
    routine_model: str
    complex_model: str


def _arguments() -> _Arguments:
    parser: Final = argparse.ArgumentParser(description="Classify the deployment risk of a pull request")
    parser.add_argument("--title", required=True)
    parser.add_argument("--body", default="")
    parser.add_argument("--changed-files", type=int, required=True)
    parser.add_argument("--additions", type=int, required=True)
    parser.add_argument("--deletions", type=int, required=True)
    parser.add_argument("--routine-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--complex-model", default="anthropic/claude-opus-4-8")
    return _Arguments.model_validate(vars(parser.parse_args()))


async def _main() -> int:
    arguments: Final = _arguments()
    pull_request: Final = PullRequest(
        title=arguments.title,
        body=arguments.body,
        diff=sys.stdin.read(),
        changed_files=arguments.changed_files,
        additions=arguments.additions,
        deletions=arguments.deletions,
    )
    agent: Final = PRRiskAgent(
        routine_model=arguments.routine_model,
        complex_model=arguments.complex_model,
    )
    result: Final = await agent.classify(pull_request)
    if isinstance(result, PRRiskAssessment):
        print(result.model_dump_json(indent=2))  # noqa: T201  # command output is the CLI's public result
        return 0
    print(json.dumps({"error": result.error}), file=sys.stderr)  # noqa: T201  # command errors belong on stderr
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
