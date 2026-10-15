"""One SDK process for the per-user GitHub Copilot metric cells: a provider 401 on a caller's own Copilot session
and one on the deployment's own key, then a report of the deployment series on that process's default registry.
Run as a script by ``run_copilot_session_worker`` so each process registers the Prometheus metrics exactly once."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Final

import litellm
import openai
from litellm.integrations.prometheus import PrometheusLogger
from litellm.llms.github_copilot.per_user_auth import GithubCopilotUserSession
from prometheus_client import REGISTRY
from pydantic import TypeAdapter

COPILOT_DEPLOYMENT: Final = "copilot-per-user-deployment"
OPERATOR_DEPLOYMENT: Final = "operator-key-deployment"
CALLER_SESSION_TOKEN: Final = "caller-copilot-session-token"
OPERATOR_KEY: Final = "rotated-operator-key"
FAILURES: Final = "litellm_deployment_failure_responses_total"
TOTALS: Final = "litellm_deployment_total_requests_total"
LLM_FAILED: Final = "litellm_llm_api_failed_requests_metric_total"
STATE: Final = "litellm_deployment_state"
DEPLOYMENT_SERIES: Final = frozenset({FAILURES, TOTALS, LLM_FAILED, STATE})
WORKER_TIMEOUT_SECONDS: Final = 120


@dataclass(frozen=True, slots=True)
class Outcome:
    deployment: str
    status: int | None


@dataclass(frozen=True, slots=True)
class Series:
    metric: str
    model_id: str
    value: float


@dataclass(frozen=True, slots=True)
class WorkerReport:
    outcomes: tuple[Outcome, ...]
    series: tuple[Series, ...]

    def values(self, metric: str, model_id: str) -> tuple[float, ...]:
        return tuple(s.value for s in self.series if s.metric == metric and s.model_id == model_id)


_REPORT: Final = TypeAdapter(WorkerReport)


def run_copilot_session_worker(api_base: str) -> WorkerReport:
    completed: Final = subprocess.run(
        (sys.executable, "-P", __file__, api_base),
        capture_output=True,
        text=True,
        timeout=WORKER_TIMEOUT_SECONDS,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    return _REPORT.validate_json(completed.stdout.splitlines()[-1])


def _deployment_series() -> tuple[Series, ...]:
    return tuple(
        Series(metric=sample.name, model_id=str(sample.labels.get("model_id", "")), value=float(sample.value))
        for family in REGISTRY.collect()
        for sample in family.samples
        if sample.name in DEPLOYMENT_SERIES
    )


def _router(api_base: str) -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": COPILOT_DEPLOYMENT,
                "litellm_params": {
                    "model": "github_copilot/gpt-4o",
                    "litellm_credential_name": "copilot-per-user",
                    "github_copilot_auth_type": "per_user_oauth",
                },
                "model_info": {"id": COPILOT_DEPLOYMENT},
            },
            {
                "model_name": OPERATOR_DEPLOYMENT,
                "litellm_params": {"model": "openai/gpt-5.6", "api_base": api_base, "api_key": OPERATOR_KEY},
                "model_info": {"id": OPERATOR_DEPLOYMENT},
            },
        ],
        num_retries=0,
    )


async def _attempt(router: litellm.Router, model: str, **extra: object) -> Outcome:
    try:
        await router.acompletion(model=model, messages=[{"role": "user", "content": f"{model} worker"}], **extra)
    except openai.APIStatusError as error:
        return Outcome(deployment=model, status=error.status_code)
    return Outcome(deployment=model, status=None)


async def main(api_base: str) -> None:
    litellm.callbacks = [PrometheusLogger()]  # test-quality-ok: this worker process exits right after the report
    session: Final = GithubCopilotUserSession(token=CALLER_SESSION_TOKEN, api_base=api_base)
    router: Final = _router(api_base)
    outcomes: Final = (
        await _attempt(router, COPILOT_DEPLOYMENT, github_copilot_user_session=session),
        await _attempt(router, OPERATOR_DEPLOYMENT),
    )
    report: Final = WorkerReport(outcomes=outcomes, series=_deployment_series())
    sys.stdout.write(json.dumps(_REPORT.dump_python(report)) + "\n")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
