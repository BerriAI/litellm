import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol, cast

import httpx
import litellm
from pydantic import TypeAdapter

from litellm import Router
from litellm.router_strategy.complexity_router.complexity_router import (
    ComplexityRouter,
    _JEV_TIER_CRITERIA,
)
from litellm.router_strategy.complexity_router.config import (
    ComplexityRouterConfig,
    OpenSourceClassifierConfig,
)
from litellm.router_strategy.complexity_router.jev_classifier import (
    DEFAULT_JEV_INSTRUCTIONS,
    JevChoiceAnswer,
    JevClassifierClient,
    JevSystemOneRequest,
    JevSystemOneResponse,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO = Path("/home/ubuntu/repos/litellm")
PINNED_SOURCE = Path(os.environ.get("ROUTING_POC_PINNED_SOURCE", str(ROOT / "litellm_src_pin")))
PINNED_TREE_COMMIT = "b88f64e65f29a34ee7363a3e2cd8189e850e4387"
PROMPT_PATH = ROOT / "heldout_baseline_prompt.json"
PROMPT_SHA_PATH = ROOT / "heldout_baseline_prompt.sha256"
BASELINE_LABEL = "current Jev complexity baseline (not a historical rerun)"
REQUEST_TIMEOUT_S = 120.0
TIER_MODELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "SIMPLE": "haiku-4.5",
        "MEDIUM": "sonnet-5",
        "COMPLEX": "opus-5",
        "REASONING": "opus-5",
        "NON_REASONING": "haiku-4.5",
    }
)


@dataclass(frozen=True, slots=True)
class BaselineCall:
    call_id: str | None
    cost_usd: float | None
    latency_s: float
    error: str | None


@dataclass(frozen=True, slots=True)
class BaselineRoute:
    tier: str | None
    model: str
    reason: str


class CaptureClient:
    def __init__(self) -> None:
        self.request: JevSystemOneRequest | None = None

    async def evaluate(
        self,
        request: JevSystemOneRequest,
        timeout_s: float,
        request_kwargs: Mapping[str, object] | None = None,
    ) -> JevSystemOneResponse:
        self.request = request
        answer = JevChoiceAnswer(
            type="choice",
            choice="SIMPLE",
            probabilities=MappingProxyType({"SIMPLE": 1.0}),
            confidence=1.0,
        )
        return JevSystemOneResponse(answers=MappingProxyType({"tier": answer}))


class AsyncPostClient(Protocol):
    async def post(
        self,
        url: str,
        *,
        json: Mapping[str, object],
        headers: Mapping[str, str],
        timeout: float,
    ) -> httpx.Response: ...


class GatewayJevClassifierClient:
    def __init__(self, client: AsyncPostClient, api_base: str, api_key: str) -> None:
        self._client = client
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self.last_call: BaselineCall | None = None
        self.last_request: Mapping[str, object] | None = None

    async def evaluate(
        self,
        request: JevSystemOneRequest,
        timeout_s: float,
        request_kwargs: Mapping[str, object] | None = None,
    ) -> JevSystemOneResponse:
        body: Final = request.model_dump(mode="json")
        self.last_request = body
        started: Final = time.perf_counter()
        try:
            response: Final = await self._client.post(
                f"{self._api_base}/v1/systemone",
                json=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=REQUEST_TIMEOUT_S,
            )
            response.raise_for_status()
            payload: Final = TypeAdapter(dict[str, object]).validate_python(response.json())
            cost: Final = _header_cost(response.headers)
            self.last_call = BaselineCall(
                call_id=response.headers.get("x-litellm-call-id"),
                cost_usd=cost,
                latency_s=time.perf_counter() - started,
                error=None,
            )
            return TypeAdapter(JevSystemOneResponse).validate_python(
                {
                    **payload,
                    "classifier_cost": cost,
                    "classifier_provider": "typesafe",
                }
            )
        except Exception as error:
            self.last_call = BaselineCall(
                call_id=None,
                cost_usd=None,
                latency_s=time.perf_counter() - started,
                error=type(error).__name__,
            )
            raise


def _header_cost(headers: Mapping[str, str]) -> float | None:
    raw: Final = headers.get("x-litellm-response-cost") or headers.get(
        "llm_provider-x-litellm-response-cost"
    )
    try:
        value: Final = float(raw) if raw is not None else None
    except (TypeError, ValueError, OverflowError):
        return None
    if value is None or value < 0 or not value < float("inf"):
        return None
    return value


def source_config() -> ComplexityRouterConfig:
    if PINNED_SOURCE not in Path(litellm.__file__).resolve().parents:
        raise RuntimeError("load the pinned LiteLLM tree through PYTHONPATH before using the Jev baseline")
    if _git_sha(PINNED_SOURCE) != PINNED_TREE_COMMIT:
        raise RuntimeError("the Jev baseline must use the pinned complexity-router tree")
    classifier: Final = OpenSourceClassifierConfig(
        provider="typesafe",
        model="typesafe/jev-latest",
        timeout_ms=120_000,
    )
    return ComplexityRouterConfig(
        classifier_type="oss_classifier",
        opensource_classifier_config=classifier,
    )


def source_router(client: JevClassifierClient) -> tuple[ComplexityRouter, ComplexityRouterConfig]:
    config: Final = source_config()
    router: Final = ComplexityRouter(
        model_name="heldout-baseline",
        litellm_router_instance=cast(Router, object()),
        complexity_router_config=config.model_dump(exclude_none=True),
        jev_client=client,
    )
    return router, config


def active_tiers(config: ComplexityRouterConfig) -> tuple[tuple[str, str], ...]:
    return tuple((tier.value, label) for tier, label in config.labeled_tiers())


def choose_baseline_route(
    tier: str | None,
    active: Sequence[tuple[str, str]],
    classifier_failed: bool = False,
) -> BaselineRoute:
    if classifier_failed:
        return BaselineRoute(tier=tier, model="sonnet-5", reason="classifier_failure")
    active_names: Final = frozenset(name.upper() for name, _ in active)
    tier_name: Final = tier.upper() if isinstance(tier, str) else None
    model: Final = TIER_MODELS.get(tier_name or "")
    if tier_name not in active_names or model is None:
        return BaselineRoute(tier=tier_name, model="sonnet-5", reason="unknown_tier")
    return BaselineRoute(tier=tier_name, model=model, reason=f"tier:{tier_name}")


def _git_sha(path: Path) -> str:
    result: Final = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def render_prompt_artifact(statement: str, task_id: str) -> tuple[str, str]:
    capture: Final = CaptureClient()
    router, config = source_router(capture)
    outcome: Final = asyncio.run(router.aclassify(prompt=statement))
    if outcome.cause != "jev_classifier" or capture.request is None:
        raise RuntimeError("pinned source router did not build a Jev request")
    body: Final = capture.request.model_dump(mode="json")
    if not isinstance(body.get("state"), str):
        raise ValueError("pinned source request is missing its classifier state")
    questions: Final[object] = body.get("questions")
    tier_question: Final[object] = questions.get("tier") if isinstance(questions, Mapping) else None
    if not isinstance(tier_question, Mapping):
        raise ValueError("pinned source request is missing its tier question")
    active: Final = active_tiers(config)
    expected_criteria: Final = {
        label: _JEV_TIER_CRITERIA[tier.upper()]
        for tier, label in active
    }
    if tier_question.get("instructions") != DEFAULT_JEV_INSTRUCTIONS:
        raise ValueError("pinned source request did not use DEFAULT_JEV_INSTRUCTIONS")
    if tier_question.get("criteria") != expected_criteria:
        raise ValueError("pinned source request criteria do not match active labeled tiers")
    pinned_sha: Final = _git_sha(PINNED_SOURCE)
    if pinned_sha != PINNED_TREE_COMMIT:
        raise ValueError(f"pinned complexity-router source changed: {pinned_sha}")
    payload: Final = {
        "baseline": BASELINE_LABEL,
        "source_repository_sha": _git_sha(SOURCE_REPO),
        "complexity_router_tree_commit": pinned_sha,
        "sample_instance_id": task_id,
        "active_tiers": [{"tier": tier, "label": label} for tier, label in active],
        "non_reasoning_active": any(tier.upper() == "NON_REASONING" for tier, _ in active),
        "tier_criteria_source": "_JEV_TIER_CRITERIA",
        "request": body,
    }
    content: Final = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if PROMPT_PATH.exists() or PROMPT_SHA_PATH.exists():
        raise FileExistsError("refusing to overwrite the frozen baseline prompt")
    PROMPT_PATH.write_text(content)
    digest: Final = hashlib.sha256(content.encode()).hexdigest()
    PROMPT_SHA_PATH.write_text(f"{digest}  {PROMPT_PATH.name}\n")
    return content, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--task-id")
    arguments = parser.parse_args()
    if not arguments.render or not isinstance(arguments.task_id, str):
        parser.error("--render and --task-id are required")
    tasks: Final[object] = json.loads((ROOT / "heldout_tasks.json").read_text())
    task = next(
        entry
        for entry in tasks["tasks"]
        if entry["instance_id"] == arguments.task_id
    )
    rows: Final = tuple(
        json.loads(line)
        for line in (ROOT / "heldout_agent_dataset" / "train.jsonl").read_text().splitlines()
    )
    statement: Final = next(row["problem_statement"] for row in rows if row["instance_id"] == arguments.task_id)
    _, digest = render_prompt_artifact(statement, arguments.task_id)
    print(json.dumps({"baseline_prompt_sha256": digest, "sample_instance_id": task["instance_id"]}))


if __name__ == "__main__":
    main()
