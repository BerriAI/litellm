"""Seed or reset the Task Insights sample data: `python -m litellm.task_insights.demo_seed [--reset]`.

The sample traffic goes into the real request log and ledger tables as already-applied deliveries, so it adds nothing
to spend totals, budgets or the Usage charts. The real observation projection then reads it, and the sample taxonomy
and assignments go through the same freeze and save paths the analysis worker uses. No model is called. Rerunning
replaces the sample data, with fresh dates, and leaves every other record alone.
"""

import argparse
import asyncio
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from itertools import chain
from typing import Final

from litellm.models.task_insights import (
    TASK_INSIGHTS_ANALYSIS_TAG,
    Decision,
    DiscoveredTask,
    MapPlacement,
)
from litellm.repositories.task_insights_demo_repository import TaskInsightsDemoRepository
from litellm.repositories.task_insights_repository import TaskInsightsDatabase, TaskInsightsRepository
from litellm.task_insights.demo_content import (
    DEMO_MODELS,
    DEMO_PREFIX,
    DEMO_TASKS,
    DEMO_TEAMS,
    OFF_TOPIC_REQUESTS,
    DemoModel,
    DemoTask,
)
from litellm.task_insights.observation import observe_applied_deliveries

SESSION_COUNT: Final = 110
_WINDOW_DAYS: Final = 14
_SAMPLE_MODEL: Final = "sample-precomputed"
_SAMPLE_PROVENANCE: Final = {"source": "sample", "note": "Precomputed sample data, no model call"}


def unit(*parts: object) -> float:
    """A deterministic value in [0, 1) for the given key, so every sample row is reproducible on its own."""
    digest: Final = sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _pick(weights: tuple[float, ...], value: float) -> int:
    total: Final = sum(weights)
    thresholds: Final = tuple(sum(weights[: index + 1]) / total for index in range(len(weights)))
    return next((index for index, threshold in enumerate(thresholds) if value < threshold), len(weights) - 1)


@dataclass(frozen=True, slots=True)
class DemoRequest:
    request_id: str
    session_id: str
    team_id: str
    task: DemoTask | None
    model: DemoModel
    start: datetime
    instruction: str
    latest_text: str
    stored_prompt: bool
    spend: float | None
    latency_ms: int | None
    status: str
    retried_on: DemoModel | None
    analysis: bool
    turn: int


def _session_requests(index: int, now: datetime) -> tuple[DemoRequest, ...]:
    session: Final = f"{DEMO_PREFIX}session-{index:03d}"
    off_topic: Final = unit(session, "off-topic") < 0.05
    task: Final = (
        None if off_topic else DEMO_TASKS[_pick(tuple(task.weight for task in DEMO_TASKS), unit(session, "task"))]
    )
    instructions: Final = OFF_TOPIC_REQUESTS if task is None else task.instructions
    instruction: Final = instructions[int(unit(session, "instruction") * len(instructions))]
    mix: Final = (0.2, 0.4, 0.4) if task is None else task.model_mix
    model: Final = DEMO_MODELS[_pick(mix, unit(session, "model"))]
    team: Final = DEMO_TEAMS[int(unit(session, "team") * len(DEMO_TEAMS))][0]
    turns: Final = 1 if task is None else 1 + int(unit(session, "turns") ** 2 * 8)
    started: Final = now - timedelta(days=_WINDOW_DAYS * unit(session, "start"), minutes=10)
    follow_up_turn: Final = 2 + int(unit(session, "follow-up-turn") * 4) if task is not None and task.follow_ups else 0
    return tuple(
        _request(session, team, task, model, instruction, started, turn, follow_up_turn) for turn in range(turns)
    )


def _request(
    session: str,
    team: str,
    task: DemoTask | None,
    model: DemoModel,
    instruction: str,
    started: datetime,
    turn: int,
    follow_up_turn: int,
) -> DemoRequest:
    request_id: Final = f"{DEMO_PREFIX}{session.removeprefix(DEMO_PREFIX)}-turn-{turn}"
    follow_ups: Final = task.follow_ups if task is not None else ()
    latest: Final = (
        follow_ups[int(unit(request_id, "follow-up") * len(follow_ups))]
        if follow_ups and follow_up_turn and turn >= follow_up_turn
        else instruction
    )
    failed: Final = unit(request_id, "failed") < 0.05
    cost_factor: Final = task.cost_factor if task is not None else 0.4
    cost: Final = round(model.base_cost * cost_factor * (0.5 + unit(request_id, "cost")) * (1 + 0.15 * turn), 6)
    latency: Final = model.base_latency_ms * cost_factor**0.5 * (0.6 + 0.9 * unit(request_id, "latency") ** 2)
    return DemoRequest(
        request_id=request_id,
        session_id=session,
        team_id=team,
        task=task,
        model=model,
        start=started + timedelta(minutes=3 * turn),
        instruction=instruction,
        latest_text=latest,
        stored_prompt=unit(request_id, "prompt-stored") >= 0.06,
        spend=None if unit(request_id, "cost-missing") < 0.03 else (round(cost * 0.2, 6) if failed else cost),
        latency_ms=None if unit(request_id, "latency-missing") < 0.04 or failed else int(latency),
        status="failure" if failed else "success",
        retried_on=DEMO_MODELS[1] if model is DEMO_MODELS[0] and unit(request_id, "retry") < 0.08 else None,
        analysis=False,
        turn=turn,
    )


def _analysis_requests(now: datetime) -> tuple[DemoRequest, ...]:
    model: Final = DEMO_MODELS[2]
    return tuple(
        DemoRequest(
            request_id=f"{DEMO_PREFIX}analysis-{index:02d}",
            session_id=f"{DEMO_PREFIX}analysis",
            team_id=DEMO_TEAMS[0][0],
            task=None,
            model=model,
            start=now - timedelta(days=1, minutes=index),
            instruction="Distill these requests into one-line intents",
            latest_text="Distill these requests into one-line intents",
            stored_prompt=True,
            spend=0.0011,
            latency_ms=1400,
            status="success",
            retried_on=None,
            analysis=True,
            turn=0,
        )
        for index in range(8)
    )


def demo_requests(now: datetime) -> tuple[DemoRequest, ...]:
    sessions: Final = tuple(_session_requests(index, now) for index in range(SESSION_COUNT))
    return (*chain.from_iterable(sessions), *_analysis_requests(now))


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec="milliseconds")


def _tool_step(step: int) -> tuple[Mapping[str, object], ...]:
    return (
        {"role": "assistant", "content": [{"type": "tool_use", "id": f"call-{step}", "name": "run", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"call-{step}", "content": "ok"}]},
    )


def _messages(request: DemoRequest) -> tuple[Mapping[str, object], ...]:
    tool_turns: Final = tuple(chain.from_iterable(_tool_step(step) for step in range(request.turn)))
    follow_up: Final = (
        ({"role": "user", "content": request.latest_text},) if request.latest_text != request.instruction else ()
    )
    return ({"role": "user", "content": request.instruction}, *tool_turns, *follow_up)


def _payload(request: DemoRequest, delivery_model: DemoModel, *, failed_attempt: bool) -> Mapping[str, object]:
    spend: Final = round(delivery_model.base_cost * 0.1, 6) if failed_attempt else request.spend
    status: Final = "failure" if failed_attempt else request.status
    return {
        "request_id": request.request_id,
        "call_type": "anthropic_messages",
        "api_key": f"{DEMO_PREFIX}key",
        **({"spend": spend} if spend is not None else {}),
        "prompt_tokens": 1800 + 400 * request.turn,
        "completion_tokens": 350,
        "startTime": _iso(request.start),
        "model": delivery_model.provider_model.split("/", 1)[1],
        "model_group": delivery_model.model_name,
        "custom_llm_provider": "anthropic",
        "team_id": request.team_id,
        "request_tags": json.dumps([TASK_INSIGHTS_ANALYSIS_TAG] if request.analysis else ["task-insights-sample"]),
        "status": status,
        **(
            {"request_duration_ms": request.latency_ms} if request.latency_ms is not None and not failed_attempt else {}
        ),
        "metadata": {"status": status},
    }


def _spend_log(request: DemoRequest) -> Mapping[str, object]:
    end: Final = request.start + timedelta(milliseconds=request.latency_ms or 0)
    return {
        "request_id": request.request_id,
        "call_type": "anthropic_messages",
        "api_key": f"{DEMO_PREFIX}key",
        "spend": request.spend,
        "total_tokens": 2150 + 400 * request.turn,
        "prompt_tokens": 1800 + 400 * request.turn,
        "completion_tokens": 350,
        "startTime": _iso(request.start),
        "endTime": _iso(end),
        "request_duration_ms": request.latency_ms,
        "model": request.model.provider_model.split("/", 1)[1],
        "model_id": f"{DEMO_PREFIX}{request.model.key}",
        "model_group": request.model.model_name,
        "custom_llm_provider": "anthropic",
        "team_id": request.team_id,
        "session_id": request.session_id,
        "status": request.status,
        "metadata": {"status": request.status, "task_insights_sample": True},
        "request_tags": [TASK_INSIGHTS_ANALYSIS_TAG] if request.analysis else ["task-insights-sample"],
        "proxy_server_request": (
            {"model": request.model.model_name, "messages": list(_messages(request))} if request.stored_prompt else {}
        ),
    }


def _deliveries(request: DemoRequest) -> tuple[Mapping[str, object], ...]:
    final: Final = {
        "delivery_id": request.request_id,
        "request_id": request.request_id,
        "accounting_payload": _payload(request, request.model, failed_attempt=False),
        "created_at": _iso(request.start),
    }
    if request.retried_on is None:
        return (final,)
    return (
        {
            "delivery_id": f"{request.request_id}-attempt-1",
            "request_id": request.request_id,
            "accounting_payload": _payload(request, request.retried_on, failed_attempt=True),
            "created_at": _iso(request.start),
        },
        final,
    )


def _task_id(task: DemoTask) -> str:
    return f"{DEMO_PREFIX}{task.key}"


def demo_taxonomy(requests: tuple[DemoRequest, ...]) -> tuple[DiscoveredTask, ...]:
    return tuple(
        DiscoveredTask(
            task_id=_task_id(task),
            name=task.name,
            description=task.description,
            inclusion_criteria=task.inclusion,
            exclusion_criteria=task.exclusion,
            examples=task.instructions[:3],
            required_capabilities=task.capabilities,
            map_x=task.center[0],
            map_y=task.center[1],
            discovery_size=sum(1 for request in requests if request.task is task and request.turn == 0),
        )
        for task in DEMO_TASKS
    )


def demo_placements(requests: tuple[DemoRequest, ...]) -> tuple[MapPlacement, ...]:
    """Session openers stand in for the discovery sample, scattered around their task the way an embedding layout
    would be. They are sample positions, not a projection of real embeddings."""
    return tuple(
        MapPlacement(
            request_id=request.request_id,
            task_id=_task_id(request.task),
            x=request.task.center[0]
            + 7
            * math.cos(2 * math.pi * unit(request.request_id, "angle"))
            * math.sqrt(unit(request.request_id, "radius")),
            y=request.task.center[1]
            + 7
            * math.sin(2 * math.pi * unit(request.request_id, "angle"))
            * math.sqrt(unit(request.request_id, "radius")),
        )
        for request in requests
        if request.task is not None and request.turn == 0
    )


def demo_decision(request: DemoRequest) -> Decision:
    task: Final = request.task
    if task is None:
        return Decision(
            outcome="unknown",
            task_id=None,
            probabilities={_task_id(item): round(1 / len(DEMO_TASKS), 4) for item in DEMO_TASKS},
            top_probability=round(1 / len(DEMO_TASKS), 4),
            fit_probability=round(0.05 + 0.1 * unit(request.request_id, "fit"), 4),
            decision_model=_SAMPLE_MODEL,
            provenance=_SAMPLE_PROVENANCE,
        )
    top: Final = round(0.62 + 0.35 * unit(request.request_id, "confidence"), 4)
    rest: Final = round((1 - top) / (len(DEMO_TASKS) - 1), 4)
    return Decision(
        outcome="assigned",
        task_id=_task_id(task),
        probabilities={_task_id(item): top if item is task else rest for item in DEMO_TASKS},
        top_probability=top,
        fit_probability=round(0.8 + 0.2 * unit(request.request_id, "fit"), 4),
        decision_model=_SAMPLE_MODEL,
        provenance=_SAMPLE_PROVENANCE,
    )


class ForeignTaxonomyError(RuntimeError):
    pass


async def seed_demo(database: TaskInsightsDatabase, now: Callable[[], datetime]) -> int:
    demo: Final = TaskInsightsDemoRepository(database, DEMO_PREFIX)
    if await demo.has_foreign_taxonomy():
        raise ForeignTaxonomyError(
            "A task taxonomy that is not sample data exists, so the sample taxonomy would replace it. Seed a fresh database."
        )
    await demo.reset()
    seeded_at: Final = now()
    requests: Final = demo_requests(seeded_at)
    await demo.insert_traffic(
        teams=tuple({"team_id": team_id, "team_alias": alias} for team_id, alias in DEMO_TEAMS),
        spend_logs=tuple(_spend_log(request) for request in requests),
        deliveries=tuple(chain.from_iterable(map(_deliveries, requests))),
        applied_at=_iso(seeded_at),
    )
    repository: Final = TaskInsightsRepository(database)
    await observe_applied_deliveries(
        repository, now=now, batch_size=200, max_batches=100, request_id_prefix=DEMO_PREFIX
    )
    await repository.save_draft_taxonomy(
        demo_taxonomy(requests), demo_placements(requests), _SAMPLE_PROVENANCE, seeded_at, source="sample"
    )
    await repository.freeze_draft(seeded_at)
    by_id: Final = {request.request_id: request for request in requests}
    pending: Final = tuple(
        item for item in await repository.pending_requests(len(requests) + 1) if item.request_id in by_id
    )
    return await repository.save_decisions(
        tuple((item, demo_decision(by_id[item.request_id])) for item in pending),
        seeded_at,
        await repository.taxonomy_version(),
    )


async def _main(reset_only: bool) -> None:
    from litellm.task_insights.database import connected_database

    async with connected_database() as database:
        if reset_only:
            await TaskInsightsDemoRepository(database, DEMO_PREFIX).reset()
            print("Task Insights sample data removed")  # noqa: T201  # CLI output
            return
        assigned: Final = await seed_demo(database, lambda: datetime.now(timezone.utc))
        print(f"Task Insights sample data seeded: {assigned} sample assignments, no model calls")  # noqa: T201  # CLI output


def main() -> None:
    parser: Final = argparse.ArgumentParser(description="Seed or reset the Task Insights sample data")
    parser.add_argument("--reset", action="store_true", help="Remove the sample data and exit")
    asyncio.run(_main(bool(parser.parse_args().reset)))


if __name__ == "__main__":
    main()
