"""Task discovery, after LLooM's concept induction (Lam et al., CHI 2024, https://arxiv.org/abs/2404.12259): cluster
request texts, have an LLM synthesize a named definition per cluster, then score every request against the reviewed
definitions. As in LLooM, each request is first distilled into a one-line intent, so code and repository vocabulary
do not dominate the clusters."""

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Final, Protocol

from pydantic import ConfigDict, Field

from litellm._logging import verbose_proxy_logger
from litellm.models.task_insights import (
    DiscoveredTask,
    DiscoveryRun,
    MapPlacement,
    PendingRequest,
    TaskAnalysisStore,
    TaskCapability,
)
from litellm.types.llms.base import LiteLLMBaseModel

MIN_DISCOVERY_REQUESTS: Final = 20
MAX_TASKS: Final = 12
_EXAMPLES_PER_CLUSTER: Final = 12
_KEPT_EXAMPLES: Final = 3
_MAP_SPAN: Final = 100.0


@dataclass(frozen=True, slots=True)
class ClusterLayout:
    labels: tuple[int, ...]
    points: tuple[tuple[float, float], ...]
    provenance: Mapping[str, str | int | float | None]


class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...

    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]: ...


class Clusterer(Protocol):
    def cluster(self, vectors: tuple[tuple[float, ...], ...]) -> ClusterLayout: ...


class SynthesizedTask(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, max_length=120)
    description: str = Field(max_length=2000)
    inclusion_criteria: str = Field(max_length=2000)
    exclusion_criteria: str = Field(max_length=2000)
    required_capabilities: tuple[TaskCapability, ...] = ()
    example_indices: tuple[int, ...] = ()


class Synthesizer(Protocol):
    @property
    def model_name(self) -> str: ...

    async def distill(self, texts: tuple[str, ...]) -> tuple[str, ...]: ...

    async def synthesize(self, examples: tuple[str, ...]) -> SynthesizedTask: ...


def embedding_text(request: PendingRequest) -> str:
    if not request.context_text:
        return request.request_text
    return f"{request.request_text}\n\nSession context: {request.context_text}"


def _normalized(points: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
    xs: Final = tuple(x for x, _ in points)
    ys: Final = tuple(y for _, y in points)
    span_x: Final = (max(xs) - min(xs)) or 1.0
    span_y: Final = (max(ys) - min(ys)) or 1.0
    return tuple(
        ((x - min(xs)) / span_x * _MAP_SPAN * 0.9 + 5.0, (y - min(ys)) / span_y * _MAP_SPAN * 0.9 + 5.0)
        for x, y in points
    )


def _task_id(run: DiscoveryRun, label: int) -> str:
    start, end = run.window
    return "task-" + sha256(f"{start.isoformat()}|{end.isoformat()}|{label}".encode()).hexdigest()[:12]


def _clusters(labels: tuple[int, ...]) -> tuple[tuple[int, tuple[int, ...]], ...]:
    members: Final = {
        label: tuple(index for index, item in enumerate(labels) if item == label)
        for label in sorted(frozenset(labels) - {-1})
    }
    return tuple(sorted(members.items(), key=lambda item: (-len(item[1]), item[0])))


def _examples(synthesized: SynthesizedTask, texts: tuple[str, ...], fallback: tuple[str, ...]) -> tuple[str, ...]:
    chosen: Final = tuple(texts[index] for index in synthesized.example_indices if 0 <= index < len(texts))
    return (chosen or fallback)[:_KEPT_EXAMPLES]


async def _discovered(
    run: DiscoveryRun,
    label: int,
    members: tuple[int, ...],
    samples: tuple[PendingRequest, ...],
    points: tuple[tuple[float, float], ...],
    synthesizer: Synthesizer,
) -> DiscoveredTask:
    texts: Final = tuple(samples[index].request_text[:600] for index in members[:_EXAMPLES_PER_CLUSTER])
    synthesized: Final = await synthesizer.synthesize(texts)
    return DiscoveredTask(
        task_id=_task_id(run, label),
        name=synthesized.name,
        description=synthesized.description,
        inclusion_criteria=synthesized.inclusion_criteria,
        exclusion_criteria=synthesized.exclusion_criteria,
        examples=_examples(synthesized, texts, texts),
        required_capabilities=synthesized.required_capabilities,
        map_x=sum(points[index][0] for index in members) / len(members),
        map_y=sum(points[index][1] for index in members) / len(members),
        discovery_size=len(members),
    )


async def discover(
    store: TaskAnalysisStore,
    embedder: Embedder,
    clusterer: Clusterer,
    synthesizer: Synthesizer,
    *,
    now: Callable[[], datetime],
) -> bool:
    """Run one queued discovery. Every failure is recorded on the state row for the admin to see, and the frozen
    taxonomy is untouched until an admin freezes the new draft."""
    run: Final = await store.claim_discovery()
    if run is None:
        return False
    try:
        await _run(run, store, embedder, clusterer, synthesizer, now)
    except Exception as error:  # noqa: BLE001  # surfaced on the state row instead of crashing the worker
        verbose_proxy_logger.exception("Task Insights discovery failed")
        await store.fail_discovery(f"{type(error).__name__}: {error}", now())
    return True


async def _run(
    run: DiscoveryRun,
    store: TaskAnalysisStore,
    embedder: Embedder,
    clusterer: Clusterer,
    synthesizer: Synthesizer,
    now: Callable[[], datetime],
) -> None:
    samples: Final = await store.discovery_samples(run)
    if len(samples) < MIN_DISCOVERY_REQUESTS:
        await store.fail_discovery(
            f"Discovery needs at least {MIN_DISCOVERY_REQUESTS} requests with stored prompt text in the window, "
            f"found {len(samples)}",
            now(),
        )
        return
    intents: Final = await synthesizer.distill(tuple(embedding_text(sample) for sample in samples))
    if len(intents) != len(samples):
        await store.fail_discovery(f"Distillation returned {len(intents)} intents for {len(samples)} requests", now())
        return
    vectors: Final = await embedder.embed(intents)
    layout: Final = await asyncio.to_thread(clusterer.cluster, vectors)
    points: Final = _normalized(layout.points)
    found: Final = _clusters(layout.labels)
    if not found:
        await store.fail_discovery(
            "Clustering found no groups, so the current draft and frozen taxonomy are left as they are", now()
        )
        return
    clusters: Final = found[:MAX_TASKS]
    dropped: Final = found[MAX_TASKS:]
    tasks: Final = tuple(
        [await _discovered(run, label, members, samples, points, synthesizer) for label, members in clusters]
    )
    task_by_label: Final = {label: _task_id(run, label) for label, _ in clusters}
    placements: Final = tuple(
        MapPlacement(request_id=sample.request_id, task_id=task_by_label.get(label), x=x, y=y)
        for sample, label, (x, y) in zip(samples, layout.labels, points)
    )
    await store.save_draft_taxonomy(
        tasks,
        placements,
        {
            **layout.provenance,
            "sample_size": len(samples),
            "task_count": len(tasks),
            "noise_count": sum(1 for label in layout.labels if label == -1),
            "dropped_cluster_count": len(dropped),
            "dropped_request_count": sum(len(members) for _, members in dropped),
            "embedding_model": embedder.model_name,
            "synthesis_model": synthesizer.model_name,
        },
        now(),
    )
