from datetime import datetime, timezone

import pytest

from litellm.models.task_insights import DiscoveryRun, PendingRequest
from litellm.task_insights.discovery import (
    MAX_TASKS,
    MIN_DISCOVERY_REQUESTS,
    ClusterLayout,
    SynthesizedTask,
    discover,
    embedding_text,
)

_AT = datetime(2026, 11, 11, tzinfo=timezone.utc)
_RUN = DiscoveryRun(window=(datetime(2026, 11, 1, tzinfo=timezone.utc), _AT), max_requests=500)


def _samples(count: int) -> tuple[PendingRequest, ...]:
    return tuple(
        PendingRequest(
            request_id=f"r{index}", text_hash=f"h{index}", request_text=f"request {index}", context_text=None
        )
        for index in range(count)
    )


class _Store:
    def __init__(self, samples, run=_RUN) -> None:
        self.samples = samples
        self.run = run
        self.saved = None
        self.failed: str | None = None

    async def claim_discovery(self):
        return self.run

    async def discovery_samples(self, run):
        return self.samples

    async def save_draft_taxonomy(self, tasks, placements, provenance, finished_at, source="discovered"):
        self.saved = (tasks, placements, dict(provenance))

    async def fail_discovery(self, error, finished_at):
        self.failed = error


class _Embedder:
    model_name = "qwen-embed"

    def __init__(self) -> None:
        self.embedded: tuple[str, ...] = ()

    async def embed(self, texts):
        self.embedded = texts
        return tuple((float(index), 0.0) for index, _ in enumerate(texts))


class _Clusterer:
    def __init__(self, labels) -> None:
        self.labels = labels

    def cluster(self, vectors):
        return ClusterLayout(
            labels=self.labels,
            points=tuple((float(index), float(index % 3)) for index in range(len(vectors))),
            provenance={"clusterer": "fake"},
        )


class _Synthesizer:
    model_name = "claude"

    def __init__(self, intents_short_by: int = 0) -> None:
        self.intents_short_by = intents_short_by
        self.synthesized = 0

    async def distill(self, texts):
        return tuple(f"intent of {text}" for text in texts[: len(texts) - self.intents_short_by])

    async def synthesize(self, examples):
        self.synthesized += 1
        return SynthesizedTask(
            name=f"Task {self.synthesized}",
            description="desc",
            inclusion_criteria="in",
            exclusion_criteria="out",
            example_indices=(1, 99),
        )


async def _discover(store, labels, synthesizer=None):
    embedder = _Embedder()
    ran = await discover(store, embedder, _Clusterer(labels), synthesizer or _Synthesizer(), now=lambda: _AT)
    return ran, embedder


@pytest.mark.asyncio
async def test_drafts_one_task_per_cluster_from_distilled_intents_and_keeps_noise_unassigned():
    samples = _samples(MIN_DISCOVERY_REQUESTS)
    labels = tuple(0 if index < 12 else (1 if index < 18 else -1) for index in range(len(samples)))
    store = _Store(samples)

    ran, embedder = await _discover(store, labels)

    assert ran is True
    assert embedder.embedded[0] == "intent of request 0"
    tasks, placements, provenance = store.saved
    assert [(task.name, task.discovery_size) for task in tasks] == [("Task 1", 12), ("Task 2", 6)]
    assert tasks[0].examples == ("request 1",)
    assert sum(1 for point in placements if point.task_id is None) == 2
    assert {point.task_id for point in placements} - {None} == {task.task_id for task in tasks}
    assert all(5 <= point.x <= 95 and 5 <= point.y <= 95 for point in placements)
    assert (provenance["noise_count"], provenance["dropped_cluster_count"], provenance["embedding_model"]) == (
        2,
        0,
        "qwen-embed",
    )


@pytest.mark.asyncio
async def test_reports_the_clusters_past_the_task_limit_instead_of_hiding_them():
    samples = _samples(MIN_DISCOVERY_REQUESTS * 2)
    labels = tuple(index % (MAX_TASKS + 2) for index in range(len(samples)))
    store = _Store(samples)

    await _discover(store, labels)

    tasks, placements, provenance = store.saved
    assert len(tasks) == MAX_TASKS
    assert provenance["dropped_cluster_count"] == 2
    assert provenance["dropped_request_count"] == sum(1 for point in placements if point.task_id is None)


@pytest.mark.asyncio
async def test_too_few_requests_fail_without_calling_models():
    store = _Store(_samples(MIN_DISCOVERY_REQUESTS - 1))
    synthesizer = _Synthesizer()

    _, embedder = await _discover(store, (), synthesizer)

    assert store.failed and "at least" in store.failed
    assert (store.saved, embedder.embedded, synthesizer.synthesized) == (None, (), 0)


@pytest.mark.asyncio
async def test_no_clusters_leaves_the_existing_taxonomy_alone():
    store = _Store(_samples(MIN_DISCOVERY_REQUESTS))

    await _discover(store, (-1,) * MIN_DISCOVERY_REQUESTS)

    assert store.saved is None
    assert store.failed and "no groups" in store.failed


@pytest.mark.asyncio
async def test_a_distillation_that_drops_requests_fails_the_run():
    store = _Store(_samples(MIN_DISCOVERY_REQUESTS))

    await _discover(store, (0,) * MIN_DISCOVERY_REQUESTS, _Synthesizer(intents_short_by=1))

    assert store.saved is None and store.failed and "Distillation" in store.failed


@pytest.mark.asyncio
async def test_an_unexpected_error_is_recorded_on_the_state():
    class _Broken(_Synthesizer):
        async def distill(self, texts):
            raise ConnectionError("gateway down")

    store = _Store(_samples(MIN_DISCOVERY_REQUESTS))

    ran, _ = await _discover(store, (), _Broken())

    assert ran is True
    assert store.failed == "ConnectionError: gateway down"
    assert store.saved is None


@pytest.mark.asyncio
async def test_nothing_runs_unless_discovery_was_queued():
    store = _Store(_samples(MIN_DISCOVERY_REQUESTS), run=None)

    ran, embedder = await _discover(store, ())

    assert (ran, embedder.embedded, store.failed) == (False, (), None)


def test_embedding_text_adds_session_context_only_when_present():
    request = PendingRequest(request_id="r", text_hash="h", request_text="Fix it", context_text=None)

    assert embedding_text(request) == "Fix it"
    assert embedding_text(request.model_copy(update={"context_text": "CI"})) == "Fix it\n\nSession context: CI"
