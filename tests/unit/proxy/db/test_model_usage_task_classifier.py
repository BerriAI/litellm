import asyncio
from collections.abc import Mapping, Sequence
from itertools import chain
from types import MappingProxyType, SimpleNamespace
from typing import Final, Literal
from unittest.mock import AsyncMock

import pytest

from litellm.proxy.db.model_usage_rollup import ModelUsageKey, ModelUsageTransaction
from litellm.proxy.db.model_insights_tasks import load_model_insight_tasks
from litellm.proxy.db.model_usage_task_classifier import (
    ModelUsageTaskClassifier,
    classify_pending_model_usage,
)
from litellm.router_strategy.complexity_router.jev_classifier import (
    JevChoiceAnswer,
    JevClassifierClient,
    JevSystemOneRequest,
    JevSystemOneResponse,
)
from litellm.types.model_insights import ModelInsightTaskClassifierConfig

_EXPECTED_ANSWER_LABELS: Final = (
    "debugging",
    "translation",
    "summarization",
    "code_generation",
    "classification",
    "content_writing",
    "customer_support",
    "data_extraction",
)


class _FakePrismaClient:
    def __init__(self, config: Mapping[str, str] | None = None) -> None:
        self.model_usage_transactions: list[ModelUsageTransaction] = []
        self._model_usage_transactions_lock: Final = asyncio.Lock()
        config_row: Final = SimpleNamespace(param_value=dict(config)) if config is not None else None
        self.db: Final = SimpleNamespace(
            litellm_config=SimpleNamespace(find_unique=AsyncMock(return_value=config_row))
        )

    async def append_model_usage_transactions(
        self,
        transactions: Sequence[ModelUsageTransaction],
    ) -> None:
        async with self._model_usage_transactions_lock:
            self.model_usage_transactions.extend(transactions)


class _RecordingClassifierClient:
    def __init__(
        self,
        mode: Literal["labels", "unknown_and_missing", "raise"] = "labels",
    ) -> None:
        self.mode: Final = mode
        self.requests: list[JevSystemOneRequest] = []

    async def evaluate(
        self,
        request: JevSystemOneRequest,
        timeout_s: float,
        request_kwargs: Mapping[str, object] | None = None,
    ) -> JevSystemOneResponse:
        self.requests.append(request)
        if self.mode == "raise":
            raise RuntimeError("classifier unavailable")
        if self.mode == "unknown_and_missing":
            return JevSystemOneResponse(
                answers=MappingProxyType(
                    {
                        "r1": JevChoiceAnswer(
                            type="choice",
                            choice="not-a-catalog-task",
                            probabilities=MappingProxyType({}),
                            confidence=1.0,
                        )
                    }
                )
            )
        answers: Final = MappingProxyType(
            {
                key: JevChoiceAnswer(
                    type="choice",
                    choice=_EXPECTED_ANSWER_LABELS[index],
                    probabilities=MappingProxyType({}),
                    confidence=1.0,
                )
                for index, key in enumerate(request.questions)
            }
        )
        return JevSystemOneResponse(answers=answers)


def _transaction(index: int) -> ModelUsageTransaction:
    return ModelUsageTransaction(
        key=ModelUsageKey(
            date="2026-09-28",
            model_group="test-model",
            model="test-model",
            custom_llm_provider="test-provider",
            task_type="uncategorized",
        ),
        spend=float(index + 1),
        prompt_tokens=index + 10,
        completion_tokens=index + 20,
        successful=index % 2 == 0,
        prompt=f"request {index}",
    )


async def _configured_classifier(
    client: JevClassifierClient,
    queue_capacity: int = 10_000,
) -> ModelUsageTaskClassifier:
    classifier: Final = ModelUsageTaskClassifier(queue_capacity=queue_capacity)
    await classifier.activate(ModelInsightTaskClassifierConfig(provider="jev", model="jev-latest"), client)
    return classifier


@pytest.mark.asyncio
async def test_twenty_prompts_are_classified_in_three_eight_prompt_batches() -> None:
    prisma: Final = _FakePrismaClient()
    client: Final = _RecordingClassifierClient()
    classifier: Final = await _configured_classifier(client)
    for index in range(20):
        await classifier.enqueue(prisma, _transaction(index))

    classified_count: Final = await classify_pending_model_usage(prisma, classifier, client)

    assert classified_count == 20
    assert sorted(len(request.questions) for request in client.requests) == [4, 8, 8]
    assert [
        (transaction.spend, transaction.key.task_type)
        for transaction in prisma.model_usage_transactions
    ] == [
        (float(index + 1), _EXPECTED_ANSWER_LABELS[index % len(_EXPECTED_ANSWER_LABELS)])
        for index in range(20)
    ]
    assert all(transaction.prompt is None for transaction in prisma.model_usage_transactions)
    expected_criteria: Final = {
        task_type: task.description for task_type, task in load_model_insight_tasks().items()
    }
    assert all(
        question.criteria == expected_criteria
        for question in chain.from_iterable(request.questions.values() for request in client.requests)
    )


@pytest.mark.asyncio
async def test_unknown_and_missing_answers_fall_back_to_uncategorized() -> None:
    prisma: Final = _FakePrismaClient()
    client: Final = _RecordingClassifierClient(mode="unknown_and_missing")
    classifier: Final = await _configured_classifier(client)
    await classifier.enqueue(prisma, _transaction(0))
    await classifier.enqueue(prisma, _transaction(1))

    await classify_pending_model_usage(prisma, classifier, client)

    assert [transaction.key.task_type for transaction in prisma.model_usage_transactions] == [
        "uncategorized",
        "uncategorized",
    ]
    assert [transaction.spend for transaction in prisma.model_usage_transactions] == [1.0, 2.0]
    assert all(transaction.prompt is None for transaction in prisma.model_usage_transactions)


@pytest.mark.asyncio
async def test_client_failure_forwards_the_chunk_uncategorized_with_counts_intact() -> None:
    prisma: Final = _FakePrismaClient()
    client: Final = _RecordingClassifierClient(mode="raise")
    classifier: Final = await _configured_classifier(client)
    source: Final = tuple(_transaction(index) for index in range(3))
    for transaction in source:
        await classifier.enqueue(prisma, transaction)

    await classify_pending_model_usage(prisma, classifier, client)

    assert [
        (
            transaction.spend,
            transaction.prompt_tokens,
            transaction.completion_tokens,
            transaction.successful,
            transaction.key.task_type,
            transaction.prompt,
        )
        for transaction in prisma.model_usage_transactions
    ] == [
        (item.spend, item.prompt_tokens, item.completion_tokens, item.successful, "uncategorized", None)
        for item in source
    ]


@pytest.mark.asyncio
async def test_unconfigured_classifier_enqueues_uncategorized_without_a_prompt() -> None:
    prisma: Final = _FakePrismaClient()
    classifier: Final = ModelUsageTaskClassifier()

    await classifier.enqueue(prisma, _transaction(0))

    assert await classifier.pending_count() == 0
    assert [
        (transaction.key.task_type, transaction.prompt)
        for transaction in prisma.model_usage_transactions
    ] == [("uncategorized", None)]


@pytest.mark.asyncio
async def test_pending_queue_overflow_enqueues_the_extra_transaction_uncategorized() -> None:
    prisma: Final = _FakePrismaClient()
    client: Final = _RecordingClassifierClient()
    classifier: Final = await _configured_classifier(client, queue_capacity=1)

    await classifier.enqueue(prisma, _transaction(0))
    await classifier.enqueue(prisma, _transaction(1))

    assert await classifier.pending_count() == 1
    assert [
        (transaction.spend, transaction.key.task_type, transaction.prompt)
        for transaction in prisma.model_usage_transactions
    ] == [(2.0, "uncategorized", None)]


@pytest.mark.asyncio
async def test_classification_job_refreshes_provider_config_from_the_database() -> None:
    prisma: Final = _FakePrismaClient({"provider": "jev", "model": "jev-latest"})
    client: Final = _RecordingClassifierClient()
    classifier: Final = ModelUsageTaskClassifier(client_builder=lambda _: client)

    await classifier.refresh_from_db(prisma)
    await classifier.enqueue(prisma, _transaction(0))
    await classify_pending_model_usage(prisma, classifier)

    assert prisma.db.litellm_config.find_unique.await_count == 2
    assert [
        (transaction.spend, transaction.key.task_type, transaction.prompt)
        for transaction in prisma.model_usage_transactions
    ] == [(1.0, "debugging", None)]


@pytest.mark.asyncio
async def test_stale_database_refresh_does_not_clear_a_newer_activation() -> None:
    prisma: Final = _FakePrismaClient()
    read_started: Final = asyncio.Event()
    release_read: Final = asyncio.Event()

    async def delayed_find_unique(*, where: Mapping[str, str]) -> None:
        assert where == {"param_name": "model_insights_task_classifier"}
        read_started.set()
        await release_read.wait()
        return None

    prisma.db.litellm_config.find_unique.side_effect = delayed_find_unique
    client: Final = _RecordingClassifierClient()
    classifier: Final = ModelUsageTaskClassifier(client_builder=lambda _: client)
    refresh_task: Final = asyncio.create_task(classifier.refresh_from_db(prisma))

    await read_started.wait()
    await classifier.activate(ModelInsightTaskClassifierConfig(provider="jev", model="jev-latest"), client)
    release_read.set()
    await refresh_task
    await classifier.enqueue(prisma, _transaction(0))

    assert await classifier.pending_count() == 1
    assert prisma.model_usage_transactions == []
