import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.db.model_usage_rollup import (
    ModelUsageKey,
    ModelUsageTransaction,
    build_model_usage_transaction,
    flush_model_usage_transactions,
    model_usage_task_type,
)


class _FakeBatcher:
    def __init__(self) -> None:
        self.litellm_dailymodelusage = MagicMock()

    async def __aenter__(self) -> "_FakeBatcher":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


def _prisma(batch_: MagicMock) -> MagicMock:
    prisma = MagicMock()
    prisma.db.batch_ = batch_
    return prisma


def _payload(**overrides: Any) -> dict[str, Any]:
    return {
        "spend": 0.25,
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "startTime": datetime(2026, 9, 28, 13, tzinfo=timezone.utc),
        "model": "openai/gpt-5.4-mini",
        "model_group": "fast-chat",
        "metadata": "{}",
        "request_tags": "[]",
        "custom_llm_provider": "openai",
        "status": "success",
        **overrides,
    }


def _transaction(model: str, spend: float, successful: bool = True) -> ModelUsageTransaction:
    return ModelUsageTransaction(
        key=ModelUsageKey(
            date="2026-09-28", model_group=model, model=model, custom_llm_provider="openai", task_type="debugging"
        ),
        spend=spend,
        prompt_tokens=10,
        completion_tokens=5,
        successful=successful,
    )


async def _no_sleep(seconds: float) -> None:
    return None


def test_model_usage_task_type_reads_task_tag_or_defaults() -> None:
    assert model_usage_task_type('["team-a", "task:classification"]') == "classification"
    assert model_usage_task_type('["task:made-up"]') == "uncategorized"
    assert model_usage_task_type('["debugging"]') == "uncategorized"
    assert model_usage_task_type("[]") == "uncategorized"
    assert model_usage_task_type("not json") == "uncategorized"


def test_build_model_usage_transaction_keys_on_day_model_and_task() -> None:
    transaction = build_model_usage_transaction(_payload(request_tags='["task:debugging"]', status="failure"))

    assert transaction == ModelUsageTransaction(
        key=ModelUsageKey(
            date="2026-09-28",
            model_group="fast-chat",
            model="openai/gpt-5.4-mini",
            custom_llm_provider="openai",
            task_type="debugging",
        ),
        spend=0.25,
        prompt_tokens=10,
        completion_tokens=20,
        successful=False,
    )


def test_build_model_usage_transaction_falls_back_for_missing_model_fields() -> None:
    transaction = build_model_usage_transaction(
        _payload(model="", model_group=None, custom_llm_provider=None, startTime="2026-09-28T01:02:03Z")
    )

    assert transaction is not None
    assert transaction.key == ModelUsageKey(
        date="2026-09-28",
        model_group="unknown",
        model="unknown",
        custom_llm_provider="unknown",
        task_type="uncategorized",
    )


@pytest.mark.parametrize(
    "overrides",
    [{"metadata": '{"internal_call_origin": "health_check"}'}, {"startTime": "bad"}],
)
def test_build_model_usage_transaction_skips_internal_calls_and_bad_dates(overrides: dict[str, Any]) -> None:
    assert build_model_usage_transaction(_payload(**overrides)) is None


@pytest.mark.asyncio
async def test_flush_aggregates_each_rollup_row_into_one_upsert() -> None:
    batcher = _FakeBatcher()
    prisma = _prisma(MagicMock(return_value=batcher))

    await flush_model_usage_transactions(
        prisma_client=prisma,
        transactions=[
            _transaction("gpt-5", 0.5),
            _transaction("claude", 1.0),
            _transaction("gpt-5", 0.25, successful=False),
            _transaction("gpt-5", 0.25),
        ],
    )

    upserts = {
        call.kwargs["where"]["date_model_group_model_custom_llm_provider_task_type"]["model"]: call.kwargs["data"]
        for call in batcher.litellm_dailymodelusage.upsert.call_args_list
    }
    assert list(upserts) == ["claude", "gpt-5"]
    gpt = upserts["gpt-5"]
    assert gpt["create"]["spend"] == 1.0
    assert gpt["create"]["prompt_tokens"] == 30
    assert gpt["create"]["completion_tokens"] == 15
    assert gpt["create"]["request_count"] == 3
    assert gpt["create"]["successful_requests"] == 2
    assert gpt["create"]["failed_requests"] == 1
    assert gpt["update"] == {
        "spend": {"increment": 1.0},
        "prompt_tokens": {"increment": 30},
        "completion_tokens": {"increment": 15},
        "request_count": {"increment": 3},
        "successful_requests": {"increment": 2},
        "failed_requests": {"increment": 1},
    }
    assert upserts["claude"]["create"]["request_count"] == 1


@pytest.mark.asyncio
async def test_flush_with_no_transactions_touches_nothing() -> None:
    prisma = _prisma(MagicMock())
    await flush_model_usage_transactions(prisma_client=prisma, transactions=[])
    prisma.db.batch_.assert_not_called()


@pytest.mark.asyncio
async def test_flush_retries_connection_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    batcher = _FakeBatcher()
    prisma = _prisma(MagicMock(side_effect=[httpx.ConnectError("down"), batcher]))
    monkeypatch.setattr("litellm.proxy.db.model_usage_rollup.asyncio.sleep", _no_sleep)

    await flush_model_usage_transactions(prisma_client=prisma, transactions=[_transaction("gpt-5", 0.1)])

    assert prisma.db.batch_.call_count == 2
    batcher.litellm_dailymodelusage.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_flush_does_not_retry_ambiguous_errors() -> None:
    prisma = _prisma(MagicMock(side_effect=httpx.ReadTimeout("ambiguous")))

    with pytest.raises(httpx.ReadTimeout):
        await flush_model_usage_transactions(prisma_client=prisma, transactions=[_transaction("gpt-5", 0.1)])

    prisma.db.batch_.assert_called_once()


@pytest.mark.asyncio
async def test_request_time_path_queues_usage_instead_of_writing_to_the_db(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.db import model_insights_task_classifier as classifier

    monkeypatch.setattr(classifier._HOLDER, "batcher", None)
    prisma = MagicMock()
    prisma.model_usage_transactions = []
    prisma._model_usage_transactions_lock = asyncio.Lock()

    await DBSpendUpdateWriter()._batch_database_updates(
        response_cost=0.25,
        user_id="u1",
        hashed_token="t1",
        team_id=None,
        org_id=None,
        end_user_id=None,
        prisma_client=prisma,
        litellm_proxy_budget_name=None,
        payload=_payload(request_id="req-1"),
    )

    assert [transaction.key.model for transaction in prisma.model_usage_transactions] == ["openai/gpt-5.4-mini"]
    prisma.db.litellm_dailymodelusage.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_untagged_usage_is_classified_in_the_background_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock

    from litellm.proxy.db import model_insights_task_classifier as classifier
    from litellm.router_strategy.complexity_router.jev_classifier import JevChoiceAnswer, JevSystemOneResponse

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        async def evaluate(self, request: object, timeout_s: float, request_kwargs: object = None) -> object:
            self.calls += 1
            return JevSystemOneResponse(
                answers={"task": JevChoiceAnswer(type="choice", choice="debugging", probabilities={}, confidence=1)}
            )

    class Router:
        def get_model_list(self) -> list[dict[str, object]]:
            return [
                {
                    "model_name": "tasks",
                    "litellm_params": {"model": "typesafe/jev-latest", "api_key": "sk-test"},
                    "model_info": {"id": "jev"},
                }
            ]

    client = Client()
    prisma = MagicMock()
    prisma.model_usage_transactions = []
    prisma._model_usage_transactions_lock = asyncio.Lock()
    prisma.db.litellm_config.find_unique = AsyncMock(
        return_value=MagicMock(param_value='{"enabled": true, "model_id": "jev"}')
    )
    batcher = classifier.TaskClassifierBatcher(
        classifier.TaskClassifierStore(prisma, Router, lambda deployment: client, lambda: 0.0)
    )
    monkeypatch.setattr(classifier._HOLDER, "batcher", batcher)
    flushed: list[ModelUsageTransaction] = []

    async def capture(prisma_client: object, transactions: list[ModelUsageTransaction]) -> None:
        flushed.extend(transactions)

    monkeypatch.setattr("litellm.proxy.db.model_usage_rollup.flush_model_usage_transactions", capture)
    payload = _payload(request_id="req-1")
    payload["proxy_server_request"] = '{"messages": [{"role": "user", "content": "Fix this traceback"}]}'

    await DBSpendUpdateWriter()._enqueue_model_usage_transaction(payload=payload, prisma_client=prisma)

    assert prisma.model_usage_transactions == []
    assert client.calls == 0
    assert await batcher.pending_count() == 1

    from litellm.proxy import utils

    async def no_op(*args: object, **kwargs: object) -> None:
        return None

    async def no_logs(*args: object, **kwargs: object) -> list[dict[str, object]]:
        return []

    monkeypatch.setattr(utils, "dequeue_spend_logs", no_logs)
    monkeypatch.setattr(utils.ProxyUpdateSpend, "update_spend_logs", no_op)
    monkeypatch.setattr("litellm.proxy.guardrails.usage_tracking.process_spend_logs_guardrail_usage", no_op)
    monkeypatch.setattr("litellm.proxy.db.spend_log_tool_index.flush_tool_usage_transactions", no_op)
    monkeypatch.setattr("litellm.proxy.db.baseline_accounting.flush_baseline_accounting", no_op)
    prisma.tool_usage_transactions = []
    prisma._tool_usage_transactions_lock = asyncio.Lock()
    prisma.autorouter_turn_transactions = []
    prisma._autorouter_turn_transactions_lock = asyncio.Lock()

    await utils._run_spend_logs_job(prisma, None, MagicMock())

    assert [(item.key.model, item.key.task_type) for item in flushed] == [("openai/gpt-5.4-mini", "debugging")]
    assert client.calls == 1
    assert await batcher.pending_count() == 0
