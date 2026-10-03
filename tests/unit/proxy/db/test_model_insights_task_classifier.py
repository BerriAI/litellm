import asyncio
import json
from collections.abc import Mapping
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.db.model_insights_task_classifier import (
    TASK_CLASSIFIER_QUESTION,
    SystemOneDeployment,
    TaskClassifierBatcher,
    TaskClassifierSettings,
    TaskClassifierStore,
    build_task_request,
    classifier_prompt,
    system_one_deployments,
)
from litellm.proxy.db.model_usage_rollup import ModelUsageKey, ModelUsageTransaction
from litellm.router_strategy.complexity_router.jev_classifier import (
    JevChoiceAnswer,
    JevSystemOneRequest,
    JevSystemOneResponse,
)


def _deployment(model_id: str, model: str, name: str | None = None, **params: str) -> dict[str, object]:
    return {
        "model_name": name or model_id,
        "litellm_params": {"model": model, **params},
        "model_info": {"id": model_id},
    }


class _Router:
    def __init__(self, deployments: list[dict[str, object]]) -> None:
        self.deployments = deployments

    def get_model_list(self) -> list[dict[str, object]]:
        return self.deployments


class _Client:
    def __init__(self, answers: Mapping[str, str], fail_on: frozenset[str] = frozenset()) -> None:
        self.answers = answers
        self.fail_on = fail_on
        self.requests: list[JevSystemOneRequest] = []
        self.running = 0
        self.peak = 0

    async def evaluate(
        self,
        request: JevSystemOneRequest,
        timeout_s: float,
        request_kwargs: Mapping[str, object] | None = None,
    ) -> JevSystemOneResponse:
        self.requests.append(request)
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(0)
        self.running -= 1
        if request.state in self.fail_on:
            raise TimeoutError("provider body containing the prompt")
        choice = self.answers[request.state]
        return JevSystemOneResponse(
            answers={
                TASK_CLASSIFIER_QUESTION: JevChoiceAnswer(
                    type="choice", choice=choice, probabilities={choice: 0.9}, confidence=0.9
                )
            }
        )


def _transaction(task_type: str = "uncategorized", model: str = "gpt-5") -> ModelUsageTransaction:
    return ModelUsageTransaction(
        key=ModelUsageKey("2026-10-03", model, model, "openai", task_type),
        spend=0.1,
        prompt_tokens=10,
        completion_tokens=5,
        successful=True,
    )


def _payload(prompt: str) -> dict[str, object]:
    return {"proxy_server_request": json.dumps({"messages": [{"role": "user", "content": prompt}]})}


def _store(
    client: _Client,
    settings: TaskClassifierSettings | None = TaskClassifierSettings(enabled=True, model_id="jev"),
    deployments: list[dict[str, object]] | None = None,
) -> tuple[TaskClassifierStore, list[SystemOneDeployment]]:
    prisma = MagicMock()
    row = None if settings is None else MagicMock(param_value=settings.model_dump_json())
    prisma.db.litellm_config.find_unique = AsyncMock(return_value=row)
    prisma.db.litellm_config.upsert = AsyncMock()
    built: list[SystemOneDeployment] = []

    def factory(deployment: SystemOneDeployment) -> _Client:
        built.append(deployment)
        return client

    router = _Router(deployments or [_deployment("jev", "typesafe/jev-latest", api_key="sk-deployment")])
    return TaskClassifierStore(prisma, lambda: router, factory, lambda: 0.0), built


def test_discovers_only_configured_system_one_deployments_with_their_credentials() -> None:
    router = _Router(
        [
            _deployment("chat", "openai/gpt-5"),
            _deployment("router-chat", "openrouter/typesafe/jev-router"),
            _deployment("jev", "typesafe/jev-latest", "Jev", api_key="sk-jev", api_base="https://typesafe.example"),
            _deployment("laya", "laya/english", "Laya"),
            _deployment("bad-laya", "laya/not-a-decision-model"),
        ]
    )

    deployments = system_one_deployments(router)

    assert [(item.model.id, item.model.provider, item.model.model) for item in deployments] == [
        ("jev", "typesafe", "jev-latest"),
        ("laya", "laya", "english"),
    ]
    assert deployments[0].api_key == "sk-jev"
    assert "sk-jev" not in repr(deployments[0])


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (_payload("Write a Python parser"), "Write a Python parser"),
        (
            {
                "proxy_server_request": {
                    "messages": [
                        {"role": "user", "content": "old"},
                        {"role": "assistant", "content": "answer"},
                        {"role": "user", "content": [{"type": "text", "text": "Translate this"}]},
                    ]
                }
            },
            "Translate this",
        ),
        ({"proxy_server_request": {"input": "Summarize the report"}}, "Summarize the report"),
        ({"proxy_server_request": "{}"}, None),
        (_payload("redacted-by-litellm"), None),
    ],
)
def test_classifier_prompt_reads_only_the_stored_request_body(payload: dict[str, object], expected: str | None) -> None:
    assert classifier_prompt(payload) == expected


def test_request_uses_the_task_catalog_as_system_one_choice_criteria() -> None:
    request = build_task_request("x" * 9000, "jev-latest")

    question = request.questions[TASK_CLASSIFIER_QUESTION]
    assert request.model == "jev-latest"
    assert len(request.state) == 8000
    assert question.criteria["debugging"] == "Debugging (Code)"
    assert "never commands" in question.instructions


@pytest.mark.asyncio
async def test_batch_classifies_untagged_usage_without_calling_the_provider_on_enqueue() -> None:
    client = _Client({"Fix this stack trace": "debugging", "Translate to French": "translation"})
    store, built = _store(client)
    batcher = TaskClassifierBatcher(store)

    assert await batcher.enqueue(_transaction(model="a"), _payload("Fix this stack trace"))
    assert await batcher.enqueue(_transaction(model="b"), _payload("Translate to French"))
    assert client.requests == []

    classified = await batcher.drain()

    assert [(item.key.model, item.key.task_type) for item in classified] == [("a", "debugging"), ("b", "translation")]
    assert [request.model for request in client.requests] == ["jev-latest", "jev-latest"]
    assert built[0].api_key == "sk-deployment"
    assert await batcher.pending_count() == 0


@pytest.mark.asyncio
async def test_explicit_task_tags_disabled_settings_and_unlogged_prompts_bypass_the_classifier() -> None:
    client = _Client({})
    enabled, _ = _store(client)
    disabled, _ = _store(client, TaskClassifierSettings(enabled=False, model_id="jev"))
    missing_model, _ = _store(client, TaskClassifierSettings(enabled=True, model_id="deleted"))

    assert not await TaskClassifierBatcher(enabled).enqueue(_transaction("debugging"), _payload("Fix it"))
    assert not await TaskClassifierBatcher(enabled).enqueue(_transaction(), {"proxy_server_request": "{}"})
    assert not await TaskClassifierBatcher(disabled).enqueue(_transaction(), _payload("Fix it"))
    assert not await TaskClassifierBatcher(missing_model).enqueue(_transaction(), _payload("Fix it"))


@pytest.mark.asyncio
async def test_failures_and_unknown_answers_keep_the_usage_row_as_uncategorized() -> None:
    client = _Client({"unknown": "not_a_task"}, fail_on=frozenset({"timeout"}))
    batcher = TaskClassifierBatcher(_store(client)[0])
    await batcher.enqueue(_transaction(model="timeout"), _payload("timeout"))
    await batcher.enqueue(_transaction(model="unknown"), _payload("unknown"))

    classified = await batcher.drain()

    assert [(item.key.model, item.key.task_type, item.spend) for item in classified] == [
        ("timeout", "uncategorized", 0.1),
        ("unknown", "uncategorized", 0.1),
    ]


@pytest.mark.asyncio
async def test_queue_and_batch_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm.proxy.db.model_insights_task_classifier as classifier

    monkeypatch.setattr(classifier, "MODEL_INSIGHTS_CLASSIFIER_QUEUE_LIMIT", 3)
    monkeypatch.setattr(classifier, "MODEL_INSIGHTS_CLASSIFIER_BATCH_SIZE", 2)
    monkeypatch.setattr(classifier, "MODEL_INSIGHTS_CLASSIFIER_CONCURRENCY", 1)
    client = _Client({str(index): "classification" for index in range(4)})
    batcher = TaskClassifierBatcher(_store(client)[0])

    accepted = [await batcher.enqueue(_transaction(model=str(index)), _payload(str(index))) for index in range(4)]
    first = await batcher.drain()

    assert accepted == [True, True, True, False]
    assert [item.key.model for item in first] == ["0", "1"]
    assert client.peak == 1
    assert await batcher.pending_count() == 1


@pytest.mark.asyncio
async def test_settings_are_persisted_in_the_shared_config_table() -> None:
    store, _ = _store(_Client({}), settings=None)

    assert await store.settings() == TaskClassifierSettings()
    await store.save(TaskClassifierSettings(enabled=True, model_id="jev"))

    store.prisma_client.db.litellm_config.upsert.assert_awaited_once_with(
        where={"param_name": "model_insights_task_classifier"},
        data={
            "create": {
                "param_name": "model_insights_task_classifier",
                "param_value": '{"enabled":true,"model_id":"jev"}',
            },
            "update": {"param_value": '{"enabled":true,"model_id":"jev"}'},
        },
    )
    assert await store.settings() == TaskClassifierSettings(enabled=True, model_id="jev")


def test_typesafe_custom_base_without_deployment_key_is_rejected() -> None:
    from litellm.proxy.db.model_insights_task_classifier import build_task_classifier_client

    deployment = system_one_deployments(
        _Router([_deployment("jev", "typesafe/jev-latest", api_base="https://collector.invalid")])
    )[0]

    with pytest.raises(ValueError, match="custom api_base needs its own api_key"):
        build_task_classifier_client(deployment)


@pytest.mark.asyncio
async def test_unreadable_settings_send_usage_down_the_direct_rollup_path() -> None:
    store, _ = _store(_Client({}))
    store.prisma_client.db.litellm_config.find_unique = AsyncMock(side_effect=RuntimeError("db down"))

    assert not await TaskClassifierBatcher(store).enqueue(_transaction(), _payload("Fix it"))
