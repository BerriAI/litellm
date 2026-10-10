import json
from collections.abc import Mapping
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from litellm.proxy.policy_engine.policy_registry import PolicyRegistry
from litellm.types.proxy.policy_engine import PolicyCreateRequest, PolicyUpdateRequest

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

_REQUESTED_PIPELINE = {"mode": "pre_call", "steps": [{"guardrail": "pii-guard", "on_fail": "next"}]}

_STORED_PIPELINE = {
    "mode": "pre_call",
    "steps": [
        {
            "guardrail": "pii-guard",
            "on_fail": "next",
            "on_pass": "allow",
            "on_error": None,
            "pass_data": False,
            "modify_response_message": None,
        }
    ],
}

_MALFORMED_PIPELINES = [
    pytest.param({"mode": "pre_call", "steps": []}, "steps", id="no-steps"),
    pytest.param({"mode": "pre_call"}, "steps", id="steps-missing"),
    pytest.param({"steps": [{"guardrail": "pii-guard"}]}, "mode", id="mode-missing"),
    pytest.param({"mode": "during_call", "steps": [{"guardrail": "pii-guard"}]}, "mode", id="unknown-mode"),
    pytest.param(
        {"mode": "pre_call", "steps": [{"guardrail": "pii-guard"}], "order": "strict"}, "order", id="unknown-key"
    ),
    pytest.param({"mode": "pre_call", "steps": "pii-guard"}, "steps", id="steps-not-a-list"),
]


class _PolicyTable:
    def __init__(self) -> None:
        self.writes: list[Mapping[str, object]] = []

    def _row(self, data: Mapping[str, object]) -> SimpleNamespace:
        pipeline = data.get("pipeline")
        return SimpleNamespace(
            policy_id="policy-1",
            policy_name=data.get("policy_name", "pii-policy"),
            version_number=1,
            version_status=data.get("version_status", "draft"),
            parent_version_id=None,
            is_latest=True,
            published_at=None,
            production_at=None,
            inherit=None,
            description=None,
            guardrails_add=data.get("guardrails_add", []),
            guardrails_remove=data.get("guardrails_remove", []),
            condition=None,
            pipeline=json.loads(pipeline) if isinstance(pipeline, str) else None,
            created_at=_NOW,
            updated_at=_NOW,
            created_by=None,
            updated_by=None,
        )

    async def create(self, data: Mapping[str, object]) -> SimpleNamespace:
        self.writes.append(data)
        return self._row(data)

    async def find_unique(self, where: Mapping[str, object]) -> SimpleNamespace:
        return self._row({"policy_name": "pii-policy", "version_status": "draft"})

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> SimpleNamespace:
        self.writes.append(data)
        return self._row(data)


def _prisma_client(table: _PolicyTable) -> SimpleNamespace:
    return SimpleNamespace(db=SimpleNamespace(litellm_policytable=table))


@pytest.mark.asyncio
async def test_add_policy_to_db_stores_the_pipeline_with_step_defaults_filled_in():
    table = _PolicyTable()
    registry = PolicyRegistry()

    response = await registry.add_policy_to_db(
        PolicyCreateRequest(policy_name="pii-policy", guardrails_add=["pii-guard"], pipeline=_REQUESTED_PIPELINE),
        _prisma_client(table),
    )

    assert [json.loads(str(write["pipeline"])) for write in table.writes] == [_STORED_PIPELINE]
    assert response.pipeline == _STORED_PIPELINE
    stored_policy = registry.get_policy("pii-policy")
    assert stored_policy is not None
    assert stored_policy.pipeline is not None
    assert stored_policy.pipeline.model_dump() == _STORED_PIPELINE


@pytest.mark.asyncio
async def test_update_policy_in_db_stores_the_pipeline_with_step_defaults_filled_in():
    table = _PolicyTable()

    response = await PolicyRegistry().update_policy_in_db(
        "policy-1",
        PolicyUpdateRequest(pipeline=_REQUESTED_PIPELINE),
        _prisma_client(table),
    )

    assert [json.loads(str(write["pipeline"])) for write in table.writes] == [_STORED_PIPELINE]
    assert response.pipeline == _STORED_PIPELINE


@pytest.mark.parametrize(("pipeline", "rejected_field"), _MALFORMED_PIPELINES)
@pytest.mark.asyncio
async def test_add_policy_to_db_rejects_a_malformed_pipeline_before_writing(
    pipeline: dict[str, object], rejected_field: str
):
    table = _PolicyTable()
    registry = PolicyRegistry()

    with pytest.raises(Exception, match="Error adding policy to DB: 1 validation error") as raised:
        await registry.add_policy_to_db(
            PolicyCreateRequest(policy_name="pii-policy", pipeline=pipeline),
            _prisma_client(table),
        )

    assert str(raised.value).startswith(
        f"Error adding policy to DB: 1 validation error for GuardrailPipeline\n{rejected_field}\n"
    )
    assert table.writes == []
    assert registry.get_policy("pii-policy") is None


@pytest.mark.parametrize(("pipeline", "rejected_field"), _MALFORMED_PIPELINES)
@pytest.mark.asyncio
async def test_update_policy_in_db_rejects_a_malformed_pipeline_before_writing(
    pipeline: dict[str, object], rejected_field: str
):
    table = _PolicyTable()

    with pytest.raises(Exception, match="Error updating policy in DB: 1 validation error") as raised:
        await PolicyRegistry().update_policy_in_db(
            "policy-1",
            PolicyUpdateRequest(pipeline=pipeline),
            _prisma_client(table),
        )

    assert str(raised.value).startswith(
        f"Error updating policy in DB: 1 validation error for GuardrailPipeline\n{rejected_field}\n"
    )
    assert table.writes == []
