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


_VERSION_CONTENT = {
    "inherit": "base-policy",
    "description": "blocks pii",
    "guardrails_add": ["pii-guard"],
    "guardrails_remove": ["legacy-guard"],
    "condition": {"model": "gpt-4"},
    "pipeline": _STORED_PIPELINE,
}

_CHANGED_VERSION_CONTENT = {
    "inherit": None,
    "description": "",
    "guardrails_add": ["pii-guard", "toxicity-guard"],
    "guardrails_remove": [],
    "condition": {"model": "gpt-4o"},
    "pipeline": None,
}


class _VersionsTable:
    def __init__(self, rows: Mapping[str, SimpleNamespace]) -> None:
        self.rows = rows
        self.lookups: list[Mapping[str, object]] = []

    async def find_unique(self, where: Mapping[str, object]) -> SimpleNamespace | None:
        self.lookups.append(where)
        return self.rows.get(str(where["policy_id"]))


def _version_row(policy_id: str, content: Mapping[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        policy_id=policy_id,
        policy_name="pii-policy",
        version_number=1,
        version_status="draft",
        parent_version_id=None,
        is_latest=True,
        published_at=None,
        production_at=None,
        created_at=_NOW,
        updated_at=_NOW,
        created_by=None,
        updated_by=None,
        **{**_VERSION_CONTENT, **content},
    )


def _versions_client(table: _VersionsTable) -> SimpleNamespace:
    return SimpleNamespace(db=SimpleNamespace(litellm_policytable=table))


@pytest.mark.parametrize("field", list(_VERSION_CONTENT))
@pytest.mark.asyncio
async def test_compare_versions_reports_exactly_the_one_field_that_differs(field: str):
    value_a = _VERSION_CONTENT[field]
    value_b = _CHANGED_VERSION_CONTENT[field]
    table = _VersionsTable(
        {
            "version-a": _version_row("version-a", {}),
            "version-b": _version_row("version-b", {field: value_b}),
        }
    )

    result = await PolicyRegistry().compare_versions("version-a", "version-b", _versions_client(table))

    assert result.field_diffs == {field: {"version_a": value_a, "version_b": value_b}}
    assert [type(value) for value in result.field_diffs[field].values()] == [type(value_a), type(value_b)]
    assert (result.version_a.policy_id, result.version_b.policy_id) == ("version-a", "version-b")
    assert table.lookups == [{"policy_id": "version-a"}, {"policy_id": "version-b"}]


@pytest.mark.asyncio
async def test_compare_versions_lists_every_differing_field_in_content_order_with_each_side():
    table = _VersionsTable(
        {
            "version-a": _version_row("version-a", _CHANGED_VERSION_CONTENT),
            "version-b": _version_row("version-b", {}),
        }
    )

    result = await PolicyRegistry().compare_versions("version-a", "version-b", _versions_client(table))

    assert list(result.field_diffs) == list(_VERSION_CONTENT)
    assert result.field_diffs == {
        field: {"version_a": _CHANGED_VERSION_CONTENT[field], "version_b": _VERSION_CONTENT[field]}
        for field in _VERSION_CONTENT
    }
    assert result.field_diffs["inherit"]["version_a"] is None
    assert result.field_diffs["pipeline"]["version_a"] is None


@pytest.mark.asyncio
async def test_compare_versions_ignores_version_metadata_and_treats_unset_guardrail_lists_as_empty():
    row_a = _version_row("version-a", {"guardrails_add": None, "guardrails_remove": None})
    row_b = _version_row("version-b", {"guardrails_add": [], "guardrails_remove": []})
    row_b.policy_name = "renamed-policy"
    row_b.version_number = 7
    row_b.version_status = "production"
    row_b.parent_version_id = "version-a"
    row_b.is_latest = False
    row_b.created_by = "admin"
    table = _VersionsTable({"version-a": row_a, "version-b": row_b})

    result = await PolicyRegistry().compare_versions("version-a", "version-b", _versions_client(table))

    assert result.field_diffs == {}
    assert (result.version_a.version_number, result.version_b.version_number) == (1, 7)
    assert (result.version_a.guardrails_add, result.version_b.guardrails_add) == ([], [])


@pytest.mark.parametrize(
    ("stored_versions", "missing_version"),
    [
        pytest.param(("version-b",), "version-a", id="first-missing"),
        pytest.param(("version-a",), "version-b", id="second-missing"),
        pytest.param((), "version-a", id="both-missing-names-the-first"),
    ],
)
@pytest.mark.asyncio
async def test_compare_versions_names_the_missing_version_after_looking_both_up(
    stored_versions: tuple[str, ...], missing_version: str
):
    table = _VersionsTable({policy_id: _version_row(policy_id, {}) for policy_id in stored_versions})

    with pytest.raises(Exception, match=f"^Error comparing versions: Policy {missing_version} not found$") as raised:
        await PolicyRegistry().compare_versions("version-a", "version-b", _versions_client(table))

    assert type(raised.value) is Exception
    assert table.lookups == [{"policy_id": "version-a"}, {"policy_id": "version-b"}]
