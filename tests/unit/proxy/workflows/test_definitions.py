import hashlib
from collections.abc import Mapping
from typing import Final

import pytest
from liteflow import ChatCompletion, Input, Python, Workflow, compile_workflow
from pydantic import JsonValue

from litellm.models.workflow_definition import (
    DefinitionError,
    PythonNode,
    WorkflowBundle,
    WorkflowDraft,
    WorkflowSnapshot,
)
from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.workflows.definitions import authorize_team, freeze_draft, published_workflow, validate_bundle
from litellm.repositories.workflow_definition_repository import WorkflowDefinitionRepository
from tests.unit.proxy.workflow_definition_rows import DefinitionRows, bundle, caller


def transform(text: str) -> str:
    return str(1 // 0)


def _python_bundle(source: str) -> WorkflowBundle:
    workflow: Final = Workflow(
        name="example",
        nodes=[
            ChatCompletion(model="gateway-model", messages=Input.messages),
            Python(fn=transform),
        ],
    )
    original: Final = WorkflowBundle.model_validate(compile_workflow(workflow))
    node: Final = original.nodes[-1]
    assert isinstance(node, PythonNode)
    program: Final = node.program.model_copy(
        update={"source": source, "source_sha256": hashlib.sha256(source.encode()).hexdigest()}
    )
    return original.model_copy(update={"nodes": (*original.nodes[:-1], node.model_copy(update={"program": program}))})


def test_validation_compiles_python_without_executing_its_body() -> None:
    assert validate_bundle(_python_bundle("def transform(text: str) -> str:\n    return str(1 // 0)\n")) is None


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef transform(text: str) -> str:\n    return text\n",
        "def transform(text: str) -> str:\n    return open(text).read()\n",
    ],
)
def test_validation_rejects_imports_and_host_capabilities(source: str) -> None:
    result: Final = validate_bundle(_python_bundle(source))
    assert isinstance(result, DefinitionError) and result.code == "invalid_bundle"
    assert result.param == "nodes.node_2.program"


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"$ref": {"type": "string"}, "$id": {"type": "number"}}},
        {"const": {"$ref": "literal"}},
        {"enum": [{"$dynamicRef": "literal"}]},
    ],
)
def test_schema_keywords_inside_property_names_and_literals_are_data(schema: Mapping[str, JsonValue]) -> None:
    assert validate_bundle(bundle().model_copy(update={"input_schema": schema})) is None


@pytest.mark.parametrize(
    "schema",
    [
        {"properties": {"safe": {"$ref": "https://example.com/schema"}}},
        {"patternProperties": {"^a": {"$id": "https://example.com/schema"}}},
        {"$dynamicRef": "https://example.com/schema"},
        {"$recursiveRef": "https://example.com/schema"},
    ],
)
def test_reference_scope_is_checked_only_at_schema_positions(schema: Mapping[str, JsonValue]) -> None:
    assert validate_bundle(bundle().model_copy(update={"input_schema": schema})) == DefinitionError(
        "invalid_bundle", "Schemas must use local references and bounded nesting", "bundle.input_schema"
    )


def test_forward_reference_cannot_reach_a_provider() -> None:
    original: Final = bundle().model_dump(mode="json", by_alias=True, exclude_unset=True)
    forward: Final = WorkflowBundle.model_validate(
        {**original, "nodes": [{**original["nodes"][0], "messages": {"from": "later", "output": "result"}}]}
    )
    assert validate_bundle(forward) == DefinitionError(
        "invalid_bundle", "Bindings must reference input messages or an earlier node result", "nodes.node_1.messages"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY])
async def test_key_team_restriction_precedes_global_user_role(role: LitellmUserRoles) -> None:
    store: Final = WorkflowDefinitionRepository(DefinitionRows().database)
    identity: Final = caller("reviewer").model_copy(update={"user_role": role, "via_virtual_key": True})
    assert await authorize_team(store, identity, "other-team", "read") == DefinitionError(
        "not_found", "Workflow not found"
    )
    session: Final = identity.model_copy(update={"is_session_token": True})
    assert not isinstance(await authorize_team(store, session, "other-team", "read"), DefinitionError)


@pytest.mark.asyncio
async def test_application_identity_can_invoke_but_cannot_freeze_a_draft() -> None:
    store: Final = WorkflowDefinitionRepository(DefinitionRows().database)
    draft: Final = await store.create(bundle(), "team", "author")
    assert isinstance(draft, WorkflowDraft)
    snapshot: Final = await store.publish(draft.workflow_id, "team", draft.etag, draft.digest, "reviewer")
    assert isinstance(snapshot, WorkflowSnapshot)
    assert await published_workflow(store, caller(None), "example") == snapshot
    denied: Final = await freeze_draft(store, caller(None), draft.workflow_id, draft.etag)
    assert isinstance(denied, DefinitionError) and denied.code == "not_found"
