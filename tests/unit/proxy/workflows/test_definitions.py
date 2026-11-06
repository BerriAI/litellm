import hashlib
from typing import Final

import pytest

from litellm.models.workflow_definition import DefinitionError, WorkflowBundle, WorkflowDraft, WorkflowSnapshot
from litellm.proxy.workflows.definitions import freeze_draft, published_workflow, validate_bundle
from litellm.repositories.workflow_definition_repository import WorkflowDefinitionRepository
from tests.unit.proxy.workflow_definition_rows import DefinitionRows, bundle, caller


def _python_bundle(source: str) -> WorkflowBundle:
    original: Final = bundle().model_dump(mode="json", by_alias=True, exclude_unset=True)
    return WorkflowBundle.model_validate(
        {
            **original,
            "output": {"from": "node_2", "output": "result"},
            "nodes": [
                *original["nodes"],
                {
                    "id": "node_2",
                    "type": "python",
                    "timeout_seconds": 10.0,
                    "retry": {"max_attempts": 1, "backoff_seconds": 0.0},
                    "inputs": {"text": {"from": "node_1", "output": "result"}},
                    "program": {
                        "contract_version": "1",
                        "source": source,
                        "function_name": "transform",
                        "parameter_names": ["text"],
                        "input_schema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                            "additionalProperties": False,
                        },
                        "output_schema": {"type": "string"},
                        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    },
                },
            ],
        }
    )


def test_validation_compiles_python_without_executing_its_body() -> None:
    program: Final = _python_bundle("def transform(text: str) -> str:\n    return str(1 // 0)\n")
    assert validate_bundle(program) is None


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef transform(text: str) -> str:\n    return text\n",
        "def transform(text: str) -> str:\n    return open(text).read()\n",
    ],
)
def test_validation_rejects_imports_and_host_capabilities(source: str) -> None:
    result: Final = validate_bundle(_python_bundle(source))
    assert isinstance(result, DefinitionError)
    assert result.code == "invalid_bundle"
    assert result.param == "nodes.node_2.program"


def test_validation_rejects_forward_references_and_external_schema_refs() -> None:
    original: Final = bundle().model_dump(mode="json", by_alias=True, exclude_unset=True)
    forward: Final = WorkflowBundle.model_validate(
        {**original, "nodes": [{**original["nodes"][0], "messages": {"from": "later", "output": "result"}}]}
    )
    remote: Final = WorkflowBundle.model_validate({**original, "input_schema": {"$ref": "https://example.com/schema"}})
    assert validate_bundle(forward) == DefinitionError(
        "invalid_bundle", "Bindings must reference input messages or an earlier node result", "nodes.node_1.messages"
    )
    assert validate_bundle(remote) == DefinitionError(
        "invalid_bundle", "Schemas must use local references and bounded nesting", "bundle.input_schema"
    )


@pytest.mark.parametrize("keyword", ["$id", "$dynamicRef", "$recursiveRef"])
def test_validation_rejects_schema_reference_scope_changes(keyword: str) -> None:
    original: Final = bundle().model_dump(mode="json", by_alias=True, exclude_unset=True)
    scoped: Final = WorkflowBundle.model_validate({**original, "input_schema": {keyword: "https://example.com/schema"}})
    assert validate_bundle(scoped) == DefinitionError(
        "invalid_bundle", "Schemas must use local references and bounded nesting", "bundle.input_schema"
    )


@pytest.mark.asyncio
async def test_application_identity_can_resolve_published_workflow_but_cannot_freeze_draft() -> None:
    store: Final = WorkflowDefinitionRepository(DefinitionRows().database)
    draft: Final = await store.create(bundle(), "team", "author")
    assert isinstance(draft, WorkflowDraft)
    snapshot: Final = await store.publish(draft.workflow_id, "team", draft.etag, draft.digest, "reviewer")
    assert isinstance(snapshot, WorkflowSnapshot)
    assert await published_workflow(store, caller(None), "example") == snapshot
    denied: Final = await freeze_draft(store, caller(None), draft.workflow_id, draft.etag)
    assert isinstance(denied, DefinitionError)
    assert denied.code == "not_found"
