from collections.abc import Mapping
from typing import Final, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.litellm_core_utils.prompt_templates.json_schema_utils import schema_children
from litellm.models.team import LiteLLM_TeamTable
from litellm.models.workflow_definition import (
    Binding,
    ChatCompletionNode,
    DefinitionError,
    PythonNode,
    ValueBinding,
    WorkflowBundle,
    WorkflowDefinitionStore,
    WorkflowDraft,
    WorkflowSnapshot,
    parsed_bundle,
)
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth, user_api_key_has_admin_view
from litellm.proxy.management.teams.authz import is_team_admin, is_team_member
from litellm.proxy.workflows.monty.compiler import compile_program
from litellm.proxy.workflows.monty.models import MontyFailure, MontyLimits

_MESSAGES: Final = TypeAdapter(list[ChatCompletionMessageParam])
_LIMITS: Final = MontyLimits()
_WRITER_ROLES: Final = frozenset(
    {LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.ORG_ADMIN, LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.TEAM}
)


def requested_team(caller: UserAPIKeyAuth, team_id: str | None) -> str | DefinitionError:
    selected: Final = team_id if team_id is not None else caller.team_id
    if not selected:
        return DefinitionError("team_required", "Specify team_id for this workflow", "team_id")
    if caller.team_id and caller.team_id != selected and not caller.is_session_token:
        return DefinitionError("not_found", "Workflow not found")
    return selected


def author_identity(caller: UserAPIKeyAuth) -> str | DefinitionError:
    if not caller.user_id or caller.user_role not in _WRITER_ROLES:
        return DefinitionError("forbidden", "Workflow authoring requires a team member identity")
    return caller.user_id


async def authorize_team(
    store: WorkflowDefinitionStore,
    caller: UserAPIKeyAuth,
    team_id: str,
    action: Literal["read", "write", "publish", "invoke"],
) -> LiteLLM_TeamTable | DefinitionError:
    scope: Final = requested_team(caller, team_id)
    if isinstance(scope, DefinitionError):
        return scope
    team: Final = await store.get_team(scope)
    if team is None or team.blocked:
        return DefinitionError("not_found", "Workflow not found")
    if action == "invoke" and (caller.team_id == scope or user_api_key_has_admin_view(caller)):
        return team
    if not user_api_key_has_admin_view(caller) and not is_team_member(caller, team):
        return DefinitionError("not_found", "Workflow not found")
    if action == "read":
        return team
    if isinstance(identity := author_identity(caller), DefinitionError):
        return identity
    if action == "publish" and caller.user_role != LitellmUserRoles.PROXY_ADMIN and not is_team_admin(caller, team):
        return DefinitionError("forbidden", "Publishing requires a team administrator")
    return team


async def authorized_draft(
    store: WorkflowDefinitionStore,
    caller: UserAPIKeyAuth,
    workflow_id: str,
    team_id: str | None = None,
    *,
    action: Literal["read", "write", "publish"] = "read",
) -> WorkflowDraft | DefinitionError:
    key_team: Final = None if caller.is_session_token else caller.team_id
    definition: Final = await store.get_draft(workflow_id, team_id if team_id is not None else key_team)
    if definition is None:
        return DefinitionError("not_found", "Workflow not found")
    access: Final = await authorize_team(store, caller, definition.draft.team_id, action)
    return access if isinstance(access, DefinitionError) else definition.draft


async def published_workflow(
    store: WorkflowDefinitionStore, caller: UserAPIKeyAuth, name: str, team_id: str | None = None
) -> WorkflowSnapshot | DefinitionError:
    scope: Final = requested_team(caller, team_id)
    if isinstance(scope, DefinitionError):
        return scope
    access: Final = await authorize_team(store, caller, scope, "invoke")
    if isinstance(access, DefinitionError):
        return access
    snapshot: Final = await store.get_published(name, scope)
    return snapshot if snapshot is not None else DefinitionError("not_found", "Published workflow not found")


async def snapshot_by_id(
    store: WorkflowDefinitionStore, caller: UserAPIKeyAuth, snapshot_id: str
) -> WorkflowSnapshot | DefinitionError:
    key_team: Final = None if caller.is_session_token else caller.team_id
    snapshot: Final = await store.get_snapshot(snapshot_id, key_team)
    if snapshot is None:
        return DefinitionError("not_found", "Workflow snapshot not found")
    access: Final = await authorize_team(store, caller, snapshot.team_id, "read")
    return access if isinstance(access, DefinitionError) else snapshot


def check_etag(draft: WorkflowDraft, etag: str | None) -> DefinitionError | None:
    if etag is None:
        return DefinitionError("precondition_required", "If-Match is required", "If-Match")
    if draft.etag != etag:
        return DefinitionError("stale_draft", "The draft has changed. Read it again before retrying", "If-Match")
    return None


async def prepare_draft(
    store: WorkflowDefinitionStore,
    caller: UserAPIKeyAuth,
    workflow_id: str,
    etag: str | None,
    team_id: str | None = None,
) -> WorkflowSnapshot | DefinitionError:
    draft: Final = await authorized_draft(store, caller, workflow_id, team_id, action="write")
    if isinstance(draft, DefinitionError):
        return draft
    if (failure := check_etag(draft, etag)) is not None:
        return failure
    if (invalid := validate_bundle(parsed_bundle(draft.bundle))) is not None:
        return invalid
    return await store.prepare_draft(workflow_id, draft.team_id, draft.etag)


def _local_references(schema: Mapping[str, object], depth: int = 0) -> bool:
    if depth > 64:
        return False
    if any(keyword in schema for keyword in ("$id", "$dynamicRef", "$recursiveRef", "$dynamicAnchor")):
        return False
    reference: Final = schema.get("$ref")
    if reference is not None and (not isinstance(reference, str) or not reference.startswith("#/")):
        return False
    return all(_local_references(child, depth + 1) for child in schema_children(schema))


def _schema_error(schema: Mapping[str, JsonValue], param: str) -> DefinitionError | None:
    try:
        Draft202012Validator.check_schema(schema)
        if not _local_references(schema):
            return DefinitionError("invalid_bundle", "Schemas must use local references and bounded nesting", param)
    except (SchemaError, RecursionError):
        return DefinitionError("invalid_bundle", "Invalid JSON schema", param)
    return None


def _binding_error(binding: Binding, previous: frozenset[str], param: str) -> DefinitionError | None:
    if isinstance(binding, ValueBinding):
        return None
    if binding.source == "$input" and binding.output == "messages":
        return None
    if binding.source in previous and binding.output == "result":
        return None
    return DefinitionError("invalid_bundle", "Bindings must reference input messages or an earlier node result", param)


def _node_error(node: ChatCompletionNode | PythonNode, previous: frozenset[str]) -> DefinitionError | None:
    if node.id in previous or node.id == "runs":
        return DefinitionError("invalid_bundle", "Node IDs must be unique and cannot be reserved", f"nodes.{node.id}")
    if isinstance(node, ChatCompletionNode):
        if node.model.startswith("workflow/") or "://" in node.model:
            return DefinitionError("invalid_bundle", "Use a configured gateway model group", f"nodes.{node.id}.model")
        if (failure := _binding_error(node.messages, previous, f"nodes.{node.id}.messages")) is not None:
            return failure
        if isinstance(node.messages, ValueBinding):
            try:
                _MESSAGES.validate_python(node.messages.value, strict=True)
            except ValidationError:
                return DefinitionError("invalid_bundle", "Invalid chat messages", f"nodes.{node.id}.messages")
        if node.response_format is not None:
            return _schema_error(node.response_format.json_schema.schema_, f"nodes.{node.id}.response_format")
        return None
    if set(node.inputs) != set(node.program.parameter_names):
        return DefinitionError(
            "invalid_bundle", "Bind exactly the Python program parameters", f"nodes.{node.id}.inputs"
        )
    for name, binding in node.inputs.items():
        if (failure := _binding_error(binding, previous, f"nodes.{node.id}.inputs.{name}")) is not None:
            return failure
    compiled: Final = compile_program(node.program, _LIMITS)
    if isinstance(compiled, MontyFailure):
        return DefinitionError("invalid_bundle", compiled.message[:1024], f"nodes.{node.id}.program")
    return None


def validate_bundle(bundle: WorkflowBundle) -> DefinitionError | None:
    if bundle.name == "runs":
        return DefinitionError("invalid_bundle", "The workflow name is reserved", "bundle.name")
    if len(bundle.model_dump_json(by_alias=True).encode()) > 1024 * 1024:
        return DefinitionError("invalid_bundle", "Workflow bundles cannot exceed 1 MiB", "bundle")
    if (failure := _schema_error(bundle.input_schema, "bundle.input_schema")) is not None:
        return failure
    for index, node in enumerate(bundle.nodes):
        previous: frozenset[str] = frozenset(prior.id for prior in bundle.nodes[:index])
        if (node_failure := _node_error(node, previous)) is not None:
            return node_failure
    if bundle.output.source != bundle.nodes[-1].id or bundle.output.output != "result":
        return DefinitionError("invalid_bundle", "The workflow output must be the final node result", "bundle.output")
    return None
