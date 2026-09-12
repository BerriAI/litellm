import asyncio
import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, TypedDict

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.constants import REDACTED_BY_LITELM_STRING
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.proxy.management_helpers.object_permission_utils import (
    handle_update_object_permission_common,
)
from litellm.proxy.utils import PrismaClient
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.table_repositories import AgentsRepository, ObjectPermissionRepository
from litellm.types.agents import AgentConfig, AgentResponse, PatchAgentRequest

if TYPE_CHECKING:
    from prisma import models as prisma_models


class AgentObjectPermissionRecord(Protocol):
    def model_dump(self) -> dict[str, object]: ...

    def dict(self) -> dict[str, object]: ...


class AgentRecordDump(TypedDict):
    agent_id: str
    agent_name: str
    litellm_params: dict[str, object] | None
    agent_card_params: dict[str, object]
    static_headers: dict[str, str] | None
    extra_headers: list[str] | None
    object_permission: dict[str, object] | None
    spend: float
    tpm_limit: int | None
    rpm_limit: int | None
    session_tpm_limit: int | None
    session_rpm_limit: int | None
    created_at: datetime
    updated_at: datetime
    created_by: str | None
    updated_by: str | None


class AgentRecord(Protocol):
    @property
    def agent_id(self) -> str: ...

    @property
    def agent_name(self) -> str: ...

    @property
    def litellm_params(self) -> Mapping[str, object] | None: ...

    @property
    def object_permission_id(self) -> str | None: ...

    @property
    def object_permission(self) -> AgentObjectPermissionRecord | None: ...

    @property
    def spend(self) -> float: ...

    def model_dump(self) -> AgentRecordDump: ...

    def __iter__(self) -> Iterator[tuple[str, object]]: ...


class AgentTableClient(Protocol):
    async def create(
        self,
        data: Mapping[str, object],
        include: Mapping[str, object] | None = None,
    ) -> AgentRecord: ...

    async def find_unique(
        self,
        where: Mapping[str, object],
        include: Mapping[str, object] | None = None,
    ) -> AgentRecord | None: ...

    async def find_many(
        self,
        where: Mapping[str, object] | None = None,
        order: Mapping[str, str] | None = None,
        include: Mapping[str, object] | None = None,
    ) -> Sequence[AgentRecord]: ...

    async def update(
        self,
        data: Mapping[str, object],
        where: Mapping[str, object],
        include: Mapping[str, object] | None = None,
    ) -> AgentRecord | None: ...

    async def delete(
        self,
        where: Mapping[str, object],
        include: Mapping[str, object] | None = None,
    ) -> AgentRecord | None: ...


def agents_table(prisma_client: PrismaClient) -> AgentTableClient:
    table: Final[AgentTableClient] = AgentsRepository(prisma_client).table  # pyright: ignore[reportAssignmentType]  # prisma rows type model_dump() as dict[str, Any]
    return table


def object_permission_table(
    prisma_client: PrismaClient,
) -> "TableActions[prisma_models.LiteLLM_ObjectPermissionTable]":
    table: Final[TableActions[prisma_models.LiteLLM_ObjectPermissionTable]] = ObjectPermissionRepository(
        prisma_client
    ).table
    return table


def _dump_agent_params(raw: Mapping[str, object]) -> dict[str, object]:
    model_dump: Final[Callable[[], dict[str, object]] | None] = getattr(raw, "model_dump", None)
    if model_dump is not None:
        return model_dump()
    return dict(raw) if raw else {}


_AGENT_PARAMS_MASKER: Final = SensitiveDataMasker()
_REDACT_AGENT_PARAMS_MAX_DEPTH: Final = 10
_AGENT_PARAMS_ADAPTER: Final[TypeAdapter[dict[str, object]]] = TypeAdapter(
    dict[str, object]
)  # mutable-ok: safe_dumps() and AgentResponse.litellm_params both require a real dict, not a Mapping
_AGENT_PARAMS_SEQUENCE_ADAPTER: Final[TypeAdapter[tuple[object, ...]]] = TypeAdapter(tuple[object, ...])
_EMPTY_LITELLM_PARAMS: Final[Mapping[str, object]] = MappingProxyType({})


def redact_sensitive_agent_litellm_params(litellm_params: object, _depth: int = 0) -> object:
    """
    Replace credential-bearing values in an agent's litellm_params with
    ``REDACTED_BY_LITELM_STRING`` while preserving non-secret keys (``model``,
    ``is_public``, rate-limit config). Used so list/get/create/update
    responses never echo a stored provider credential back to the caller.

    Handles a plain dict, a JSON-serialized string (some callers hold the
    in-memory registry's params that way), and ``None`` at the top level;
    anything else is passed through. Recursion depth is bounded to match the
    convention documented in ``tests/code_coverage_tests/recursive_detector.py``.
    """
    if litellm_params is None:
        return None
    if isinstance(litellm_params, str):
        if _depth >= _REDACT_AGENT_PARAMS_MAX_DEPTH:
            return REDACTED_BY_LITELM_STRING
        try:
            parsed_params: Final = _AGENT_PARAMS_ADAPTER.validate_json(litellm_params)
        except ValidationError:
            return REDACTED_BY_LITELM_STRING
        return json.dumps(_redact_agent_params_tree(parsed_params, _depth + 1))
    return _redact_agent_params_tree(litellm_params, _depth)


def _redact_agent_params_tree(value: object, _depth: int) -> object:
    """Structural recursion over an already-parsed litellm_params value: a
    dict redacts sensitive keys and recurses into the rest, a list redacts
    each element (so a secret nested inside a list of provider configs is
    still caught), and anything else -- including a plain string leaf, which
    must never be re-interpreted as a JSON blob -- passes through unchanged.
    """
    if _depth >= _REDACT_AGENT_PARAMS_MAX_DEPTH:
        return REDACTED_BY_LITELM_STRING
    if isinstance(value, list):
        typed_items: Final = _AGENT_PARAMS_SEQUENCE_ADAPTER.validate_python(value)
        return tuple(_redact_agent_params_tree(item, _depth + 1) for item in typed_items)
    if not isinstance(value, dict):
        return value
    typed_params: Final = _AGENT_PARAMS_ADAPTER.validate_python(value)
    return {
        key: (
            REDACTED_BY_LITELM_STRING
            if _AGENT_PARAMS_MASKER.is_sensitive_key(key)
            else _redact_agent_params_tree(nested_value, _depth + 1)
        )
        for key, nested_value in typed_params.items()
    }  # mutable-ok: consumed by json.dumps()/AgentResponse.litellm_params, both of which require a real dict


def parse_agent_litellm_params(value: object) -> Mapping[str, object]:
    """Normalize a stored litellm_params column to a read-only mapping.

    The prisma Json column comes back as either an already-parsed dict or a
    JSON string depending on the read path, so handle both rather than
    assuming one. Only ever read from (merge-source lookups), never mutated
    or re-serialized directly, so a read-only view is enough here.
    """
    if isinstance(value, str):
        try:
            return _AGENT_PARAMS_ADAPTER.validate_json(value)
        except ValidationError:
            return _EMPTY_LITELLM_PARAMS
    if isinstance(value, Mapping):
        try:
            return _AGENT_PARAMS_ADAPTER.validate_python(value)
        except ValidationError:
            return _EMPTY_LITELLM_PARAMS
    return _EMPTY_LITELLM_PARAMS


_MISSING_AGENT_PARAM: Final = object()
_RESTORE_AGENT_PARAMS_MAX_DEPTH: Final = 10


def _restore_redacted_nested_value(incoming_value: object, existing_value: object, _depth: int) -> object:
    """Recurse into a non-sensitively-named dict/list value so a secret
    nested underneath it (e.g. inside a list of per-provider configs) is
    still restored, not just top-level keys. Mirrors the shapes
    ``redact_sensitive_agent_litellm_params`` recurses into on read, so
    restore and redact stay symmetric.

    List elements are paired with the existing list by position: with no
    stable per-element identity in an arbitrary ``dict[str, object]`` schema,
    index is the same correspondence every other part of this restore (and
    the endpoints' existing full-replace-on-PUT semantics) already assumes.
    This correctly preserves a masked secret across an ordinary edit of that
    same entry's other fields; it does not protect against a caller who both
    reorders/resizes the list AND echoes back a masked marker in the same
    request, which is a known, narrow limitation (see LIT-6736 PR discussion)
    rather than a cross-entry credential leak in the common case.

    A value collapsed to the flat marker by the read side's depth cap is
    recovered wholesale from ``existing_value`` (rather than the marker
    string itself getting persisted) whenever ``existing_value`` isn't
    already that same flat marker. Depth-bounded like its read-side
    counterpart; a value at the cap is returned unchanged rather than
    corrupted.
    """
    if incoming_value == REDACTED_BY_LITELM_STRING and existing_value != REDACTED_BY_LITELM_STRING:
        return existing_value
    if _depth >= _RESTORE_AGENT_PARAMS_MAX_DEPTH:
        return incoming_value
    if isinstance(incoming_value, Mapping):
        typed_incoming_map: Final = _AGENT_PARAMS_ADAPTER.validate_python(incoming_value)
        existing_map: Final = (
            _AGENT_PARAMS_ADAPTER.validate_python(existing_value)
            if isinstance(existing_value, Mapping)
            else _EMPTY_LITELLM_PARAMS
        )
        return _restore_redacted_litellm_params(typed_incoming_map, existing_map, _depth + 1)
    if isinstance(incoming_value, (list, tuple)):
        typed_incoming_seq: Final = _AGENT_PARAMS_SEQUENCE_ADAPTER.validate_python(incoming_value)
        existing_seq: Final = (
            _AGENT_PARAMS_SEQUENCE_ADAPTER.validate_python(existing_value)
            if isinstance(existing_value, (list, tuple))
            else ()
        )
        return tuple(
            _restore_redacted_nested_value(
                item,
                existing_seq[index] if index < len(existing_seq) else None,
                _depth + 1,
            )
            for index, item in enumerate(typed_incoming_seq)
        )
    return incoming_value


def _resolved_agent_param_value(
    key: str,
    incoming: Mapping[str, object],
    existing: Mapping[str, object],
    _depth: int,
) -> object:
    """The value ``key`` should end up with in a restored litellm_params, or
    ``_MISSING_AGENT_PARAM`` when it should be dropped entirely."""
    if key in incoming:
        value: Final = incoming[key]
        if _AGENT_PARAMS_MASKER.is_sensitive_key(key):
            return existing.get(key, _MISSING_AGENT_PARAM) if value == REDACTED_BY_LITELM_STRING else value
        return _restore_redacted_nested_value(value, existing.get(key), _depth)
    if _AGENT_PARAMS_MASKER.is_sensitive_key(key):
        return existing.get(key, _MISSING_AGENT_PARAM)
    return _MISSING_AGENT_PARAM


def _restore_redacted_litellm_params(
    incoming: Mapping[str, object],
    existing: Mapping[str, object],
    _depth: int = 0,
) -> dict[str, object]:
    """Restore the real credential behind any litellm_params value the caller
    echoed back as ``REDACTED_BY_LITELM_STRING``, and behind any sensitive key
    omitted entirely, so an edit to an unrelated field never overwrites (or
    silently drops) a stored provider credential -- the UI never has to
    read-and-resend a secret to keep it. Recurses into nested dicts and lists
    so a secret nested under a non-sensitively-named key is restored too.

    A sensitive key given a real (non-marker) value, including an explicit
    empty string, is treated as a deliberate update -- that's how a caller
    clears a credential. Non-sensitive keys always take the incoming value
    (recursed into), matching the endpoints' existing full-replace-on-PUT /
    merge-on-PATCH semantics for everything that isn't a secret.
    """
    all_keys: Final = frozenset(incoming) | frozenset(existing)
    return {
        key: value
        for key in all_keys
        if (value := _resolved_agent_param_value(key, incoming, existing, _depth)) is not _MISSING_AGENT_PARAM
    }  # mutable-ok: fed to safe_dumps() for JSON-column storage, which requires a real dict


class GrantMigrationResult(NamedTuple):
    rewritten: int
    missed: int


class AgentRegistry:
    def __init__(self):
        self.agent_list: list[AgentResponse] = []
        self.config_agents: tuple[AgentConfig, ...] = ()
        self.config_agent_legacy_ids: Mapping[str, str] = MappingProxyType({})

    def reset_agent_list(self):
        self.agent_list = []

    def register_agent(self, agent_config: AgentResponse):
        self.agent_list.append(agent_config)

    def deregister_agent(self, agent_name: str):
        self.agent_list = [agent for agent in self.agent_list if agent.agent_name != agent_name]

    def get_agent_list(self, agent_names: Sequence[str] | None = None) -> tuple[AgentResponse, ...]:
        if agent_names is not None:
            return tuple(agent for agent in self.agent_list if agent.agent_name in agent_names)
        return tuple(self.agent_list)

    def get_public_agent_list(self) -> tuple[AgentResponse, ...]:
        public_agent_groups: Final = litellm.public_agent_groups
        if public_agent_groups is None:
            return ()
        return tuple(
            agent for agent in self.agent_list if not self.ids_for_agent(agent.agent_id).isdisjoint(public_agent_groups)
        )

    def _create_agent_id(self, agent_config: AgentConfig) -> str:
        return hashlib.sha256(agent_config["agent_name"].encode()).hexdigest()

    def _create_legacy_agent_id(self, agent_config: AgentConfig) -> str:
        return hashlib.sha256(json.dumps(agent_config, sort_keys=True).encode()).hexdigest()

    def ids_for_agent(self, agent_id: str) -> frozenset[str]:
        return frozenset(
            {agent_id, *(legacy for legacy, stable in self.config_agent_legacy_ids.items() if stable == agent_id)}
        )

    def stable_agent_id(self, agent_id: str) -> str:
        return self.config_agent_legacy_ids.get(agent_id, agent_id)

    def load_agents_from_config(self, agent_config: Sequence[AgentConfig] | None = None):
        """
        Register the agents declared in config.yaml and remember them for later rebuilds.

        A config entry is skipped when its ``agent_name`` is already registered, so a
        database record always wins over a config entry that reuses its name and the
        registry never holds two agents under one name. Enforcing that here rather than
        in the caller keeps the guarantee independent of the order the two sources load
        in. Passing ``None`` leaves the remembered agents untouched; passing an empty
        sequence clears them.
        """
        if agent_config is None:
            return

        for agent_config_item in agent_config:
            if not isinstance(agent_config_item, dict):
                raise ValueError("agent_config must be a list of dictionaries")

        self.config_agents = tuple(agent_config)
        self.config_agent_legacy_ids = MappingProxyType(
            {
                self._create_legacy_agent_id(agent_config_item): self._create_agent_id(agent_config_item)
                for agent_config_item in agent_config
                if agent_config_item.get("agent_name") and agent_config_item.get("agent_card_params")
            }
        )

        for agent_config_item in agent_config:
            agent_name = agent_config_item.get("agent_name")
            agent_card_params = agent_config_item.get("agent_card_params")
            if not all([agent_name, agent_card_params]):
                continue

            if any(agent.agent_name == agent_name for agent in self.agent_list):
                continue

            # create a stable hash id for config item
            config_hash = self._create_agent_id(agent_config_item)

            self.register_agent(agent_config=AgentResponse(agent_id=config_hash, **agent_config_item))

    def load_agents_from_db_and_config(
        self,
        agent_config: Sequence[AgentConfig] | None = None,
        db_agents: Sequence[Mapping[str, object]] | None = None,
    ):
        """
        Rebuild the registry from the DB rows plus the agents declared in config.yaml.

        ``agent_config`` defaults to the agents remembered by the last
        ``load_agents_from_config`` call, so a periodic DB reload does not drop
        config-defined agents.

        The DB rows are registered first so that a config entry reusing one of their
        names is dropped by ``load_agents_from_config``, mirroring how config-declared
        MCP servers are unioned under the database registry. Name lookups and
        deregistration both address a single agent, so the registry must never hold two
        under one name.
        """
        self.reset_agent_list()

        if db_agents:
            for db_agent in db_agents:
                if not isinstance(db_agent, dict):
                    raise ValueError("db_agents must be a list of dictionaries")

                self.register_agent(agent_config=AgentResponse.model_validate(db_agent))

        self.load_agents_from_config(agent_config if agent_config is not None else self.config_agents)
        return self.agent_list

    async def migrate_legacy_grant_ids(
        self, table: "TableActions[prisma_models.LiteLLM_ObjectPermissionTable]"
    ) -> GrantMigrationResult:
        """
        Rewrite object_permission.agents rows holding a legacy full-entry hash to the
        stable name-derived id.

        Only the running proxy can do this: the legacy hash is computed from the
        resolved config entry (secrets included), so no SQL migration can know it.
        Persisting the stable id here is what keeps a grant alive across a later
        secret rotation, which re-mints the legacy hash and would otherwise orphan
        the stored value. Idempotent; runs of it after the first find no rows.

        Each write is a compare-and-swap against the agents array read above, so a
        grant edited concurrently is left untouched; the runtime alias keeps covering
        it and the next boot retries the rewrite.
        """
        legacy_ids: Final = tuple(legacy for legacy, stable in self.config_agent_legacy_ids.items() if legacy != stable)
        if not legacy_ids:
            return GrantMigrationResult(rewritten=0, missed=0)
        rows: Final = await table.find_many(where={"agents": {"has_some": legacy_ids}})
        updates: Final = tuple(
            (
                row.object_permission_id,
                tuple(row.agents or ()),
                tuple(dict.fromkeys(self.stable_agent_id(agent_id) for agent_id in row.agents or ())),
            )
            for row in rows
        )
        counts: Final = await asyncio.gather(
            *(
                table.update_many(
                    where={"object_permission_id": object_permission_id, "agents": {"equals": snapshot_agents}},
                    data={"agents": translated_agents},
                )
                for object_permission_id, snapshot_agents, translated_agents in updates
            )
        )
        rewritten: Final = sum(counts)
        return GrantMigrationResult(rewritten=rewritten, missed=len(updates) - rewritten)

    ###########################################################
    ########### DB management helpers for agents ###########
    ############################################################
    async def add_agent_to_db(
        self,
        agent: AgentConfig,
        prisma_client: PrismaClient,
        created_by: str,
        agent_id: str | None = None,
    ) -> AgentResponse:
        """
        Add an agent to the database.

        If ``agent_id`` is provided, it is used as the primary key for the new
        row (otherwise the DB generates a UUID). Callers pass an explicit ID
        when the agent_card_params must reference the agent's own URL before
        the row exists, e.g. the A2A merge in ``create_agent``.
        """
        try:
            agent_name: Final = agent.get("agent_name")

            # Serialize litellm_params. A create has no stored row to restore a
            # secret behind, so a sensitive key submitted as the redaction
            # marker (e.g. a stray client re-post) is dropped rather than
            # persisted as the literal placeholder string.
            litellm_params_obj: Final = agent.get("litellm_params", {})
            litellm_params_dict: Final = _restore_redacted_litellm_params(
                _dump_agent_params(litellm_params_obj), _EMPTY_LITELLM_PARAMS
            )
            litellm_params: Final[str] = safe_dumps(litellm_params_dict)

            # Serialize agent_card_params
            agent_card_params_obj: Final = agent.get("agent_card_params", {})
            agent_card_params_dict: Final[dict[str, object]] = _dump_agent_params(agent_card_params_obj)
            agent_card_params: Final[str] = safe_dumps(agent_card_params_dict)

            # Handle object_permission (MCP tool access for agent)
            object_permission_id: str | None = None
            if agent.get("object_permission") is not None:
                agent_copy: Final = dict(agent)
                object_permission_id = await handle_update_object_permission_common(agent_copy, None, prisma_client)

            # Serialize static_headers
            static_headers_obj: Final = agent.get("static_headers")
            static_headers_val: Final[str | None] = safe_dumps(dict(static_headers_obj)) if static_headers_obj else None

            extra_headers_val: Final = agent.get("extra_headers")

            create_data: Final[dict[str, object]] = {
                "agent_name": agent_name,
                "litellm_params": litellm_params,
                "agent_card_params": agent_card_params,
                "created_by": created_by,
                "updated_by": created_by,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
            if agent_id is not None:
                create_data["agent_id"] = agent_id
            if static_headers_val is not None:
                create_data["static_headers"] = static_headers_val
            if extra_headers_val is not None:
                create_data["extra_headers"] = extra_headers_val
            if object_permission_id is not None:
                create_data["object_permission_id"] = object_permission_id

            for rate_field in (
                "tpm_limit",
                "rpm_limit",
                "session_tpm_limit",
                "session_rpm_limit",
            ):
                _val = agent.get(rate_field)
                if _val is not None:
                    create_data[rate_field] = _val

            # Create agent in DB
            created_agent: Final = await agents_table(prisma_client).create(
                data=create_data,
                include={"object_permission": True},
            )

            created_agent_dict: Final = created_agent.model_dump()
            if created_agent.object_permission is not None:
                try:
                    created_agent_dict["object_permission"] = created_agent.object_permission.model_dump()
                except Exception:
                    created_agent_dict["object_permission"] = created_agent.object_permission.dict()
            return AgentResponse(**created_agent_dict)
        except Exception as e:
            raise Exception(f"Error adding agent to DB: {e}")

    async def delete_agent_from_db(self, agent_id: str, prisma_client: PrismaClient) -> Mapping[str, object]:
        """
        Delete an agent from the database
        """
        try:
            deleted_agent: Final = await agents_table(prisma_client).delete(where={"agent_id": agent_id})
            if deleted_agent is None:
                raise ValueError(f"Agent not found, passed agent_id={agent_id}")
            return dict(deleted_agent)
        except Exception as e:
            raise Exception(f"Error deleting agent from DB: {e}")

    async def patch_agent_in_db(
        self,
        agent_id: str,
        agent: PatchAgentRequest,
        prisma_client: PrismaClient,
        updated_by: str,
    ) -> AgentResponse:
        """
        Patch an agent in the database.

        Get the existing agent from the database and patch it with the new values.

        Args:
            agent_id: The ID of the agent to patch
            agent: The new agent values to patch
            prisma_client: The Prisma client to use
            updated_by: The user ID of the user who is patching the agent

        Returns:
            The patched agent
        """
        try:
            existing_record: Final = await agents_table(prisma_client).find_unique(where={"agent_id": agent_id})
            if existing_record is None:
                raise Exception(f"Agent with ID {agent_id} not found")
            existing_agent: Final[Mapping[str, object]] = dict(existing_record)

            augment_agent: Final = {**existing_agent, **agent}
            update_data: Final[dict[str, object]] = {}
            if augment_agent.get("agent_name"):
                update_data["agent_name"] = augment_agent.get("agent_name")
            if "litellm_params" in agent:
                existing_litellm_params: Final = parse_agent_litellm_params(existing_agent.get("litellm_params"))
                update_data["litellm_params"] = safe_dumps(
                    _restore_redacted_litellm_params(
                        _dump_agent_params(agent.get("litellm_params") or _EMPTY_LITELLM_PARAMS),
                        existing_litellm_params,
                    )
                )
            if augment_agent.get("agent_card_params"):
                update_data["agent_card_params"] = safe_dumps(augment_agent.get("agent_card_params"))

            for rate_field in (
                "tpm_limit",
                "rpm_limit",
                "session_tpm_limit",
                "session_rpm_limit",
            ):
                if rate_field in agent:
                    update_data[rate_field] = agent.get(rate_field)
            if "static_headers" in agent:
                headers_value: Final = agent.get("static_headers")
                update_data["static_headers"] = safe_dumps(dict(headers_value) if headers_value is not None else {})
            if "extra_headers" in agent:
                extra_headers_value: Final = agent.get("extra_headers")
                update_data["extra_headers"] = extra_headers_value if extra_headers_value is not None else []
            if agent.get("object_permission") is not None:
                agent_copy: Final = dict(augment_agent)
                existing_object_permission_id: Final = existing_record.object_permission_id
                object_permission_id: Final = await handle_update_object_permission_common(
                    agent_copy,
                    existing_object_permission_id,
                    prisma_client,
                )
                if object_permission_id is not None:
                    update_data["object_permission_id"] = object_permission_id
            # Patch agent in DB
            patched_agent: Final = await agents_table(prisma_client).update(
                where={"agent_id": agent_id},
                data={
                    **update_data,
                    "updated_by": updated_by,
                    "updated_at": datetime.now(timezone.utc),
                },
                include={"object_permission": True},
            )
            if patched_agent is None:
                raise ValueError(f"Agent not found, passed agent_id={agent_id}")
            patched_agent_dict: Final = patched_agent.model_dump()
            if patched_agent.object_permission is not None:
                try:
                    patched_agent_dict["object_permission"] = patched_agent.object_permission.model_dump()
                except Exception:
                    patched_agent_dict["object_permission"] = patched_agent.object_permission.dict()
            return AgentResponse(**patched_agent_dict)
        except Exception as e:
            raise Exception(f"Error patching agent in DB: {e}")

    async def update_agent_in_db(
        self,
        agent_id: str,
        agent: AgentConfig,
        prisma_client: PrismaClient,
        updated_by: str,
    ) -> AgentResponse:
        """
        Update an agent in the database
        """
        try:
            agent_name: Final = agent.get("agent_name")

            # A PUT fully replaces litellm_params from the request body, so the
            # existing row is read up front to restore any sensitive key the
            # caller echoed back redacted (or omitted) rather than persisting
            # the marker -- or nothing -- over the real stored credential.
            existing_row: Final = await agents_table(prisma_client).find_unique(
                where={"agent_id": agent_id}  # mutable-ok: prisma's query builder rejects a Mapping/MappingProxyType
            )
            existing_litellm_params: Final = parse_agent_litellm_params(
                existing_row.litellm_params if existing_row is not None else None
            )

            # Serialize litellm_params
            litellm_params_obj: Final = agent.get("litellm_params", {})
            litellm_params_dict: Final = _restore_redacted_litellm_params(
                _dump_agent_params(litellm_params_obj), existing_litellm_params
            )
            litellm_params: Final[str] = safe_dumps(litellm_params_dict)

            # Serialize agent_card_params
            agent_card_params_obj: Final = agent.get("agent_card_params", {})
            agent_card_params_dict: Final[dict[str, object]] = _dump_agent_params(agent_card_params_obj)
            agent_card_params: Final[str] = safe_dumps(agent_card_params_dict)

            # Serialize static_headers for update
            static_headers_obj_u: Final = agent.get("static_headers")
            static_headers_val_u: Final[str] = (
                safe_dumps(dict(static_headers_obj_u)) if static_headers_obj_u is not None else safe_dumps({})
            )
            extra_headers_val_u: Final = agent.get("extra_headers") or []

            update_data: Final[dict[str, object]] = {
                "agent_name": agent_name,
                "litellm_params": litellm_params,
                "agent_card_params": agent_card_params,
                "static_headers": static_headers_val_u,
                "extra_headers": extra_headers_val_u,
                "updated_by": updated_by,
                "updated_at": datetime.now(timezone.utc),
            }

            for rate_field in (
                "tpm_limit",
                "rpm_limit",
                "session_tpm_limit",
                "session_rpm_limit",
            ):
                _val = agent.get(rate_field)
                if _val is not None:
                    update_data[rate_field] = _val

            if agent.get("object_permission") is not None:
                existing_object_permission_id: Final = (
                    existing_row.object_permission_id if existing_row is not None else None
                )
                agent_copy: Final = dict(agent)
                object_permission_id: Final = await handle_update_object_permission_common(
                    agent_copy,
                    existing_object_permission_id,
                    prisma_client,
                )
                if object_permission_id is not None:
                    update_data["object_permission_id"] = object_permission_id

            # Update agent in DB
            updated_agent: Final = await agents_table(prisma_client).update(
                where={"agent_id": agent_id},
                data=update_data,
                include={"object_permission": True},
            )

            if updated_agent is None:
                raise ValueError(f"Agent not found, passed agent_id={agent_id}")
            updated_agent_dict: Final = updated_agent.model_dump()
            if updated_agent.object_permission is not None:
                try:
                    updated_agent_dict["object_permission"] = updated_agent.object_permission.model_dump()
                except Exception:
                    updated_agent_dict["object_permission"] = updated_agent.object_permission.dict()
            return AgentResponse(**updated_agent_dict)
        except Exception as e:
            raise Exception(f"Error updating agent in DB: {e}")

    @staticmethod
    async def get_all_agents_from_db(
        prisma_client: PrismaClient,
    ) -> list[dict[str, object]]:
        """
        Get all agents from the database
        """
        try:
            agents_from_db: Final = await agents_table(prisma_client).find_many(
                order={"created_at": "desc"},
                include={"object_permission": True},
            )

            agents: Final[list[dict[str, object]]] = []
            for agent in agents_from_db:
                agent_dict = dict(agent)
                # object_permission is eagerly loaded via include above
                if agent.object_permission is not None:
                    try:
                        agent_dict["object_permission"] = agent.object_permission.model_dump()
                    except Exception:
                        agent_dict["object_permission"] = agent.object_permission.dict()
                agents.append(agent_dict)

            return agents
        except Exception as e:
            raise Exception(f"Error getting agents from DB: {e}")

    def get_agent_by_id(
        self,
        agent_id: str,
    ) -> AgentResponse | None:
        """
        Get an agent by its ID from the database
        """
        try:
            for agent in self.agent_list:
                if agent.agent_id == agent_id:
                    return agent

            translated_id: Final = self.config_agent_legacy_ids.get(agent_id)
            if translated_id is None:
                return None

            for agent in self.agent_list:
                if agent.agent_id == translated_id:
                    return agent

            return None
        except Exception as e:
            raise Exception(f"Error getting agent from DB: {e}")

    def get_agent_by_name(self, agent_name: str) -> AgentResponse | None:
        """
        Get an agent by its name from the database
        """
        try:
            for agent in self.agent_list:
                if agent.agent_name == agent_name:
                    return agent

            return None
        except Exception as e:
            raise Exception(f"Error getting agent from DB: {e}")


global_agent_registry: Final = AgentRegistry()
AGENT_RECONCILE_LOCK: Final = asyncio.Lock()
