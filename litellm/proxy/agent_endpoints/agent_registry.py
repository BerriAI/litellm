import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Protocol, TypedDict

import litellm
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy.management_helpers.object_permission_utils import (
    handle_update_object_permission_common,
)
from litellm.proxy.utils import PrismaClient
from litellm.repositories.table_repositories import AgentsRepository
from litellm.types.agents import AgentConfig, AgentResponse, PatchAgentRequest


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
    agent_id: str
    agent_name: str
    object_permission_id: str | None
    object_permission: AgentObjectPermissionRecord | None
    spend: float

    def model_dump(self) -> AgentRecordDump: ...

    def __iter__(self) -> Iterator[tuple[str, object]]: ...


class AgentTableClient(Protocol):
    async def create(
        self,
        data: Mapping[str, object],
        include: Mapping[str, bool] | None = None,
    ) -> AgentRecord: ...

    async def find_unique(
        self,
        where: Mapping[str, object],
        include: Mapping[str, bool] | None = None,
    ) -> AgentRecord | None: ...

    async def find_many(
        self,
        where: Mapping[str, object] | None = None,
        order: Mapping[str, str] | None = None,
        include: Mapping[str, bool] | None = None,
    ) -> Sequence[AgentRecord]: ...

    async def update(
        self,
        where: Mapping[str, object],
        data: Mapping[str, object],
        include: Mapping[str, bool] | None = None,
    ) -> AgentRecord: ...

    async def delete(self, where: Mapping[str, object]) -> AgentRecord: ...


def agents_table(prisma_client: PrismaClient) -> AgentTableClient:
    table: AgentTableClient = AgentsRepository(prisma_client).table
    return table


class AgentRegistry:
    def __init__(self):
        self.agent_list: list[AgentResponse] = []

    def reset_agent_list(self):
        self.agent_list = []

    def register_agent(self, agent_config: AgentResponse):
        self.agent_list.append(agent_config)

    def deregister_agent(self, agent_name: str):
        self.agent_list = [agent for agent in self.agent_list if agent.agent_name != agent_name]

    def get_agent_list(self, agent_names: Sequence[str] | None = None):
        if agent_names is not None:
            return [agent for agent in self.agent_list if agent.agent_name in agent_names]
        return self.agent_list

    def get_public_agent_list(self) -> list[AgentResponse]:
        public_agent_list: list[AgentResponse] = []
        if litellm.public_agent_groups is None:
            return public_agent_list
        for agent in self.agent_list:
            if agent.agent_id in litellm.public_agent_groups:
                public_agent_list.append(agent)
        return public_agent_list

    def _create_agent_id(self, agent_config: AgentConfig) -> str:
        return hashlib.sha256(json.dumps(agent_config, sort_keys=True).encode()).hexdigest()

    def load_agents_from_config(self, agent_config: Sequence[AgentConfig] | None = None):
        if agent_config is None:
            return None

        for agent_config_item in agent_config:
            if not isinstance(agent_config_item, dict):
                raise ValueError("agent_config must be a list of dictionaries")

            agent_name = agent_config_item.get("agent_name")
            agent_card_params = agent_config_item.get("agent_card_params")
            if not all([agent_name, agent_card_params]):
                continue

            # create a stable hash id for config item
            config_hash = self._create_agent_id(agent_config_item)

            self.register_agent(agent_config=AgentResponse(agent_id=config_hash, **agent_config_item))  # type: ignore

    def load_agents_from_db_and_config(
        self,
        agent_config: Sequence[AgentConfig] | None = None,
        db_agents: list[dict[str, Any]] | None = None,
    ):
        self.reset_agent_list()

        if agent_config:
            for agent_config_item in agent_config:
                if not isinstance(agent_config_item, dict):
                    raise ValueError("agent_config must be a list of dictionaries")

                self.register_agent(
                    agent_config=AgentResponse(
                        agent_id=self._create_agent_id(agent_config_item),
                        **agent_config_item,
                    )
                )  # type: ignore

        if db_agents:
            for db_agent in db_agents:
                if not isinstance(db_agent, dict):
                    raise ValueError("db_agents must be a list of dictionaries")

                self.register_agent(agent_config=AgentResponse(**db_agent))  # type: ignore
        return self.agent_list

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
            agent_name = agent.get("agent_name")

            # Serialize litellm_params
            litellm_params_obj: Any = agent.get("litellm_params", {})
            if hasattr(litellm_params_obj, "model_dump"):
                litellm_params_dict = litellm_params_obj.model_dump()
            else:
                litellm_params_dict = dict(litellm_params_obj) if litellm_params_obj else {}
            litellm_params: str = safe_dumps(litellm_params_dict)

            # Serialize agent_card_params
            agent_card_params_obj: Any = agent.get("agent_card_params", {})
            if hasattr(agent_card_params_obj, "model_dump"):
                agent_card_params_dict = agent_card_params_obj.model_dump()
            else:
                agent_card_params_dict = dict(agent_card_params_obj) if agent_card_params_obj else {}
            agent_card_params: str = safe_dumps(agent_card_params_dict)

            # Handle object_permission (MCP tool access for agent)
            object_permission_id: str | None = None
            if agent.get("object_permission") is not None:
                agent_copy = dict(agent)
                object_permission_id = await handle_update_object_permission_common(agent_copy, None, prisma_client)

            # Serialize static_headers
            static_headers_obj = agent.get("static_headers")
            static_headers_val: str | None = safe_dumps(dict(static_headers_obj)) if static_headers_obj else None

            extra_headers_val = agent.get("extra_headers")

            create_data: dict[str, object] = {
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
            created_agent = await agents_table(prisma_client).create(
                data=create_data,
                include={"object_permission": True},
            )

            created_agent_dict = created_agent.model_dump()
            if created_agent.object_permission is not None:
                try:
                    created_agent_dict["object_permission"] = created_agent.object_permission.model_dump()
                except Exception:
                    created_agent_dict["object_permission"] = created_agent.object_permission.dict()
            return AgentResponse(**created_agent_dict)  # type: ignore
        except Exception as e:
            raise Exception(f"Error adding agent to DB: {str(e)}")

    async def delete_agent_from_db(self, agent_id: str, prisma_client: PrismaClient) -> Mapping[str, object]:
        """
        Delete an agent from the database
        """
        try:
            deleted_agent = await agents_table(prisma_client).delete(where={"agent_id": agent_id})
            return dict(deleted_agent)
        except Exception as e:
            raise Exception(f"Error deleting agent from DB: {str(e)}")

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
            existing_agent = await AgentsRepository(prisma_client).table.find_unique(where={"agent_id": agent_id})
            if existing_agent is not None:
                existing_agent = dict(existing_agent)

            if existing_agent is None:
                raise Exception(f"Agent with ID {agent_id} not found")

            augment_agent = {**existing_agent, **agent}
            update_data: dict[str, Any] = {}
            if augment_agent.get("agent_name"):
                update_data["agent_name"] = augment_agent.get("agent_name")
            if augment_agent.get("litellm_params"):
                update_data["litellm_params"] = safe_dumps(augment_agent.get("litellm_params"))
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
                headers_value = agent.get("static_headers")
                update_data["static_headers"] = safe_dumps(dict(headers_value) if headers_value is not None else {})
            if "extra_headers" in agent:
                extra_headers_value = agent.get("extra_headers")
                update_data["extra_headers"] = extra_headers_value if extra_headers_value is not None else []
            if agent.get("object_permission") is not None:
                agent_copy = dict(augment_agent)
                existing_object_permission_id = existing_agent.get("object_permission_id")
                object_permission_id = await handle_update_object_permission_common(
                    agent_copy,
                    existing_object_permission_id,
                    prisma_client,
                )
                if object_permission_id is not None:
                    update_data["object_permission_id"] = object_permission_id
            # Patch agent in DB
            patched_agent = await agents_table(prisma_client).update(
                where={"agent_id": agent_id},
                data={
                    **update_data,
                    "updated_by": updated_by,
                    "updated_at": datetime.now(timezone.utc),
                },
                include={"object_permission": True},
            )
            patched_agent_dict = patched_agent.model_dump()
            if patched_agent.object_permission is not None:
                try:
                    patched_agent_dict["object_permission"] = patched_agent.object_permission.model_dump()
                except Exception:
                    patched_agent_dict["object_permission"] = patched_agent.object_permission.dict()
            return AgentResponse(**patched_agent_dict)  # type: ignore
        except Exception as e:
            raise Exception(f"Error patching agent in DB: {str(e)}")

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
            agent_name = agent.get("agent_name")

            # Serialize litellm_params
            litellm_params_obj: Any = agent.get("litellm_params", {})
            if hasattr(litellm_params_obj, "model_dump"):
                litellm_params_dict = litellm_params_obj.model_dump()
            else:
                litellm_params_dict = dict(litellm_params_obj) if litellm_params_obj else {}
            litellm_params: str = safe_dumps(litellm_params_dict)

            # Serialize agent_card_params
            agent_card_params_obj: Any = agent.get("agent_card_params", {})
            if hasattr(agent_card_params_obj, "model_dump"):
                agent_card_params_dict = agent_card_params_obj.model_dump()
            else:
                agent_card_params_dict = dict(agent_card_params_obj) if agent_card_params_obj else {}
            agent_card_params: str = safe_dumps(agent_card_params_dict)

            # Serialize static_headers for update
            static_headers_obj_u = agent.get("static_headers")
            static_headers_val_u: str = (
                safe_dumps(dict(static_headers_obj_u)) if static_headers_obj_u is not None else safe_dumps({})
            )
            extra_headers_val_u = agent.get("extra_headers") or []

            update_data: dict[str, object] = {
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
                existing_agent = await agents_table(prisma_client).find_unique(where={"agent_id": agent_id})
                existing_object_permission_id = (
                    existing_agent.object_permission_id if existing_agent is not None else None
                )
                agent_copy = dict(agent)
                object_permission_id = await handle_update_object_permission_common(
                    agent_copy,
                    existing_object_permission_id,
                    prisma_client,
                )
                if object_permission_id is not None:
                    update_data["object_permission_id"] = object_permission_id

            # Update agent in DB
            updated_agent = await agents_table(prisma_client).update(
                where={"agent_id": agent_id},
                data=update_data,
                include={"object_permission": True},
            )

            updated_agent_dict = updated_agent.model_dump()
            if updated_agent.object_permission is not None:
                try:
                    updated_agent_dict["object_permission"] = updated_agent.object_permission.model_dump()
                except Exception:
                    updated_agent_dict["object_permission"] = updated_agent.object_permission.dict()
            return AgentResponse(**updated_agent_dict)  # type: ignore
        except Exception as e:
            raise Exception(f"Error updating agent in DB: {str(e)}")

    @staticmethod
    async def get_all_agents_from_db(
        prisma_client: PrismaClient,
    ) -> list[dict[str, object]]:
        """
        Get all agents from the database
        """
        try:
            agents_from_db = await agents_table(prisma_client).find_many(
                order={"created_at": "desc"},
                include={"object_permission": True},
            )

            agents: list[dict[str, object]] = []
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
            raise Exception(f"Error getting agents from DB: {str(e)}")

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

            return None
        except Exception as e:
            raise Exception(f"Error getting agent from DB: {str(e)}")

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
            raise Exception(f"Error getting agent from DB: {str(e)}")


global_agent_registry = AgentRegistry()
