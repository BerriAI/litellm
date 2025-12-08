import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import litellm
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy.utils import PrismaClient
from litellm.types.agents import AgentConfig, AgentResponse, PatchAgentRequest


class AgentRegistry:
    def __init__(self):
        self.agent_list: List[AgentResponse] = []

    def reset_agent_list(self):
        self.agent_list = []

    def register_agent(self, agent_config: AgentResponse):
        self.agent_list.append(agent_config)

    def deregister_agent(self, agent_name: str):
        self.agent_list = [
            agent for agent in self.agent_list if agent.agent_name != agent_name
        ]

    def get_agent_list(self, agent_names: Optional[List[str]] = None):
        if agent_names is not None:
            return [
                agent for agent in self.agent_list if agent.agent_name in agent_names
            ]
        return self.agent_list

    def get_public_agent_list(self) -> List[AgentResponse]:
        public_agent_list: List[AgentResponse] = []
        if litellm.public_agent_groups is None:
            return public_agent_list
        for agent in self.agent_list:
            if agent.agent_id in litellm.public_agent_groups:
                public_agent_list.append(agent)
        return public_agent_list

    def _create_agent_id(self, agent_config: AgentConfig) -> str:
        return hashlib.sha256(
            json.dumps(agent_config, sort_keys=True).encode()
        ).hexdigest()

    def load_agents_from_config(self, agent_config: Optional[List[AgentConfig]] = None):
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
        agent_config: Optional[List[AgentConfig]] = None,
        db_agents: Optional[List[Dict[str, Any]]] = None,
    ):
        self.reset_agent_list()

        if agent_config:
            for agent_config_item in agent_config:
                if not isinstance(agent_config_item, dict):
                    raise ValueError("agent_config must be a list of dictionaries")

                self.register_agent(agent_config=AgentResponse(agent_id=self._create_agent_id(agent_config_item), **agent_config_item))  # type: ignore

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
        self, agent: AgentConfig, prisma_client: PrismaClient, created_by: str
    ) -> AgentResponse:
        """
        Add an agent to the database
        """
        try:
            agent_name = agent.get("agent_name")

            # Serialize litellm_params
            litellm_params_obj: Any = agent.get("litellm_params", {})
            if hasattr(litellm_params_obj, "model_dump"):
                litellm_params_dict = litellm_params_obj.model_dump()
            else:
                litellm_params_dict = (
                    dict(litellm_params_obj) if litellm_params_obj else {}
                )
            litellm_params: str = safe_dumps(litellm_params_dict)

            # Serialize agent_card_params
            agent_card_params_obj: Any = agent.get("agent_card_params", {})
            if hasattr(agent_card_params_obj, "model_dump"):
                agent_card_params_dict = agent_card_params_obj.model_dump()
            else:
                agent_card_params_dict = (
                    dict(agent_card_params_obj) if agent_card_params_obj else {}
                )
            agent_card_params: str = safe_dumps(agent_card_params_dict)

            # Create agent in DB
            created_agent = await prisma_client.db.litellm_agentstable.create(
                data={
                    "agent_name": agent_name,
                    "litellm_params": litellm_params,
                    "agent_card_params": agent_card_params,
                    "created_by": created_by,
                    "updated_by": created_by,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
            )

            return AgentResponse(**created_agent.model_dump())  # type: ignore
        except Exception as e:
            raise Exception(f"Error adding agent to DB: {str(e)}")

    async def delete_agent_from_db(
        self, agent_id: str, prisma_client: PrismaClient
    ) -> Dict[str, Any]:
        """
        Delete an agent from the database
        """
        try:
            deleted_agent = await prisma_client.db.litellm_agentstable.delete(
                where={"agent_id": agent_id}
            )
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

            existing_agent = await prisma_client.db.litellm_agentstable.find_unique(
                where={"agent_id": agent_id}
            )
            if existing_agent is not None:
                existing_agent = dict(existing_agent)

            if existing_agent is None:
                raise Exception(f"Agent with ID {agent_id} not found")

            augment_agent = {**existing_agent, **agent}
            update_data = {}
            if augment_agent.get("agent_name"):
                update_data["agent_name"] = augment_agent.get("agent_name")
            if augment_agent.get("litellm_params"):
                update_data["litellm_params"] = safe_dumps(
                    augment_agent.get("litellm_params")
                )
            if augment_agent.get("agent_card_params"):
                update_data["agent_card_params"] = safe_dumps(
                    augment_agent.get("agent_card_params")
                )
            # Patch agent in DB
            patched_agent = await prisma_client.db.litellm_agentstable.update(
                where={"agent_id": agent_id},
                data={
                    **update_data,
                    "updated_by": updated_by,
                    "updated_at": datetime.now(timezone.utc),
                },
            )
            return AgentResponse(**patched_agent.model_dump())  # type: ignore
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
                litellm_params_dict = (
                    dict(litellm_params_obj) if litellm_params_obj else {}
                )
            litellm_params: str = safe_dumps(litellm_params_dict)

            # Serialize agent_card_params
            agent_card_params_obj: Any = agent.get("agent_card_params", {})
            if hasattr(agent_card_params_obj, "model_dump"):
                agent_card_params_dict = agent_card_params_obj.model_dump()
            else:
                agent_card_params_dict = (
                    dict(agent_card_params_obj) if agent_card_params_obj else {}
                )
            agent_card_params: str = safe_dumps(agent_card_params_dict)

            # Update agent in DB
            updated_agent = await prisma_client.db.litellm_agentstable.update(
                where={"agent_id": agent_id},
                data={
                    "agent_name": agent_name,
                    "litellm_params": litellm_params,
                    "agent_card_params": agent_card_params,
                    "updated_by": updated_by,
                    "updated_at": datetime.now(timezone.utc),
                },
            )

            return AgentResponse(**updated_agent.model_dump())  # type: ignore
        except Exception as e:
            raise Exception(f"Error updating agent in DB: {str(e)}")

    @staticmethod
    async def get_all_agents_from_db(
        prisma_client: PrismaClient,
    ) -> List[Dict[str, Any]]:
        """
        Get all agents from the database
        """
        try:
            agents_from_db = await prisma_client.db.litellm_agentstable.find_many(
                order={"created_at": "desc"},
            )

            agents: List[Dict[str, Any]] = []
            for agent in agents_from_db:
                agents.append(dict(agent))

            return agents
        except Exception as e:
            raise Exception(f"Error getting agents from DB: {str(e)}")

    def get_agent_by_id(
        self,
        agent_id: str,
    ) -> Optional[AgentResponse]:
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

    def get_agent_by_name(self, agent_name: str) -> Optional[AgentResponse]:
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
