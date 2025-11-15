from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy.utils import PrismaClient
from litellm.types.agents import AgentConfig


class AgentRegistry:
    def __init__(self):
        self.agent_list: List[AgentConfig] = []

    def reset_agent_list(self):
        self.agent_list = []

    def register_agent(self, agent_config: AgentConfig):
        self.agent_list.append(agent_config)

    def deregister_agent(self, agent_name: str):
        self.agent_list = [
            agent for agent in self.agent_list if agent.get("agent_name") != agent_name
        ]

    def get_agent_list(self, agent_names: Optional[List[str]] = None):
        if agent_names is not None:
            return [
                agent
                for agent in self.agent_list
                if agent.get("agent_name") in agent_names
            ]
        return self.agent_list

    def get_public_agent_list(self):
        public_agent_list = []
        for agent in self.agent_list:
            if agent.get("litellm_params", {}).get("make_public", False) is True:
                public_agent_list.append(agent)
        return public_agent_list

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

            self.register_agent(agent_config=agent_config_item)

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

                self.register_agent(agent_config=agent_config_item)

        if db_agents:
            for db_agent in db_agents:
                if not isinstance(db_agent, dict):
                    raise ValueError("db_agents must be a list of dictionaries")

                self.register_agent(agent_config=AgentConfig(**db_agent))  # type: ignore
        return self.agent_list

    ###########################################################
    ########### DB management helpers for agents ###########
    ############################################################
    async def add_agent_to_db(
        self, agent: AgentConfig, prisma_client: PrismaClient, created_by: str
    ) -> Dict[str, Any]:
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

            return dict(created_agent)
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

    async def update_agent_in_db(
        self,
        agent_id: str,
        agent: AgentConfig,
        prisma_client: PrismaClient,
        updated_by: str,
    ) -> Dict[str, Any]:
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

            return dict(updated_agent)
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
    ) -> Optional[Dict[str, Any]]:
        """
        Get an agent by its ID from the database
        """
        try:
            for agent in self.agent_list:
                if agent.get("agent_id") == agent_id:
                    return dict(agent)

            return None
        except Exception as e:
            raise Exception(f"Error getting agent from DB: {str(e)}")

    def get_agent_by_name(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Get an agent by its name from the database
        """
        try:
            for agent in self.agent_list:
                if agent.get("agent_name") == agent_name:
                    return dict(agent)

            return None
        except Exception as e:
            raise Exception(f"Error getting agent from DB: {str(e)}")


global_agent_registry = AgentRegistry()
