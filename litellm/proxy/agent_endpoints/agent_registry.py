from typing import Any, Dict, List, Optional

from litellm.types.agents import AgentCard, AgentConfig


class AgentRegistry:
    def __init__(self):
        self.agent_list: List[AgentConfig] = []

    def register_agent(self, agent_config: AgentConfig):
        self.agent_list.append(agent_config)

    def get_agent_list(self, agent_names: Optional[List[str]] = None):
        if agent_names is not None:
            return [
                agent
                for agent in self.agent_list
                if agent.get("agent_name") in agent_names
            ]
        return self.agent_list

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


global_agent_registry = AgentRegistry()
