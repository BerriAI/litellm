import React, { useEffect, useState } from "react";
import { MultiSelect, type MultiSelectOption } from "@/components/shared/MultiSelect";
import { getAgentsList } from "../networking";

interface Agent {
  agent_id: string;
  agent_name: string;
  agent_config?: Record<string, any>;
  agent_card_params?: Record<string, any>;
}

interface AgentSelectorProps {
  onChange: (selected: { agents: string[]; accessGroups: string[] }) => void;
  value?: {
    agents: string[];
    accessGroups: string[];
  };
  className?: string;
  accessToken: string;
  placeholder?: string;
  disabled?: boolean;
}

const AgentSelector: React.FC<AgentSelectorProps> = ({
  onChange,
  value,
  className,
  accessToken,
  placeholder = "Select agents",
  disabled = false,
}) => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [accessGroups, setAccessGroups] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      if (!accessToken) return;
      setLoading(true);
      try {
        const response = await getAgentsList(accessToken);
        let agentsList = response?.agents || [];
        setAgents(agentsList);

        // Extract unique access groups from agents
        const groups = new Set<string>();
        agentsList.forEach((agent: Agent) => {
          const agentAccessGroups = (agent as any).agent_access_groups;
          if (agentAccessGroups && Array.isArray(agentAccessGroups)) {
            agentAccessGroups.forEach((g: string) => groups.add(g));
          }
        });
        setAccessGroups(Array.from(groups));
      } catch (error) {
        console.error("Error fetching agents:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [accessToken]);

  // Combine options, access groups first
  const options: MultiSelectOption[] = [
    ...accessGroups.map((group) => ({
      label: group,
      value: `group:${group}`,
      description: "Access Group",
    })),
    ...agents.map((agent) => ({
      label: `${agent.agent_name || agent.agent_id}`,
      value: agent.agent_id,
      description: "Agent",
    })),
  ];

  // Flatten value for Select
  const selectedValues = [...(value?.agents || []), ...(value?.accessGroups || []).map((g) => `group:${g}`)];

  // Handle selection
  const handleChange = (selected: string[]) => {
    const agentsSelected = selected.filter((v) => !v.startsWith("group:"));
    const accessGroupsSelected = selected.filter((v) => v.startsWith("group:")).map((v) => v.replace("group:", ""));
    onChange({ agents: agentsSelected, accessGroups: accessGroupsSelected });
  };

  return (
    <div>
      <MultiSelect
        options={options}
        value={selectedValues}
        onValueChange={handleChange}
        placeholder={placeholder}
        emptyText="No agents found"
        loading={loading}
        disabled={disabled}
        className={`w-full ${className ?? ""}`}
      />
    </div>
  );
};

export default AgentSelector;
