import React, { useEffect, useState } from "react";
import { Select } from "antd";
import { getAgentsList } from "../networking";

interface Agent {
  agent_id: string;
  agent_name: string;
  agent_config?: Record<string, any>;
  agent_card_params?: Record<string, any>;
}

interface AgentSelectorProps {
  onChange: (selected: { 
    agents: string[]; 
    accessGroups: string[];
  }) => void;
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
  const options = [
    ...accessGroups.map((group) => ({
      label: group,
      value: `group:${group}`,
      isAccessGroup: true,
      searchText: `${group} Access Group`,
    })),
    ...agents.map((agent) => ({
      label: `${agent.agent_name || agent.agent_id}`,
      value: agent.agent_id,
      isAccessGroup: false,
      searchText: `${agent.agent_name || agent.agent_id} ${agent.agent_id} Agent`,
    })),
  ];

  // Flatten value for Select
  const selectedValues = [
    ...(value?.agents || []),
    ...(value?.accessGroups || []).map((g) => `group:${g}`),
  ];

  // Handle selection
  const handleChange = (selected: string[]) => {
    const agentsSelected = selected.filter((v) => !v.startsWith("group:"));
    const accessGroupsSelected = selected
      .filter((v) => v.startsWith("group:"))
      .map((v) => v.replace("group:", ""));
    onChange({ agents: agentsSelected, accessGroups: accessGroupsSelected });
  };

  return (
    <div>
      <Select
        mode="multiple"
        placeholder={placeholder}
        onChange={handleChange}
        value={selectedValues}
        loading={loading}
        className={className}
        allowClear
        showSearch
        style={{ width: "100%" }}
        disabled={disabled}
        filterOption={(input, option) => {
          const searchText = options.find((opt) => opt.value === option?.value)?.searchText || "";
          return searchText.toLowerCase().includes(input.toLowerCase());
        }}
      >
        {options.map((opt) => (
          <Select.Option key={opt.value} value={opt.value} label={opt.label}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span
                style={{
                  display: "inline-block",
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: opt.isAccessGroup ? "#52c41a" : "#722ed1",
                  flexShrink: 0,
                }}
              />
              <span style={{ flex: 1 }}>{opt.label}</span>
              <span
                style={{
                  color: opt.isAccessGroup ? "#52c41a" : "#722ed1",
                  fontSize: "12px",
                  fontWeight: 500,
                  opacity: 0.8,
                }}
              >
                {opt.isAccessGroup ? "Access Group" : "Agent"}
              </span>
            </div>
          </Select.Option>
        ))}
      </Select>
    </div>
  );
};

export default AgentSelector;

