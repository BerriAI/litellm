import React, { useState, useEffect } from "react";
import { Text, Badge } from "@tremor/react";
import { UserGroupIcon } from "@heroicons/react/outline";
import { Tooltip } from "antd";
import { getAgentsList } from "../networking";

interface Agent {
  agent_id: string;
  agent_name: string;
  agent_config?: Record<string, any>;
  agent_card_params?: Record<string, any>;
}

interface AgentPermissionsProps {
  agents: string[];
  agentAccessGroups?: string[];
  accessToken?: string | null;
}

export function AgentPermissions({ 
  agents, 
  agentAccessGroups = [], 
  accessToken 
}: AgentPermissionsProps) {
  const [agentDetails, setAgentDetails] = useState<Agent[]>([]);

  // Fetch agent details when component mounts
  useEffect(() => {
    const fetchAgentDetails = async () => {
      if (accessToken && agents.length > 0) {
        try {
          const response = await getAgentsList(accessToken);
          if (response && response.agents && Array.isArray(response.agents)) {
            setAgentDetails(response.agents);
          }
        } catch (error) {
          console.error("Error fetching agents:", error);
        }
      }
    };
    fetchAgentDetails();
  }, [accessToken, agents.length]);

  // Function to get display name for agent
  const getAgentDisplayName = (agentId: string) => {
    const agentDetail = agentDetails.find((agent) => agent.agent_id === agentId);
    if (agentDetail) {
      const truncatedId = agentId.length > 7 ? `${agentId.slice(0, 3)}...${agentId.slice(-4)}` : agentId;
      return `${agentDetail.agent_name} (${truncatedId})`;
    }
    return agentId;
  };

  // Merge agents and access groups into one list
  const mergedItems = [
    ...agents.map((agent) => ({ type: "agent", value: agent })),
    ...agentAccessGroups.map((group) => ({ type: "accessGroup", value: group })),
  ];
  const totalCount = mergedItems.length;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <UserGroupIcon className="h-4 w-4 text-purple-600" />
        <Text className="font-semibold text-gray-900">Agents</Text>
        <Badge color="purple" size="xs">
          {totalCount}
        </Badge>
      </div>
      
      {totalCount > 0 ? (
        <div className="max-h-[400px] overflow-y-auto space-y-2 pr-1">
          {mergedItems.map((item, index) => (
            <div key={index} className="space-y-2">
              <div 
                className="flex items-center gap-3 py-2 px-3 rounded-lg border border-gray-200 bg-white"
              >
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  {item.type === "agent" ? (
                    <Tooltip title={`Full ID: ${item.value}`} placement="top">
                      <div className="inline-flex items-center gap-2 min-w-0">
                        <span className="inline-block w-1.5 h-1.5 bg-purple-500 rounded-full flex-shrink-0"></span>
                        <span className="text-sm font-medium text-gray-900 truncate">{getAgentDisplayName(item.value)}</span>
                      </div>
                    </Tooltip>
                  ) : (
                    <div className="inline-flex items-center gap-2 min-w-0">
                      <span className="inline-block w-1.5 h-1.5 bg-green-500 rounded-full flex-shrink-0"></span>
                      <span className="text-sm font-medium text-gray-900 truncate">{item.value}</span>
                      <span className="ml-1 px-1.5 py-0.5 text-[9px] font-semibold text-green-600 bg-green-50 border border-green-200 rounded uppercase tracking-wide flex-shrink-0">
                        Group
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-50 border border-gray-200">
          <UserGroupIcon className="h-4 w-4 text-gray-400" />
          <Text className="text-gray-500 text-sm">No agents or access groups configured</Text>
        </div>
      )}
    </div>
  );
}

export default AgentPermissions;

