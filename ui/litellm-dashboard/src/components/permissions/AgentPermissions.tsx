import React, { useState, useEffect } from "react";
import { UserGroupIcon } from "@heroicons/react/outline";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
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

export function AgentPermissions({ agents, agentAccessGroups = [], accessToken }: AgentPermissionsProps) {
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
        <p className="text-sm font-semibold text-foreground">Agents</p>
        <Badge variant="secondary">{totalCount}</Badge>
      </div>

      {totalCount > 0 ? (
        <div className="max-h-[400px] overflow-y-auto space-y-2 pr-1">
          {mergedItems.map((item, index) => (
            <div key={index} className="space-y-2">
              <div className="flex items-center gap-3 py-2 px-3 rounded-lg border border-border bg-card">
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  {item.type === "agent" ? (
                    <TooltipProvider delay={300}>
                      <Tooltip>
                        <TooltipTrigger render={<div className="inline-flex items-center gap-2 min-w-0" />}>
                          <span className="inline-block w-1.5 h-1.5 bg-purple-500 rounded-full shrink-0"></span>
                          <span className="text-sm font-medium text-foreground truncate">
                            {getAgentDisplayName(item.value)}
                          </span>
                        </TooltipTrigger>
                        <TooltipContent>{`Full ID: ${item.value}`}</TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  ) : (
                    <div className="inline-flex items-center gap-2 min-w-0">
                      <span className="inline-block w-1.5 h-1.5 bg-success rounded-full shrink-0"></span>
                      <span className="text-sm font-medium text-foreground truncate">{item.value}</span>
                      <span className="ml-1 px-1.5 py-0.5 text-[9px] font-semibold text-success bg-success/10 border border-success/20 rounded-sm uppercase tracking-wide shrink-0">
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
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-muted border border-border">
          <UserGroupIcon className="h-4 w-4 text-muted-foreground" />
          <p className="text-muted-foreground text-sm">No agents or access groups configured</p>
        </div>
      )}
    </div>
  );
}

export default AgentPermissions;
