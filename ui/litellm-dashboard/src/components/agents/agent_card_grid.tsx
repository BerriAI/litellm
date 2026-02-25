import React from "react";
import { Skeleton } from "antd";
import AgentCard from "./agent_card";
import { Agent, AgentKeyInfo } from "./types";

interface AgentCardGridProps {
  agentsList: Agent[];
  keyInfoMap: Record<string, AgentKeyInfo>;
  isLoading: boolean;
  onDeleteClick: (agentId: string, agentName: string) => void;
  accessToken: string | null;
  onAgentUpdated: () => void;
  isAdmin: boolean;
  onAgentClick: (agentId: string) => void;
}

const AgentCardGrid: React.FC<AgentCardGridProps> = ({
  agentsList,
  keyInfoMap,
  isLoading,
  onDeleteClick,
  accessToken,
  onAgentUpdated,
  isAdmin,
  onAgentClick,
}) => {
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[1, 2, 3].map((i) => (
          <Skeleton key={i} active paragraph={{ rows: 3 }} />
        ))}
      </div>
    );
  }

  if (!agentsList || agentsList.length === 0) {
    return (
      <div className="rounded-lg border border-gray-200 bg-gray-50/50 py-12 text-center">
        <p className="text-gray-500">No agents found. Create one to get started.</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {agentsList.map((agent) => (
        <AgentCard
          key={agent.agent_id}
          agent={agent}
          keyInfo={keyInfoMap[agent.agent_id]}
          onAgentClick={onAgentClick}
          onDeleteClick={isAdmin ? onDeleteClick : undefined}
          accessToken={accessToken}
          isAdmin={isAdmin}
          onAgentUpdated={onAgentUpdated}
        />
      ))}
    </div>
  );
};

export default AgentCardGrid;
