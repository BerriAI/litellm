import React from "react";
import { Skeleton } from "@/components/ui/skeleton";
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
          <div key={i} className="border border-border rounded-lg p-4 space-y-3">
            <Skeleton className="h-5 w-1/2" />
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-5/6" />
            <Skeleton className="h-3 w-3/4" />
          </div>
        ))}
      </div>
    );
  }

  if (!agentsList || agentsList.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-muted/50 py-12 text-center">
        <p className="text-muted-foreground">
          {isAdmin
            ? "No agents found. Create one to get started."
            : "No agents found. Contact an admin to create agents."}
        </p>
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
