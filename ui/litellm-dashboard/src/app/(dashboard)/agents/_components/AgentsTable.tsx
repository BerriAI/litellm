"use client";

import { SortingState } from "@tanstack/react-table";
import { Bot, CircleCheck } from "lucide-react";
import React, { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import { Agent } from "@/components/agents/types";
import { DataTable } from "@/components/shared/DataTable";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

import { getAgentsTableColumns } from "./AgentsTableColumns";

interface AgentsTableProps {
  agents: Agent[];
  isLoading: boolean;
  isAdmin: boolean;
  healthCheckEnabled: boolean;
  isHealthCheckLoading: boolean;
  onHealthCheckToggle: (checked: boolean) => void;
  onAgentClick: (agentId: string) => void;
  onDeleteClick: (agentId: string, agentName: string) => void;
}

const DEFAULT_SORTING: SortingState = [{ id: "created_at", desc: true }];

function EmptyState({ title, description }: { title: string; description: string }) {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Bot className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">{title}</div>
      <div className="text-sm text-muted-foreground">{description}</div>
    </div>
  );
}

const AgentsTable: React.FC<AgentsTableProps> = ({
  agents,
  isLoading,
  isAdmin,
  healthCheckEnabled,
  isHealthCheckLoading,
  onHealthCheckToggle,
  onAgentClick,
  onDeleteClick,
}) => {
  const { t } = useTranslation("gateway");
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);

  const columns = useMemo(
    () => getAgentsTableColumns({ isAdmin, onAgentClick, onDeleteClick, t }),
    [isAdmin, onAgentClick, onDeleteClick, t],
  );

  return (
    <DataTable
      data={agents}
      columns={columns}
      getRowId={(agent, index) => agent.agent_id || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      isLoading={isLoading}
      loadingMessage={t("agents.loading")}
      noDataMessage={<EmptyState title={t("agents.emptyTitle")} description={t("agents.emptyDescription")} />}
      size="compact"
      toolbar={() => (
        <div className="flex items-center justify-end">
          <TooltipProvider delay={300}>
            <Tooltip>
              <TooltipTrigger
                render={
                  <div className="flex items-center gap-2">
                    <CircleCheck
                      className={healthCheckEnabled ? "size-4 text-green-500" : "size-4 text-muted-foreground"}
                    />
                    <span className="text-sm text-muted-foreground">{t("agents.healthCheck")}</span>
                    <Switch
                      size="sm"
                      checked={healthCheckEnabled}
                      onCheckedChange={onHealthCheckToggle}
                      disabled={isHealthCheckLoading}
                    />
                  </div>
                }
              />
              <TooltipContent>{t("agents.healthCheckHint")}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      )}
    />
  );
};

export default AgentsTable;
