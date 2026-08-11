"use client";

import { ColumnDef } from "@tanstack/react-table";
import { MoreHorizontal, Trash2 } from "lucide-react";

import { Agent } from "@/components/agents/types";
import { DataTableSortHeader } from "@/components/shared/DataTable";
import { DateCell, IdentityCell, MoneyCell, StatusBadge } from "@/components/shared/table_cells";
import { Badge } from "@/components/ui/badge";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";
import type { TFunction } from "i18next";

interface AgentRowActionsProps {
  agent: Agent;
  onDeleteClick: (agentId: string, agentName: string) => void;
  t: TFunction<"gateway">;
}

function AgentRowActions({ agent, onDeleteClick, t }: AgentRowActionsProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t("agents.actions.open")}
        data-testid={`agent-actions-${agent.agent_id}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-44">
        <DropdownMenuItem
          variant="destructive"
          data-testid="agent-action-delete"
          onClick={() => onDeleteClick(agent.agent_id, agent.agent_name)}
        >
          <Trash2 />
          {t("agents.actions.delete")}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface AgentsTableColumnsDeps {
  isAdmin: boolean;
  onAgentClick: (agentId: string) => void;
  onDeleteClick: (agentId: string, agentName: string) => void;
  t: TFunction<"gateway">;
}

export const getAgentsTableColumns = ({
  isAdmin,
  onAgentClick,
  onDeleteClick,
  t,
}: AgentsTableColumnsDeps): ColumnDef<Agent>[] => [
  {
    id: "agent_name",
    accessorKey: "agent_name",
    meta: { title: t("agents.columns.name") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("agents.columns.name")} />,
    size: 200,
    enableSorting: true,
    cell: ({ row }) => {
      const name = row.original.agent_name;
      return (
        <span className="block max-w-52 truncate text-sm font-medium text-foreground" title={name || undefined}>
          {name || "-"}
        </span>
      );
    },
  },
  {
    id: "agent_id",
    accessorKey: "agent_id",
    meta: { title: t("agents.columns.id") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("agents.columns.id")} />,
    size: 200,
    enableSorting: true,
    cell: ({ row }) => (
      <IdentityCell
        title={row.original.agent_id}
        titleClassName="font-mono text-xs font-normal"
        onClick={() => onAgentClick(row.original.agent_id)}
      />
    ),
  },
  {
    id: "spend",
    accessorKey: "spend",
    meta: { title: t("agents.columns.spend") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("agents.columns.spend")} />,
    size: 130,
    enableSorting: true,
    cell: ({ row }) => <MoneyCell value={row.original.spend} decimals={4} />,
  },
  {
    id: "model",
    meta: { title: t("agents.columns.model") },
    header: t("agents.columns.model"),
    size: 170,
    enableSorting: false,
    cell: ({ row }) => {
      const model = row.original.litellm_params?.model;
      if (!model) {
        return <span className="text-muted-foreground">{t("agents.status.unavailable")}</span>;
      }
      return (
        <Badge variant="outline" className="max-w-40 font-normal">
          <span className="min-w-0 truncate" title={model}>
            {model}
          </span>
        </Badge>
      );
    },
  },
  {
    id: "created_at",
    accessorFn: (agent) => {
      const timestamp = agent.created_at ? new Date(agent.created_at).getTime() : 0;
      return Number.isNaN(timestamp) ? 0 : timestamp;
    },
    meta: { title: t("agents.columns.created") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("agents.columns.created")} />,
    size: 150,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.created_at} precision="date" />,
  },
  {
    id: "status",
    meta: { title: t("agents.columns.status") },
    header: t("agents.columns.status"),
    size: 130,
    enableSorting: false,
    cell: ({ row }) => {
      const hasKeys = (row.original.keys?.length ?? 0) > 0;
      return hasKeys ? (
        <StatusBadge tone="success" label={t("agents.status.active")} />
      ) : (
        <StatusBadge tone="warning" label={t("agents.status.needsSetup")} />
      );
    },
  },
  ...(isAdmin
    ? [
        {
          id: "actions",
          meta: { className: "text-right", headerClassName: "text-right" },
          header: () => <span className="sr-only">{t("agents.columns.actions")}</span>,
          size: 64,
          enableSorting: false,
          enableHiding: false,
          cell: ({ row }) => (
            <div className="flex justify-end">
              <AgentRowActions agent={row.original} onDeleteClick={onDeleteClick} t={t} />
            </div>
          ),
        } satisfies ColumnDef<Agent>,
      ]
    : []),
];
