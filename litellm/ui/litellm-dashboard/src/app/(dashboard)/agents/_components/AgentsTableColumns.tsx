"use client";

import { ColumnDef } from "@tanstack/react-table";
import { useTranslation } from "react-i18next";
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
import i18n from "@/i18n/i18n";
import { cn } from "@/lib/cva.config";

interface AgentRowActionsProps {
  agent: Agent;
  onDeleteClick: (agentId: string, agentName: string) => void;
}

function AgentRowActions({ agent, onDeleteClick }: AgentRowActionsProps) {
  const { t } = useTranslation();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t("agents.table.actions.open")}
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
          {t("agents.table.actions.delete")}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface AgentsTableColumnsDeps {
  isAdmin: boolean;
  onAgentClick: (agentId: string) => void;
  onDeleteClick: (agentId: string, agentName: string) => void;
}

export const getAgentsTableColumns = ({
  isAdmin,
  onAgentClick,
  onDeleteClick,
}: AgentsTableColumnsDeps): ColumnDef<Agent>[] => [
  {
    id: "agent_name",
    accessorKey: "agent_name",
    meta: { title: i18n.t("agents.table.columns.name") },
    header: ({ column }) => <DataTableSortHeader column={column} title={i18n.t("agents.table.columns.name")} />,
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
    meta: { title: i18n.t("agents.table.columns.id") },
    header: ({ column }) => <DataTableSortHeader column={column} title={i18n.t("agents.table.columns.id")} />,
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
    meta: { title: i18n.t("agents.table.columns.spend"), numeric: true },
    header: ({ column }) => <DataTableSortHeader column={column} title={i18n.t("agents.table.columns.spend")} />,
    size: 130,
    enableSorting: true,
    cell: ({ row }) => <MoneyCell value={row.original.spend} decimals={4} />,
  },
  {
    id: "model",
    meta: { title: i18n.t("agents.table.columns.model") },
    header: i18n.t("agents.table.columns.model"),
    size: 170,
    enableSorting: false,
    cell: ({ row }) => {
      const model = row.original.litellm_params?.model;
      if (!model) {
        return <span className="text-muted-foreground">{i18n.t("agents.table.noModel")}</span>;
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
    meta: { title: i18n.t("agents.table.columns.created") },
    header: ({ column }) => <DataTableSortHeader column={column} title={i18n.t("agents.table.columns.created")} />,
    size: 150,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.created_at} precision="date" />,
  },
  {
    id: "status",
    meta: { title: i18n.t("agents.table.columns.status") },
    header: i18n.t("agents.table.columns.status"),
    size: 130,
    enableSorting: false,
    cell: ({ row }) => {
      const hasKeys = (row.original.keys?.length ?? 0) > 0;
      if (row.original.jwt_auth_configured)
        return <StatusBadge tone="success" label={i18n.t("agents.table.status.jwtConfigured")} />;
      return hasKeys ? (
        <StatusBadge tone="success" label={i18n.t("agents.table.status.active")} />
      ) : (
        <StatusBadge tone="warning" label={i18n.t("agents.table.status.needsSetup")} />
      );
    },
  },
  ...(isAdmin
    ? [
        {
          id: "actions",
          meta: { className: "text-right", headerClassName: "text-right" },
          header: () => <span className="sr-only">{i18n.t("agents.table.columns.actions")}</span>,
          size: 64,
          enableSorting: false,
          enableHiding: false,
          cell: ({ row }) => (
            <div className="flex justify-end">
              <AgentRowActions agent={row.original} onDeleteClick={onDeleteClick} />
            </div>
          ),
        } satisfies ColumnDef<Agent>,
      ]
    : []),
];
