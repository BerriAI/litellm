"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import { ColumnDef } from "@tanstack/react-table";
import { englishTranslate, type Translate } from "@/i18n/language";
import { Copy, Info, MoreHorizontal } from "lucide-react";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { MCP_REACHABLE_DESCRIPTION } from "@/components/mcp_tools/types";
import { IdentityCell, StatusBadge, type StatusTone } from "@/components/shared/table_cells";
import { Badge } from "@/components/ui/badge";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";
import { copyToClipboard } from "@/utils/dataUtils";

export interface MCPServerData {
  server_id: string;
  server_name: string;
  alias?: string | null;
  description?: string | null;
  url: string;
  transport: string;
  auth_type: string;
  credentials?: any;
  created_at: string;
  created_by: string;
  updated_at: string;
  updated_by: string;
  teams: string[];
  mcp_access_groups: string[];
  allowed_tools: string[];
  extra_headers: any[];
  mcp_info: Record<string, any>;
  static_headers: Record<string, any>;
  status: string;
  last_health_check?: string | null;
  health_check_error?: string | null;
  command?: string | null;
  args: string[];
  env: Record<string, any>;
  [key: string]: any;
}

const STATUS_TONES: Record<string, StatusTone> = {
  active: "success",
  inactive: "error",
  unknown: "neutral",
  healthy: "success",
  reachable: "info",
  unhealthy: "error",
};

interface MCPHubRowActionsProps {
  server: MCPServerData;
  onServerClick: (server: MCPServerData) => void;
}

function MCPHubRowActions({ server, onServerClick }: MCPHubRowActionsProps) {
  const { t } = useLanguage();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t("Open MCP server actions")}
        data-testid={`mcp-hub-actions-${server.server_id}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem data-testid="mcp-hub-action-details" onClick={() => onServerClick(server)}>
          <Info />
          {t("View details")}
        </DropdownMenuItem>
        <DropdownMenuItem
          data-testid="mcp-hub-action-copy"
          onClick={() => void copyToClipboard(server.server_name, t("Server name copied"))}
        >
          <Copy />
          {t("Copy server name")}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface MCPHubTableColumnsDeps {
  t?: Translate;
  onServerClick: (server: MCPServerData) => void;
}

export const getMCPHubTableColumns = ({
  onServerClick,
  t = englishTranslate,
}: MCPHubTableColumnsDeps): ColumnDef<MCPServerData>[] => [
  {
    id: "server_name",
    accessorKey: "server_name",
    meta: { title: t("Server Name") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Server Name")} />,
    size: 200,
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => (
      <IdentityCell title={row.original.server_name} className="max-w-72" onClick={() => onServerClick(row.original)} />
    ),
  },
  {
    id: "description",
    accessorKey: "description",
    meta: { title: t("Description"), className: "hidden md:table-cell" },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Description")} />,
    size: 240,
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => (
      <span className="block max-w-72 truncate text-xs" title={row.original.description || undefined}>
        {row.original.description || "-"}
      </span>
    ),
  },
  {
    id: "transport",
    accessorKey: "transport",
    meta: { title: t("Transport"), skeleton: "badge", className: "hidden md:table-cell" },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Transport")} />,
    size: 110,
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => (
      <Badge variant="secondary" className="font-mono font-normal">
        {row.original.transport}
      </Badge>
    ),
  },
  {
    id: "auth_type",
    accessorKey: "auth_type",
    meta: { title: t("Auth Type"), skeleton: "badge", className: "hidden md:table-cell" },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Auth Type")} />,
    size: 110,
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => (
      <StatusBadge tone={row.original.auth_type === "none" ? "neutral" : "success"} label={row.original.auth_type} />
    ),
  },
  {
    id: "status",
    accessorKey: "status",
    meta: { title: t("Status"), skeleton: "badge" },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Status")} />,
    size: 110,
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => (
      <StatusBadge
        tone={STATUS_TONES[row.original.status] || "neutral"}
        label={
          Object.hasOwn(STATUS_TONES, row.original.status)
            ? t(row.original.status)
            : row.original.status || t("unknown")
        }
        tooltip={row.original.status === "reachable" ? t(MCP_REACHABLE_DESCRIPTION) : undefined}
      />
    ),
  },
  {
    id: "allowed_tools",
    meta: { title: t("Tools"), skeleton: "chips", className: "hidden lg:table-cell" },
    header: t("Tools"),
    size: 180,
    enableSorting: false,
    cell: ({ row }) => {
      const tools = row.original.allowed_tools || [];
      return (
        <div className="flex flex-col gap-1">
          <span className="text-xs font-medium">
            {tools.length > 0
              ? t(tools.length === 1 ? "{count} tool" : "{count} tools", { count: tools.length })
              : t("All tools")}
          </span>
          {tools.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {tools.slice(0, 2).map((tool) => (
                <Badge key={tool} variant="secondary">
                  {tool}
                </Badge>
              ))}
              {tools.length > 2 && <span className="text-xs text-muted-foreground">+{tools.length - 2}</span>}
            </div>
          )}
        </div>
      );
    },
  },
  {
    id: "created_by",
    accessorKey: "created_by",
    meta: { title: t("Created By"), className: "hidden xl:table-cell" },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Created By")} />,
    size: 140,
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => (
      <span className="block max-w-60 truncate text-xs" title={row.original.created_by || undefined}>
        {row.original.created_by || "-"}
      </span>
    ),
  },
  {
    id: "is_public",
    accessorFn: (row) => row.mcp_info?.is_public === true,
    meta: { title: t("Hub listing"), skeleton: "badge", className: "hidden md:table-cell" },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Hub listing")} />,
    size: 100,
    enableSorting: true,
    sortingFn: (rowA, rowB) => {
      const publicA = rowA.original.mcp_info?.is_public === true ? 1 : 0;
      const publicB = rowB.original.mcp_info?.is_public === true ? 1 : 0;
      return publicA - publicB;
    },
    cell: ({ row }) => {
      const isPublic = row.original.mcp_info?.is_public === true;
      return <StatusBadge tone={isPublic ? "success" : "neutral"} label={isPublic ? t("Listed") : t("Unlisted")} />;
    },
  },
  {
    id: "actions",
    meta: { className: "text-right", headerClassName: "text-right" },
    header: () => <span className="sr-only">{t("Actions")}</span>,
    size: 64,
    enableSorting: false,
    enableHiding: false,
    cell: ({ row }) => (
      <div className="flex justify-end">
        <MCPHubRowActions server={row.original} onServerClick={onServerClick} />
      </div>
    ),
  },
];
