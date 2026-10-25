"use client";

import { englishTranslate, type Translate } from "@/i18n/language";
import { useLanguage } from "@/contexts/LanguageContext";
import { ColumnDef } from "@tanstack/react-table";
import { Copy, Link2, MoreHorizontal, Pencil, Trash2 } from "lucide-react";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { DateCell, IdCell, IdentityCell } from "@/components/shared/table_cells";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";
import { getGatewayBaseUrl } from "@/components/networking";
import { MCPToolset } from "@/components/mcp_tools/types";
import { copyToClipboard } from "@/utils/dataUtils";

// Display-only. Toolsets persist {server_id, bare tool_name}; the gateway serves
// each tool prefixed as "{server-prefix}-{tool}". Render that qualified form so
// the same tool name on different servers stays distinguishable. This mirrors the
// backend default MCP_TOOL_PREFIX_SEPARATOR; overriding that env var only changes
// this cosmetic label, never what is stored or how tools are matched.
const MCP_TOOL_PREFIX_SEPARATOR = "-";

export function displayToolName(serverPrefix: string | undefined, toolName: string): string {
  return serverPrefix ? `${serverPrefix}${MCP_TOOL_PREFIX_SEPARATOR}${toolName}` : toolName;
}

export function toolsetEndpointUrl(toolsetName: string): string {
  return `${getGatewayBaseUrl()}/toolset/${toolsetName}/mcp`;
}

interface ToolsetRowActionsProps {
  toolset: MCPToolset;
  isAdmin: boolean;
  onEditClick: (toolset: MCPToolset) => void;
  onDeleteClick: (toolsetId: string) => void;
}

function ToolsetRowActions({ toolset, isAdmin, onEditClick, onDeleteClick }: ToolsetRowActionsProps) {
  const { t } = useLanguage();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t("Open toolset actions")}
        data-testid={`toolset-actions-${toolset.toolset_id}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem
          data-testid="toolset-action-copy-url"
          onClick={() => void copyToClipboard(toolsetEndpointUrl(toolset.toolset_name), t("Endpoint URL copied"))}
        >
          <Link2 />
          {t("Copy endpoint URL")}
        </DropdownMenuItem>
        <DropdownMenuItem
          data-testid="toolset-action-copy-id"
          onClick={() => void copyToClipboard(toolset.toolset_id, t("Toolset ID copied"))}
        >
          <Copy />
          {t("Copy toolset ID")}
        </DropdownMenuItem>
        {isAdmin && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem data-testid="toolset-action-edit" onClick={() => onEditClick(toolset)}>
              <Pencil />
              {t("Edit")}
            </DropdownMenuItem>
            <DropdownMenuItem
              variant="destructive"
              data-testid="toolset-action-delete"
              onClick={() => onDeleteClick(toolset.toolset_id)}
            >
              <Trash2 />
              {t("Delete")}
            </DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface MCPToolsetTableColumnsDeps {
  t?: Translate;
  isAdmin: boolean;
  serverPrefixById: Map<string, string>;
  onEditClick: (toolset: MCPToolset) => void;
  onDeleteClick: (toolsetId: string) => void;
}

export const getMCPToolsetTableColumns = ({
  t = englishTranslate,
  isAdmin,
  serverPrefixById,
  onEditClick,
  onDeleteClick,
}: MCPToolsetTableColumnsDeps): ColumnDef<MCPToolset>[] => [
  {
    id: "toolset_id",
    accessorKey: "toolset_id",
    meta: { title: t("Toolset ID") },
    header: t("Toolset ID"),
    size: 140,
    enableSorting: false,
    cell: ({ row }) => <IdCell value={row.original.toolset_id} />,
  },
  {
    id: "toolset_name",
    accessorKey: "toolset_name",
    meta: { title: t("Name") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Name")} />,
    size: 260,
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => (
      <IdentityCell
        title={row.original.toolset_name}
        subtitle={toolsetEndpointUrl(row.original.toolset_name)}
        className="max-w-80"
        onClick={isAdmin ? () => onEditClick(row.original) : undefined}
      />
    ),
  },
  {
    id: "description",
    accessorKey: "description",
    meta: { title: t("Description") },
    header: t("Description"),
    size: 200,
    enableSorting: false,
    cell: ({ row }) => (
      <span className="block max-w-72 truncate text-sm text-muted-foreground" title={row.original.description}>
        {row.original.description || "—"}
      </span>
    ),
  },
  {
    id: "tools",
    meta: { title: t("Tools"), skeleton: "chips" },
    header: t("Tools"),
    size: 260,
    enableSorting: false,
    cell: ({ row }) => {
      const tools = row.original.tools;
      return (
        <div className="flex max-w-xs flex-wrap gap-1">
          {tools.slice(0, 4).map((tool) => (
            <span
              key={`${tool.server_id}-${tool.tool_name}`}
              className="inline-flex items-center rounded-md bg-muted px-1.5 py-0.5 text-xs"
            >
              {displayToolName(serverPrefixById.get(tool.server_id), tool.tool_name)}
            </span>
          ))}
          {tools.length > 4 && (
            <span className="self-center text-xs text-muted-foreground">
              {t("+{count} more", { count: tools.length - 4 })}
            </span>
          )}
        </div>
      );
    },
  },
  {
    id: "created_at",
    accessorKey: "created_at",
    meta: { title: t("Created") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Created")} />,
    size: 120,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.created_at} precision="date" />,
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
        <ToolsetRowActions
          toolset={row.original}
          isAdmin={isAdmin}
          onEditClick={onEditClick}
          onDeleteClick={onDeleteClick}
        />
      </div>
    ),
  },
];
