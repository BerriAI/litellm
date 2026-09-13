"use client";

import { ColumnDef } from "@tanstack/react-table";
import { MoreHorizontal, Pencil, Trash2 } from "lucide-react";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { DateCell, IdentityCell, StatusBadge } from "@/components/shared/table_cells";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";

import { AvailableSearchProvider, SearchTool } from "./types";

const CONFIG_EDIT_HINT = "Config search tools cannot be edited on the dashboard. Please edit the config file.";
const CONFIG_DELETE_HINT = "Config search tools cannot be deleted on the dashboard. Please edit the config file.";

export const searchToolKey = (tool: SearchTool): string => tool.search_tool_id || tool.search_tool_name;

interface SearchToolRowActionsProps {
  tool: SearchTool;
  onEdit: (searchToolId: string) => void;
  onDelete: (searchToolId: string) => void;
}

function SearchToolRowActions({ tool, onEdit, onDelete }: SearchToolRowActionsProps) {
  const isFromConfig = tool.is_from_config ?? false;
  const toolId = tool.search_tool_id;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label="Open search tool actions"
        data-testid={`search-tool-actions-${searchToolKey(tool)}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem
          disabled={isFromConfig || !toolId}
          data-testid="search-tool-action-edit"
          title={isFromConfig ? CONFIG_EDIT_HINT : undefined}
          onClick={() => toolId && onEdit(toolId)}
        >
          <Pencil />
          Edit search tool
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          variant="destructive"
          disabled={isFromConfig || !toolId}
          data-testid="search-tool-action-delete"
          title={isFromConfig ? CONFIG_DELETE_HINT : undefined}
          onClick={() => toolId && onDelete(toolId)}
        >
          <Trash2 />
          Delete search tool
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface SearchToolTableColumnsDeps {
  availableProviders: AvailableSearchProvider[];
  onView: (searchToolId: string) => void;
  onEdit: (searchToolId: string) => void;
  onDelete: (searchToolId: string) => void;
}

export const getSearchToolTableColumns = ({
  availableProviders,
  onView,
  onEdit,
  onDelete,
}: SearchToolTableColumnsDeps): ColumnDef<SearchTool>[] => [
  {
    id: "search_tool_id",
    accessorKey: "search_tool_id",
    meta: { title: "Search Tool ID" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Search Tool ID" />,
    size: 200,
    enableSorting: true,
    cell: ({ row }) => {
      const tool = row.original;
      const toolId = tool.search_tool_id;
      if (tool.is_from_config || !toolId) {
        return <span className="text-muted-foreground">-</span>;
      }
      return (
        <IdentityCell title={toolId} titleClassName="font-mono text-xs font-normal" onClick={() => onView(toolId)} />
      );
    },
  },
  {
    id: "search_tool_name",
    accessorKey: "search_tool_name",
    meta: { title: "Name" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Name" />,
    size: 200,
    enableSorting: true,
    cell: ({ row }) => (
      <span className="block max-w-60 truncate text-sm font-medium" title={row.original.search_tool_name}>
        {row.original.search_tool_name || "-"}
      </span>
    ),
  },
  {
    id: "provider",
    meta: { title: "Provider" },
    header: "Provider",
    size: 160,
    enableSorting: false,
    cell: ({ row }) => {
      const provider = row.original.litellm_params.search_provider;
      const providerInfo = availableProviders.find((candidate) => candidate.provider_name === provider);
      return <span className="text-sm">{providerInfo?.ui_friendly_name || provider}</span>;
    },
  },
  {
    id: "created_at",
    accessorKey: "created_at",
    meta: { title: "Created At" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Created At" />,
    size: 130,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.created_at} precision="date" />,
  },
  {
    id: "updated_at",
    accessorKey: "updated_at",
    meta: { title: "Updated At" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Updated At" />,
    size: 130,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.updated_at} precision="date" />,
  },
  {
    id: "source",
    meta: { title: "Source", skeleton: "badge" },
    header: "Source",
    size: 100,
    enableSorting: false,
    cell: ({ row }) => {
      const isFromConfig = row.original.is_from_config ?? false;
      return <StatusBadge tone={isFromConfig ? "neutral" : "info"} label={isFromConfig ? "Config" : "DB"} />;
    },
  },
  {
    id: "actions",
    meta: { className: "text-right", headerClassName: "text-right" },
    header: () => <span className="sr-only">Actions</span>,
    size: 64,
    enableSorting: false,
    enableHiding: false,
    cell: ({ row }) => (
      <div className="flex justify-end">
        <SearchToolRowActions tool={row.original} onEdit={onEdit} onDelete={onDelete} />
      </div>
    ),
  },
];
