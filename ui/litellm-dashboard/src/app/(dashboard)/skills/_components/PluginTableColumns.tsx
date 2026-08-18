"use client";

import { ColumnDef } from "@tanstack/react-table";
import { Copy, MoreHorizontal, Trash2 } from "lucide-react";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { DateCell, IdentityCell, StatusBadge } from "@/components/shared/table_cells";
import { getCategoryBadgeColor } from "@/components/claude_code_plugins/helpers";
import { Plugin } from "@/components/claude_code_plugins/types";
import { Badge } from "@/components/ui/badge";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";
import { copyToClipboard } from "@/utils/dataUtils";

const CATEGORY_BADGE_CLASS: Record<ReturnType<typeof getCategoryBadgeColor>, string> = {
  blue: "border-blue-200 bg-blue-50 text-blue-600",
  green: "border-green-200 bg-green-50 text-green-600",
  purple: "border-purple-200 bg-purple-50 text-purple-600",
  red: "border-red-200 bg-red-50 text-red-600",
  orange: "border-orange-200 bg-orange-50 text-orange-600",
  yellow: "border-yellow-200 bg-yellow-50 text-yellow-600",
  gray: "border-gray-200 bg-gray-50 text-gray-600",
};

function PluginCategoryBadge({ category }: { category?: string }) {
  return (
    <Badge
      variant="outline"
      className={cn("whitespace-nowrap font-normal", CATEGORY_BADGE_CLASS[getCategoryBadgeColor(category)])}
    >
      {category || "Uncategorized"}
    </Badge>
  );
}

interface PluginRowActionsProps {
  plugin: Plugin;
  isAdmin: boolean;
  onDeleteClick: (pluginName: string, displayName: string) => void;
}

function PluginRowActions({ plugin, isAdmin, onDeleteClick }: PluginRowActionsProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label="Open skill actions"
        data-testid={`plugin-actions-${plugin.name}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem
          data-testid="plugin-action-copy"
          onClick={() => void copyToClipboard(plugin.id, "Skill ID copied")}
        >
          <Copy />
          Copy skill ID
        </DropdownMenuItem>
        {isAdmin && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              variant="destructive"
              data-testid="plugin-action-delete"
              onClick={() => onDeleteClick(plugin.name, plugin.name)}
            >
              <Trash2 />
              Delete
            </DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface PluginTableColumnsDeps {
  isAdmin: boolean;
  onPluginClick: (pluginId: string) => void;
  onDeleteClick: (pluginName: string, displayName: string) => void;
}

export const getPluginTableColumns = ({
  isAdmin,
  onPluginClick,
  onDeleteClick,
}: PluginTableColumnsDeps): ColumnDef<Plugin>[] => [
  {
    id: "name",
    accessorKey: "name",
    meta: { title: "Skill Name" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Skill Name" />,
    size: 220,
    enableSorting: true,
    cell: ({ row }) => (
      <IdentityCell
        title={row.original.name}
        titleClassName="font-mono text-xs font-normal"
        className="max-w-60"
        onClick={() => onPluginClick(row.original.id)}
      />
    ),
  },
  {
    id: "version",
    accessorKey: "version",
    meta: { title: "Version" },
    header: "Version",
    size: 100,
    enableSorting: false,
    cell: ({ row }) => <span className="text-sm text-muted-foreground">{row.original.version || "N/A"}</span>,
  },
  {
    id: "description",
    accessorKey: "description",
    meta: { title: "Description" },
    header: "Description",
    size: 300,
    enableSorting: false,
    cell: ({ row }) => {
      const description = row.original.description;
      return (
        <span className="block max-w-72 truncate text-sm text-muted-foreground" title={description}>
          {description || "No description"}
        </span>
      );
    },
  },
  {
    id: "category",
    accessorKey: "category",
    meta: { title: "Category", skeleton: "badge" },
    header: "Category",
    size: 150,
    enableSorting: false,
    cell: ({ row }) => <PluginCategoryBadge category={row.original.category} />,
  },
  {
    id: "enabled",
    accessorKey: "enabled",
    meta: { title: "Public", skeleton: "badge" },
    header: "Public",
    size: 100,
    enableSorting: false,
    cell: ({ row }) => (
      <StatusBadge tone={row.original.enabled ? "success" : "neutral"} label={row.original.enabled ? "Yes" : "No"} />
    ),
  },
  {
    id: "created_at",
    accessorKey: "created_at",
    sortingFn: "datetime",
    meta: { title: "Created At" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Created At" />,
    size: 160,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.created_at} />,
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
        <PluginRowActions plugin={row.original} isAdmin={isAdmin} onDeleteClick={onDeleteClick} />
      </div>
    ),
  },
];
