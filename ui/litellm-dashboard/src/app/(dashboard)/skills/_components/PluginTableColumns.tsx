"use client";

import { englishTranslate, type Translate } from "@/i18n/language";
import { useLanguage } from "@/contexts/LanguageContext";
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
  blue: "border-info/20 bg-info/10 text-info",
  green: "border-success/20 bg-success/10 text-success",
  purple:
    "border-purple-200 bg-purple-50 text-purple-600 dark:border-purple-800 dark:bg-purple-950 dark:text-purple-300",
  red: "border-destructive/20 bg-destructive/10 text-destructive",
  orange: "border-warning/20 bg-warning/10 text-warning",
  yellow: "border-warning/20 bg-warning/10 text-warning",
  gray: "border-border bg-muted text-muted-foreground",
};

function PluginCategoryBadge({ category }: { category?: string }) {
  const { t } = useLanguage();
  return (
    <Badge
      variant="outline"
      className={cn("whitespace-nowrap font-normal", CATEGORY_BADGE_CLASS[getCategoryBadgeColor(category)])}
    >
      {category || t("Uncategorized")}
    </Badge>
  );
}

interface PluginRowActionsProps {
  plugin: Plugin;
  isAdmin: boolean;
  onDeleteClick: (pluginName: string, displayName: string) => void;
}

function PluginRowActions({ plugin, isAdmin, onDeleteClick }: PluginRowActionsProps) {
  const { t } = useLanguage();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t("Open skill actions")}
        data-testid={`plugin-actions-${plugin.name}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem
          data-testid="plugin-action-copy"
          onClick={() => void copyToClipboard(plugin.id, t("Skill ID copied"))}
        >
          <Copy />
          {t("Copy skill ID")}
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
              {t("Delete")}
            </DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface PluginTableColumnsDeps {
  t?: Translate;
  isAdmin: boolean;
  onPluginClick: (pluginId: string) => void;
  onDeleteClick: (pluginName: string, displayName: string) => void;
}

export const getPluginTableColumns = ({
  t = englishTranslate,
  isAdmin,
  onPluginClick,
  onDeleteClick,
}: PluginTableColumnsDeps): ColumnDef<Plugin>[] => [
  {
    id: "name",
    accessorKey: "name",
    meta: { title: t("Skill Name") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Skill Name")} />,
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
    meta: { title: t("Version") },
    header: t("Version"),
    size: 100,
    enableSorting: false,
    cell: ({ row }) => <span className="text-sm text-muted-foreground">{row.original.version || "N/A"}</span>,
  },
  {
    id: "description",
    accessorKey: "description",
    meta: { title: t("Description") },
    header: t("Description"),
    size: 300,
    enableSorting: false,
    cell: ({ row }) => {
      const description = row.original.description;
      return (
        <span className="block max-w-72 truncate text-sm text-muted-foreground" title={description ?? undefined}>
          {description || t("No description")}
        </span>
      );
    },
  },
  {
    id: "category",
    accessorKey: "category",
    meta: { title: t("Category"), skeleton: "badge" },
    header: t("Category"),
    size: 150,
    enableSorting: false,
    cell: ({ row }) => <PluginCategoryBadge category={row.original.category ?? undefined} />,
  },
  {
    id: "enabled",
    accessorKey: "enabled",
    meta: { title: t("Public"), skeleton: "badge" },
    header: t("Public"),
    size: 100,
    enableSorting: false,
    cell: ({ row }) => (
      <StatusBadge
        tone={row.original.enabled ? "success" : "neutral"}
        label={row.original.enabled ? t("Yes") : t("No")}
      />
    ),
  },
  {
    id: "created_at",
    accessorKey: "created_at",
    sortingFn: "datetime",
    meta: { title: t("Created At") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Created At")} />,
    size: 160,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.created_at} />,
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
        <PluginRowActions plugin={row.original} isAdmin={isAdmin} onDeleteClick={onDeleteClick} />
      </div>
    ),
  },
];
