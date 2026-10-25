"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import { ColumnDef } from "@tanstack/react-table";
import { Bot, Layers, MoreHorizontal, Server, Sparkles, Trash2 } from "lucide-react";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { DateCell, IdentityCell } from "@/components/shared/table_cells";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";

import { AccessGroup } from "./types";
import { englishTranslate, type Translate } from "@/i18n/language";

interface ResourceTone {
  icon: typeof Layers;
  className: string;
}

const RESOURCE_TONES: Record<"models" | "mcpServers" | "agents" | "skills", ResourceTone> = {
  models: { icon: Layers, className: "bg-info/10 text-info ring-blue-600/20" },
  mcpServers: { icon: Server, className: "bg-info/10 text-info ring-cyan-600/20" },
  agents: {
    icon: Bot,
    className:
      "bg-purple-50 text-purple-700 ring-purple-600/20 dark:bg-purple-950 dark:text-purple-300 dark:ring-purple-400/30",
  },
  skills: {
    icon: Sparkles,
    className:
      "bg-amber-50 text-amber-700 ring-amber-600/20 dark:bg-amber-950 dark:text-amber-300 dark:ring-amber-400/30",
  },
};

function ResourcesCell({ group }: { group: AccessGroup }) {
  const { t } = useLanguage();
  const items = [
    { key: "models" as const, label: t("Models"), count: group.modelIds.length },
    { key: "mcpServers" as const, label: t("MCP Servers"), count: group.mcpServerIds.length },
    { key: "agents" as const, label: t("Agents"), count: group.agentIds.length },
    { key: "skills" as const, label: t("Skills"), count: group.skillNames.length },
  ];

  return (
    <div className="flex items-center gap-1.5">
      {items.map((item) => {
        const tone = RESOURCE_TONES[item.key];
        const Icon = tone.icon;
        return (
          <span
            key={item.key}
            title={`${item.count} ${item.label}`}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium ring-1 ring-inset [&_svg]:size-3.5",
              tone.className,
            )}
          >
            <Icon />
            <span className="tabular-nums">{item.count}</span>
          </span>
        );
      })}
    </div>
  );
}

function AccessGroupRowActions({
  group,
  onDeleteClick,
}: {
  group: AccessGroup;
  onDeleteClick: (group: AccessGroup) => void;
}) {
  const { t } = useLanguage();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t("Open access group actions")}
        data-testid={`access-group-actions-${group.id}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-44">
        <DropdownMenuItem
          variant="destructive"
          data-testid="access-group-action-delete"
          onClick={() => onDeleteClick(group)}
        >
          <Trash2 />
          {t("Delete access group")}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface AccessGroupsTableColumnsDeps {
  t?: Translate;
  canModify: boolean;
  onGroupClick: (id: string) => void;
  onDeleteClick: (group: AccessGroup) => void;
}

export const getAccessGroupsTableColumns = ({
  t = englishTranslate,
  canModify,
  onGroupClick,
  onDeleteClick,
}: AccessGroupsTableColumnsDeps): ColumnDef<AccessGroup>[] => {
  const columns: ColumnDef<AccessGroup>[] = [
    {
      id: "id",
      accessorKey: "id",
      meta: { title: "ID" },
      header: "ID",
      size: 200,
      enableSorting: false,
      cell: ({ row }) => (
        <IdentityCell
          title={row.original.id}
          titleClassName="font-mono text-xs font-normal"
          onClick={() => onGroupClick(row.original.id)}
        />
      ),
    },
    {
      id: "name",
      accessorKey: "name",
      meta: { title: t("Name") },
      header: ({ column }) => <DataTableSortHeader column={column} title={t("Name")} />,
      size: 220,
      enableSorting: true,
      cell: ({ row }) => {
        const name = row.original.name;
        return (
          <span className="block max-w-72 truncate text-sm font-medium" title={name}>
            {name || "-"}
          </span>
        );
      },
    },
    {
      id: "resources",
      meta: { title: t("Resources") },
      header: t("Resources"),
      size: 220,
      enableSorting: false,
      cell: ({ row }) => <ResourcesCell group={row.original} />,
    },
    {
      id: "createdAt",
      accessorKey: "createdAt",
      meta: { title: t("Created") },
      header: ({ column }) => <DataTableSortHeader column={column} title={t("Created")} />,
      size: 150,
      enableSorting: true,
      sortingFn: "datetime",
      cell: ({ row }) => <DateCell value={row.original.createdAt} precision="date" />,
    },
    {
      id: "updatedAt",
      accessorKey: "updatedAt",
      meta: { title: t("Updated") },
      header: t("Updated"),
      size: 150,
      enableSorting: false,
      cell: ({ row }) => <DateCell value={row.original.updatedAt} precision="date" />,
    },
  ];

  if (!canModify) {
    return columns;
  }

  return [
    ...columns,
    {
      id: "actions",
      meta: { className: "text-right", headerClassName: "text-right" },
      header: () => <span className="sr-only">{t("Actions")}</span>,
      size: 64,
      enableSorting: false,
      enableHiding: false,
      cell: ({ row }) => (
        <div className="flex justify-end">
          <AccessGroupRowActions group={row.original} onDeleteClick={onDeleteClick} />
        </div>
      ),
    },
  ];
};
