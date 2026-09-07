"use client";

import { ColumnDef } from "@tanstack/react-table";
import { Copy, ExternalLink, Info, MoreHorizontal } from "lucide-react";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { IdentityCell, StatusBadge } from "@/components/shared/table_cells";
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
import { Plugin } from "@/components/claude_code_plugins/types";

function getSkillSourceLink(skill: Plugin): { url: string; label: string } | null {
  const src = skill.source;
  if (src?.source === "github" && src.repo) {
    return { url: `https://github.com/${src.repo}`, label: src.repo };
  }
  if (src?.source === "git-subdir" && src.url) {
    const url = src.path ? `${src.url}/tree/main/${src.path}` : src.url;
    return { url, label: url.replace("https://github.com/", "") };
  }
  if (src?.source === "url" && src.url) {
    return { url: src.url, label: src.url.replace(/^https?:\/\//, "") };
  }
  return null;
}

interface SkillHubRowActionsProps {
  skill: Plugin;
  onSkillClick: (skill: Plugin) => void;
}

function SkillHubRowActions({ skill, onSkillClick }: SkillHubRowActionsProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label="Open skill actions"
        data-testid={`skill-hub-actions-${skill.id}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem data-testid="skill-hub-action-details" onClick={() => onSkillClick(skill)}>
          <Info />
          View details
        </DropdownMenuItem>
        <DropdownMenuItem
          data-testid="skill-hub-action-copy"
          onClick={() => void copyToClipboard(skill.name, "Skill name copied")}
        >
          <Copy />
          Copy skill name
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface SkillHubTableColumnsDeps {
  onSkillClick: (skill: Plugin) => void;
}

export const getSkillHubTableColumns = ({ onSkillClick }: SkillHubTableColumnsDeps): ColumnDef<Plugin>[] => [
  {
    id: "name",
    accessorKey: "name",
    meta: { title: "Skill Name" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Skill Name" />,
    size: 200,
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => (
      <IdentityCell title={row.original.name} className="max-w-72" onClick={() => onSkillClick(row.original)} />
    ),
  },
  {
    id: "description",
    accessorKey: "description",
    meta: { title: "Description" },
    header: "Description",
    size: 260,
    enableSorting: false,
    cell: ({ row }) => (
      <span className="block max-w-72 truncate text-xs" title={row.original.description || undefined}>
        {row.original.description || "-"}
      </span>
    ),
  },
  {
    id: "category",
    accessorKey: "category",
    meta: { title: "Category", skeleton: "badge" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Category" />,
    size: 130,
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) =>
      row.original.category ? (
        <Badge variant="secondary">{row.original.category}</Badge>
      ) : (
        <span className="text-xs text-muted-foreground">-</span>
      ),
  },
  {
    id: "domain",
    accessorKey: "domain",
    meta: { title: "Domain" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Domain" />,
    size: 130,
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => <span className="text-xs">{row.original.domain || "-"}</span>,
  },
  {
    id: "source",
    meta: { title: "Source" },
    header: "Source",
    size: 200,
    enableSorting: false,
    cell: ({ row }) => {
      const link = getSkillSourceLink(row.original);
      if (!link) return <span className="text-xs text-muted-foreground">-</span>;
      return (
        <a
          href={link.url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex max-w-60 items-center gap-1 text-xs text-primary hover:underline"
          title={link.label}
        >
          <span className="truncate">{link.label}</span>
          <ExternalLink className="size-3 shrink-0" />
        </a>
      );
    },
  },
  {
    id: "enabled",
    accessorKey: "enabled",
    meta: { title: "Status", skeleton: "badge" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Status" />,
    size: 100,
    enableSorting: true,
    cell: ({ row }) => (
      <StatusBadge
        tone={row.original.enabled ? "success" : "neutral"}
        label={row.original.enabled ? "Public" : "Draft"}
      />
    ),
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
        <SkillHubRowActions skill={row.original} onSkillClick={onSkillClick} />
      </div>
    ),
  },
];
