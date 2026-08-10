"use client";

import { ColumnDef } from "@tanstack/react-table";
import { useEffect, useMemo, useRef, useState } from "react";
import { MoreHorizontal, Trash2 } from "lucide-react";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { DateCell, IdentityCell } from "@/components/shared/table_cells";
import { Badge } from "@/components/ui/badge";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";

import { AutoRouterRow } from "./autoRouterRows";
import { fitPills } from "./fitPills";

function TypeCell({ row }: { row: AutoRouterRow }) {
  const { t } = useTranslation("gateway");
  const typeKey = row.typeLabel === "LLM Classifier" ? "llm" : row.typeLabel === "Heuristic" ? "heuristic" : row.kind;
  return (
    <Badge variant="secondary" className="font-normal">
      {t(`models.autoRouters.typeLabels.${typeKey}`, { defaultValue: row.typeLabel })}
    </Badge>
  );
}

function TargetsCell({ targets }: { targets: string[] }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const node = containerRef.current;
    if (!node || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver((entries) => {
      const measured = entries[0]?.contentRect.width;
      if (typeof measured === "number") setWidth(measured);
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const { visible, overflow } = useMemo(() => fitPills(targets, width), [targets, width]);

  if (targets.length === 0) {
    return <span className="text-sm text-muted-foreground">-</span>;
  }

  return (
    <div ref={containerRef} className="flex w-full min-w-0 flex-nowrap items-center gap-1 overflow-hidden">
      {visible.map((target) => (
        <Badge key={target} variant="secondary" className="max-w-full shrink truncate font-normal">
          {target}
        </Badge>
      ))}
      {overflow > 0 && (
        <span className="shrink-0 text-xs text-muted-foreground" title={targets.join(", ")}>
          +{overflow}
        </span>
      )}
    </div>
  );
}

function AutoRouterRowActions({
  row,
  onDeleteClick,
}: {
  row: AutoRouterRow;
  onDeleteClick: (row: AutoRouterRow) => void;
}) {
  const { t } = useTranslation("gateway");
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t("models.autoRouters.actions.open", { name: row.name })}
        data-testid={`auto-router-actions-${row.id}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-44">
        <DropdownMenuItem
          variant="destructive"
          data-testid="auto-router-action-delete"
          onClick={() => onDeleteClick(row)}
        >
          <Trash2 />
          {t("models.autoRouters.actions.delete")}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface AutoRoutersTableColumnsDeps {
  canModify: boolean;
  onRouterClick: (row: AutoRouterRow) => void;
  onDeleteClick: (row: AutoRouterRow) => void;
  t: TFunction<"gateway">;
}

export const getAutoRoutersTableColumns = ({
  canModify,
  onRouterClick,
  onDeleteClick,
  t,
}: AutoRoutersTableColumnsDeps): ColumnDef<AutoRouterRow>[] => [
  {
    id: "name",
    accessorKey: "name",
    meta: { title: t("models.autoRouters.columns.name") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("models.autoRouters.columns.name")} />,
    size: 260,
    enableSorting: true,
    cell: ({ row }) => <IdentityCell title={row.original.name || "-"} onClick={() => onRouterClick(row.original)} />,
  },
  {
    id: "kind",
    accessorKey: "kind",
    meta: { title: t("models.autoRouters.columns.type") },
    header: t("models.autoRouters.columns.type"),
    size: 180,
    enableSorting: false,
    cell: ({ row }) => <TypeCell row={row.original} />,
  },
  {
    id: "targets",
    meta: { title: t("models.autoRouters.columns.targets") },
    header: t("models.autoRouters.columns.targets"),
    size: 320,
    enableSorting: false,
    cell: ({ row }) => <TargetsCell targets={row.original.targets} />,
  },
  {
    id: "defaultModel",
    accessorKey: "defaultModel",
    meta: { title: t("models.autoRouters.columns.defaultModel") },
    header: t("models.autoRouters.columns.defaultModel"),
    size: 200,
    enableSorting: false,
    cell: ({ row }) =>
      row.original.defaultModel ? (
        <Badge variant="secondary" className="max-w-full truncate font-normal" title={row.original.defaultModel}>
          {row.original.defaultModel}
        </Badge>
      ) : (
        <span className="text-sm text-muted-foreground">-</span>
      ),
  },
  {
    id: "createdAt",
    accessorKey: "createdAt",
    meta: { title: t("models.autoRouters.columns.created") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("models.autoRouters.columns.created")} />,
    size: 150,
    enableSorting: true,
    sortingFn: "datetime",
    cell: ({ row }) => <DateCell value={row.original.createdAt} precision="date" />,
  },
  ...(canModify
    ? [
        {
          id: "actions",
          meta: { title: "" },
          header: "",
          size: 60,
          enableSorting: false,
          cell: ({ row }) =>
            row.original.canDelete ? <AutoRouterRowActions row={row.original} onDeleteClick={onDeleteClick} /> : null,
        } satisfies ColumnDef<AutoRouterRow>,
      ]
    : []),
];
