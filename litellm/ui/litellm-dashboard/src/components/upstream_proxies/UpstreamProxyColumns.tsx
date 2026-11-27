"use client";

import { ColumnDef } from "@tanstack/react-table";
import type { TFunction } from "i18next";
import { MoreHorizontal, Pencil, Trash2 } from "lucide-react";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { IdentityCell, StatusBadge } from "@/components/shared/table_cells";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";

import { upstreamProxyAddress, type UpstreamProxy } from "./api";

interface Deps {
  t: TFunction;
  canModify: boolean;
  onEdit: (proxy: UpstreamProxy) => void;
  onDelete: (proxy: UpstreamProxy) => void;
}

const usageSummary = (proxy: UpstreamProxy, t: TFunction): string => {
  const parts = [
    proxy.used_by_models.length > 0
      ? t("upstreamProxies.columns.models", { count: proxy.used_by_models.length })
      : null,
    proxy.used_by_credentials.length > 0
      ? t("upstreamProxies.columns.credentials", { count: proxy.used_by_credentials.length })
      : null,
  ].filter((part): part is string => part !== null);
  return parts.length > 0 ? parts.join(", ") : t("upstreamProxies.columns.unused");
};

const usageDetail = (proxy: UpstreamProxy): string =>
  [
    ...proxy.used_by_models.map((name) => `model: ${name}`),
    ...proxy.used_by_credentials.map((name) => `credential: ${name}`),
  ].join("\n");

function RowActions({ proxy, t, onEdit, onDelete }: { proxy: UpstreamProxy } & Omit<Deps, "canModify">) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t("upstreamProxies.columns.openActions", { name: proxy.name })}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-44">
        <DropdownMenuItem onClick={() => onEdit(proxy)}>
          <Pencil />
          {t("upstreamProxies.columns.edit")}
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem variant="destructive" onClick={() => onDelete(proxy)}>
          <Trash2 />
          {t("upstreamProxies.columns.delete")}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export const getUpstreamProxyColumns = ({ t, canModify, onEdit, onDelete }: Deps): ColumnDef<UpstreamProxy>[] => {
  const dataColumns: ColumnDef<UpstreamProxy>[] = [
    {
      id: "name",
      accessorKey: "name",
      meta: { title: t("upstreamProxies.columns.name") },
      header: ({ column }) => <DataTableSortHeader column={column} title={t("upstreamProxies.columns.name")} />,
      size: 200,
      enableSorting: true,
      cell: ({ row }) => (
        <IdentityCell
          title={row.original.name}
          subtitle={row.original.description ?? undefined}
          className="max-w-64"
          titleClassName="font-medium"
        />
      ),
    },
    {
      id: "address",
      accessorFn: upstreamProxyAddress,
      meta: { title: t("upstreamProxies.columns.address") },
      header: t("upstreamProxies.columns.address"),
      size: 260,
      enableSorting: false,
      cell: ({ row }) => <span className="font-mono text-sm">{upstreamProxyAddress(row.original)}</span>,
    },
    {
      id: "auth",
      accessorFn: (proxy) => proxy.username ?? "",
      meta: { title: t("upstreamProxies.columns.authentication") },
      header: t("upstreamProxies.columns.authentication"),
      size: 180,
      enableSorting: false,
      cell: ({ row }) =>
        row.original.username ? (
          <StatusBadge
            tone="info"
            label={row.original.has_password ? `${row.original.username} / ••••` : row.original.username}
          />
        ) : (
          <StatusBadge tone="neutral" label={t("upstreamProxies.columns.none")} />
        ),
    },
    {
      id: "usage",
      accessorFn: (proxy) => usageSummary(proxy, t),
      meta: { title: t("upstreamProxies.columns.usedBy") },
      header: t("upstreamProxies.columns.usedBy"),
      size: 200,
      enableSorting: false,
      cell: ({ row }) => (
        <span className="text-sm text-muted-foreground" title={usageDetail(row.original) || undefined}>
          {usageSummary(row.original, t)}
        </span>
      ),
    },
  ];
  if (!canModify) {
    return dataColumns;
  }
  return [
    ...dataColumns,
    {
      id: "actions",
      meta: { className: "text-right", headerClassName: "text-right" },
      header: () => <span className="sr-only">{t("upstreamProxies.columns.actions")}</span>,
      size: 64,
      enableSorting: false,
      enableHiding: false,
      cell: ({ row }) => (
        <div className="flex justify-end">
          <RowActions proxy={row.original} t={t} onEdit={onEdit} onDelete={onDelete} />
        </div>
      ),
    },
  ];
};
