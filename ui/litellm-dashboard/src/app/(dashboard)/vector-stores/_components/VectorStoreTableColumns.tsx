"use client";

import { ColumnDef } from "@tanstack/react-table";
import type { ParseKeys, TFunction } from "i18next";
import { Copy, MoreHorizontal, Pencil, Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { CellTooltip, DateCell, IdentityCell, StatusBadge } from "@/components/shared/table_cells";
import { getVectorStoreProviderLogoAndName } from "@/components/vector_store_providers";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { VectorStore } from "@/components/vector_store_management/types";
import { cn } from "@/lib/cva.config";
import { copyToClipboard } from "@/utils/dataUtils";

const CONFIG_STORE_HINT: ParseKeys = "vectorStores.table.actions.configHint";

function VectorStoreProviderCell({ provider }: { provider: string }) {
  const { displayName, logo } = getVectorStoreProviderLogoAndName(provider);
  return (
    <div className="flex items-center gap-2">
      {logo ? (
        <img
          src={logo}
          alt=""
          className="size-4 shrink-0"
          onError={(event) => {
            (event.currentTarget as HTMLImageElement).style.display = "none";
          }}
        />
      ) : null}
      <span className="truncate text-sm">{displayName}</span>
    </div>
  );
}

function VectorStoreFilesCell({ vectorStore }: { vectorStore: VectorStore }) {
  const { t } = useTranslation();
  const ingestedFiles = vectorStore.vector_store_metadata?.ingested_files || [];
  if (ingestedFiles.length === 0) {
    return <span className="text-sm text-muted-foreground">-</span>;
  }

  const filenames = ingestedFiles
    .map((file) => file.filename || file.file_url || t("vectorStores.table.files.unknown"))
    .join(", ");
  const displayText =
    ingestedFiles.length === 1
      ? ingestedFiles[0].filename || ingestedFiles[0].file_url || t("vectorStores.table.files.count", { count: 1 })
      : t("vectorStores.table.files.count", { count: ingestedFiles.length });

  return (
    <CellTooltip
      content={filenames}
      trigger={<span className="block max-w-60 truncate text-sm text-primary">{displayText}</span>}
    />
  );
}

interface VectorStoreRowActionsProps {
  vectorStore: VectorStore;
  onEdit: (vectorStoreId: string) => void;
  onDelete: (vectorStoreId: string) => void;
}

function VectorStoreRowActions({ vectorStore, onEdit, onDelete }: VectorStoreRowActionsProps) {
  const { t } = useTranslation();
  const isFromConfig = vectorStore.is_config ?? false;
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t("vectorStores.table.actions.open")}
        data-testid={`vector-store-actions-${vectorStore.vector_store_id}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem
          data-testid="vector-store-action-edit"
          disabled={isFromConfig}
          onClick={() => onEdit(vectorStore.vector_store_id)}
        >
          <Pencil />
          {t("vectorStores.table.actions.edit")}
        </DropdownMenuItem>
        <DropdownMenuItem
          data-testid="vector-store-action-copy"
          onClick={() => void copyToClipboard(vectorStore.vector_store_id, t("vectorStores.table.actions.idCopied"))}
        >
          <Copy />
          {t("vectorStores.table.actions.copyId")}
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          variant="destructive"
          data-testid="vector-store-action-delete"
          disabled={isFromConfig}
          onClick={() => onDelete(vectorStore.vector_store_id)}
        >
          <Trash2 />
          {t("vectorStores.table.actions.delete")}
        </DropdownMenuItem>
        {isFromConfig && (
          <div data-testid="vector-store-config-hint" className="px-2 py-1.5 text-xs text-muted-foreground">
            {t(CONFIG_STORE_HINT)}
          </div>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface VectorStoreTableColumnsDeps {
  onView: (vectorStoreId: string) => void;
  onEdit: (vectorStoreId: string) => void;
  onDelete: (vectorStoreId: string) => void;
}

export const getVectorStoreTableColumns = (
  { onView, onEdit, onDelete }: VectorStoreTableColumnsDeps,
  t: TFunction,
): ColumnDef<VectorStore>[] => [
  {
    id: "vector_store_id",
    accessorKey: "vector_store_id",
    meta: { title: t("vectorStores.table.columns.vectorStoreId") },
    header: ({ column }) => (
      <DataTableSortHeader column={column} title={t("vectorStores.table.columns.vectorStoreId")} />
    ),
    size: 220,
    enableSorting: true,
    cell: ({ row }) => (
      <IdentityCell
        title={row.original.vector_store_id}
        titleClassName="font-mono text-xs font-normal"
        className="max-w-60"
        onClick={() => onView(row.original.vector_store_id)}
      />
    ),
  },
  {
    id: "vector_store_name",
    accessorKey: "vector_store_name",
    meta: { title: t("vectorStores.table.columns.name") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("vectorStores.table.columns.name")} />,
    size: 200,
    enableSorting: true,
    cell: ({ row }) => {
      const name = row.original.vector_store_name;
      return (
        <span className="block max-w-60 truncate text-sm font-medium" title={name ?? undefined}>
          {name || "-"}
        </span>
      );
    },
  },
  {
    id: "vector_store_description",
    accessorKey: "vector_store_description",
    meta: { title: t("vectorStores.table.columns.description") },
    header: t("vectorStores.table.columns.description"),
    size: 280,
    enableSorting: false,
    cell: ({ row }) => {
      const description = row.original.vector_store_description;
      return (
        <span className="block max-w-72 truncate text-sm text-muted-foreground" title={description ?? undefined}>
          {description || "-"}
        </span>
      );
    },
  },
  {
    id: "source",
    accessorFn: (row) => row.is_config ?? false,
    meta: { title: t("vectorStores.table.columns.source"), skeleton: "badge" },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("vectorStores.table.columns.source")} />,
    size: 110,
    enableSorting: true,
    cell: ({ row }) => {
      const isFromConfig = row.original.is_config ?? false;
      return (
        <StatusBadge
          tone={isFromConfig ? "neutral" : "info"}
          label={isFromConfig ? t("vectorStores.table.source.config") : t("vectorStores.table.source.db")}
        />
      );
    },
  },
  {
    id: "files",
    meta: { title: t("vectorStores.table.columns.files") },
    header: t("vectorStores.table.columns.files"),
    size: 160,
    enableSorting: false,
    cell: ({ row }) => <VectorStoreFilesCell vectorStore={row.original} />,
  },
  {
    id: "provider",
    accessorKey: "custom_llm_provider",
    meta: { title: t("vectorStores.table.columns.provider") },
    header: t("vectorStores.table.columns.provider"),
    size: 160,
    enableSorting: false,
    cell: ({ row }) => <VectorStoreProviderCell provider={row.original.custom_llm_provider} />,
  },
  {
    id: "created_at",
    accessorKey: "created_at",
    sortingFn: "datetime",
    meta: { title: t("vectorStores.table.columns.createdAt") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("vectorStores.table.columns.createdAt")} />,
    size: 150,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.created_at} precision="date" />,
  },
  {
    id: "updated_at",
    accessorKey: "updated_at",
    sortingFn: "datetime",
    meta: { title: t("vectorStores.table.columns.updatedAt") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("vectorStores.table.columns.updatedAt")} />,
    size: 150,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.updated_at} precision="date" />,
  },
  {
    id: "actions",
    meta: { className: "text-right", headerClassName: "text-right" },
    header: () => <span className="sr-only">{t("vectorStores.table.columns.actions")}</span>,
    size: 64,
    enableSorting: false,
    enableHiding: false,
    cell: ({ row }) => (
      <div className="flex justify-end">
        <VectorStoreRowActions vectorStore={row.original} onEdit={onEdit} onDelete={onDelete} />
      </div>
    ),
  },
];
