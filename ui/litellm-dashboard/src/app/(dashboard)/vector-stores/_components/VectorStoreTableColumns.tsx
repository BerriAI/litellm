"use client";

import { ColumnDef } from "@tanstack/react-table";
import { Copy, MoreHorizontal, Pencil, Trash2 } from "lucide-react";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { CellTooltip, DateCell, IdentityCell } from "@/components/shared/table_cells";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
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

function VectorStoreProviderCell({ provider }: { provider: string }) {
  const { displayName, logo } = getProviderLogoAndName(provider);
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
  const ingestedFiles = vectorStore.vector_store_metadata?.ingested_files || [];
  if (ingestedFiles.length === 0) {
    return <span className="text-sm text-muted-foreground">-</span>;
  }

  const filenames = ingestedFiles.map((file) => file.filename || file.file_url || "Unknown").join(", ");
  const displayText =
    ingestedFiles.length === 1
      ? ingestedFiles[0].filename || ingestedFiles[0].file_url || "1 file"
      : `${ingestedFiles.length} files`;

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
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label="Open vector store actions"
        data-testid={`vector-store-actions-${vectorStore.vector_store_id}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem data-testid="vector-store-action-edit" onClick={() => onEdit(vectorStore.vector_store_id)}>
          <Pencil />
          Edit
        </DropdownMenuItem>
        <DropdownMenuItem
          data-testid="vector-store-action-copy"
          onClick={() => void copyToClipboard(vectorStore.vector_store_id, "Vector store ID copied")}
        >
          <Copy />
          Copy vector store ID
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          variant="destructive"
          data-testid="vector-store-action-delete"
          onClick={() => onDelete(vectorStore.vector_store_id)}
        >
          <Trash2 />
          Delete
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface VectorStoreTableColumnsDeps {
  onView: (vectorStoreId: string) => void;
  onEdit: (vectorStoreId: string) => void;
  onDelete: (vectorStoreId: string) => void;
}

export const getVectorStoreTableColumns = ({
  onView,
  onEdit,
  onDelete,
}: VectorStoreTableColumnsDeps): ColumnDef<VectorStore>[] => [
  {
    id: "vector_store_id",
    accessorKey: "vector_store_id",
    meta: { title: "Vector Store ID" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Vector Store ID" />,
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
    meta: { title: "Name" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Name" />,
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
    meta: { title: "Description" },
    header: "Description",
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
    id: "files",
    meta: { title: "Files" },
    header: "Files",
    size: 160,
    enableSorting: false,
    cell: ({ row }) => <VectorStoreFilesCell vectorStore={row.original} />,
  },
  {
    id: "provider",
    accessorKey: "custom_llm_provider",
    meta: { title: "Provider" },
    header: "Provider",
    size: 160,
    enableSorting: false,
    cell: ({ row }) => <VectorStoreProviderCell provider={row.original.custom_llm_provider} />,
  },
  {
    id: "created_at",
    accessorKey: "created_at",
    sortingFn: "datetime",
    meta: { title: "Created At" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Created At" />,
    size: 150,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.created_at} precision="date" />,
  },
  {
    id: "updated_at",
    accessorKey: "updated_at",
    sortingFn: "datetime",
    meta: { title: "Updated At" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Updated At" />,
    size: 150,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.updated_at} precision="date" />,
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
        <VectorStoreRowActions vectorStore={row.original} onEdit={onEdit} onDelete={onDelete} />
      </div>
    ),
  },
];
