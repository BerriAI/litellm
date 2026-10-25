"use client";

import { englishTranslate, type Translate } from "@/i18n/language";
import { useLanguage } from "@/contexts/LanguageContext";
import { ColumnDef } from "@tanstack/react-table";
import { Copy, MoreHorizontal, Trash2 } from "lucide-react";

import { StatusBadge, type StatusTone } from "@/components/shared/table_cells";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { DocumentUpload } from "@/components/vector_store_management/types";
import { cn } from "@/lib/cva.config";
import { copyToClipboard } from "@/utils/dataUtils";

const STATUS_CONFIG: Record<DocumentUpload["status"], { tone: StatusTone; label: string }> = {
  uploading: { tone: "info", label: "Uploading" },
  done: { tone: "success", label: "Ready" },
  error: { tone: "error", label: "Error" },
  removed: { tone: "neutral", label: "Removed" },
};

function formatFileSize(bytes?: number): string {
  if (!bytes) return "-";
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(2)} KB`;
  return `${(kb / 1024).toFixed(2)} MB`;
}

function DocumentRowActions({ document, onRemove }: { document: DocumentUpload; onRemove: (uid: string) => void }) {
  const { t } = useLanguage();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t("Open document actions")}
        data-testid={`document-actions-${document.uid}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem
          data-testid="document-action-copy"
          onClick={() => void copyToClipboard(document.uid, t("Document ID copied to clipboard"))}
        >
          <Copy />
          {t("Copy document ID")}
        </DropdownMenuItem>
        <DropdownMenuItem
          variant="destructive"
          data-testid="document-action-remove"
          onClick={() => onRemove(document.uid)}
        >
          <Trash2 />
          {t("Remove")}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface DocumentsTableColumnsDeps {
  t?: Translate;
  onRemove: (uid: string) => void;
}

export const getDocumentsTableColumns = ({
  t = englishTranslate,
  onRemove,
}: DocumentsTableColumnsDeps): ColumnDef<DocumentUpload>[] => [
  {
    id: "name",
    accessorKey: "name",
    meta: { title: t("Name") },
    header: t("Name"),
    enableSorting: false,
    cell: ({ row }) => (
      <div className="flex items-center gap-2">
        <span className="block max-w-72 truncate text-sm" title={row.original.name}>
          {row.original.name}
        </span>
        {row.original.size ? (
          <span className="text-xs text-muted-foreground">({formatFileSize(row.original.size)})</span>
        ) : null}
      </div>
    ),
  },
  {
    id: "status",
    accessorKey: "status",
    meta: { title: t("Status"), skeleton: "badge" },
    header: t("Status"),
    size: 150,
    enableSorting: false,
    cell: ({ row }) => {
      const config = STATUS_CONFIG[row.original.status];
      return <StatusBadge tone={config?.tone ?? "neutral"} label={config ? t(config.label) : row.original.status} />;
    },
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
        <DocumentRowActions document={row.original} onRemove={onRemove} />
      </div>
    ),
  },
];
