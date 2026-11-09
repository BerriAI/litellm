"use client";

import { ColumnDef } from "@tanstack/react-table";
import type { ParseKeys, TFunction } from "i18next";
import { Copy, MoreHorizontal, Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { DateCell, IdCell, StatusBadge } from "@/components/shared/table_cells";
import { PolicyAttachment } from "@/components/policies/types";
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

import ImpactPopover from "./impact_popover";

function ChipList({ values }: { values: string[] }) {
  if (values.length === 0) {
    return <span className="text-muted-foreground">-</span>;
  }
  return (
    <div className="flex flex-wrap items-center gap-1">
      {values.slice(0, 2).map((value) => (
        <StatusBadge key={value} tone="neutral" label={value} />
      ))}
      {values.length > 2 && (
        <StatusBadge tone="neutral" label={`+${values.length - 2}`} tooltip={values.slice(2).join(", ")} />
      )}
    </div>
  );
}

interface AttachmentRowActionsProps {
  attachment: PolicyAttachment;
  isAdmin: boolean;
  onDeleteClick: (attachmentId: string) => void;
}

const CONFIG_ATTACHMENT_HINT_KEY: ParseKeys = "policies.attachmentTable.configHint";

function AttachmentRowActions({ attachment, isAdmin, onDeleteClick }: AttachmentRowActionsProps) {
  const { t } = useTranslation();
  const isConfigAttachment = attachment.definition_location === "config";

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t("policies.attachmentTable.actions.open")}
        data-testid={`attachment-actions-${attachment.attachment_id}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem
          data-testid="attachment-action-copy-id"
          onClick={() => void copyToClipboard(attachment.attachment_id, t("policies.attachmentTable.actions.copied"))}
        >
          <Copy />
          {t("policies.attachmentTable.actions.copyId")}
        </DropdownMenuItem>
        {isAdmin && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              variant="destructive"
              data-testid="attachment-action-delete"
              disabled={isConfigAttachment}
              title={isConfigAttachment ? t(CONFIG_ATTACHMENT_HINT_KEY) : undefined}
              onClick={() => onDeleteClick(attachment.attachment_id)}
            >
              <Trash2 />
              {t("policies.attachmentTable.actions.delete")}
            </DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface AttachmentTableColumnsDeps {
  isAdmin: boolean;
  accessToken: string | null;
  onDeleteClick: (attachmentId: string) => void;
  t: TFunction;
}

export const getAttachmentTableColumns = ({
  isAdmin,
  accessToken,
  onDeleteClick,
  t,
}: AttachmentTableColumnsDeps): ColumnDef<PolicyAttachment>[] => [
  {
    id: "attachment_id",
    accessorKey: "attachment_id",
    meta: { title: t("policies.attachmentTable.columns.attachmentId") },
    header: t("policies.attachmentTable.columns.attachmentId"),
    size: 160,
    enableSorting: false,
    cell: ({ row }) => <IdCell value={row.original.attachment_id} variant="plain" />,
  },
  {
    id: "policy_name",
    accessorKey: "policy_name",
    meta: { title: t("policies.attachmentTable.columns.policy"), skeleton: "badge" },
    header: ({ column }) => (
      <DataTableSortHeader column={column} title={t("policies.attachmentTable.columns.policy")} />
    ),
    size: 180,
    enableSorting: true,
    cell: ({ row }) => <StatusBadge tone="info" label={row.original.policy_name} />,
  },
  {
    id: "scope",
    accessorFn: (row) => row.scope ?? "",
    meta: { title: t("policies.attachmentTable.columns.scope"), skeleton: "badge" },
    header: t("policies.attachmentTable.columns.scope"),
    size: 120,
    enableSorting: false,
    cell: ({ row }) => {
      const scope = row.original.scope;
      if (!scope) {
        return <span className="text-muted-foreground">-</span>;
      }
      if (scope === "*") {
        return <StatusBadge tone="warning" label={t("policies.attachmentTable.globalScope")} />;
      }
      return (
        <span className="block max-w-40 truncate text-xs" title={scope}>
          {scope}
        </span>
      );
    },
  },
  {
    id: "teams",
    meta: { title: t("policies.attachmentTable.columns.teams"), skeleton: "chips" },
    header: t("policies.attachmentTable.columns.teams"),
    size: 160,
    enableSorting: false,
    cell: ({ row }) => <ChipList values={row.original.teams ?? []} />,
  },
  {
    id: "keys",
    meta: { title: t("policies.attachmentTable.columns.keys"), skeleton: "chips" },
    header: t("policies.attachmentTable.columns.keys"),
    size: 160,
    enableSorting: false,
    cell: ({ row }) => <ChipList values={row.original.keys ?? []} />,
  },
  {
    id: "models",
    meta: { title: t("policies.attachmentTable.columns.models"), skeleton: "chips" },
    header: t("policies.attachmentTable.columns.models"),
    size: 160,
    enableSorting: false,
    cell: ({ row }) => <ChipList values={row.original.models ?? []} />,
  },
  {
    id: "tags",
    meta: { title: t("policies.attachmentTable.columns.tags"), skeleton: "chips" },
    header: t("policies.attachmentTable.columns.tags"),
    size: 160,
    enableSorting: false,
    cell: ({ row }) => <ChipList values={row.original.tags ?? []} />,
  },
  {
    id: "priority",
    accessorFn: (row) => row.priority ?? Number.POSITIVE_INFINITY,
    meta: { title: t("policies.attachmentTable.columns.priority") },
    header: ({ column }) => (
      <DataTableSortHeader column={column} title={t("policies.attachmentTable.columns.priority")} />
    ),
    size: 100,
    enableSorting: true,
    cell: ({ row }) =>
      row.original.priority == null ? (
        <span className="text-muted-foreground">-</span>
      ) : (
        <span className="font-mono text-xs">{row.original.priority}</span>
      ),
  },
  {
    id: "default",
    accessorFn: (row) => (row.default ? 1 : 0),
    meta: { title: t("policies.attachmentTable.columns.default") },
    header: ({ column }) => (
      <DataTableSortHeader column={column} title={t("policies.attachmentTable.columns.default")} />
    ),
    size: 100,
    enableSorting: true,
    cell: ({ row }) =>
      row.original.default ? (
        <StatusBadge
          tone="info"
          label={t("policies.attachmentTable.defaultBadge")}
          tooltip={t("policies.attachmentTable.defaultHint")}
        />
      ) : (
        <span className="text-muted-foreground">-</span>
      ),
  },
  {
    id: "created_at",
    accessorFn: (row) => row.created_at ?? "",
    meta: { title: t("policies.attachmentTable.columns.createdAt") },
    header: ({ column }) => (
      <DataTableSortHeader column={column} title={t("policies.attachmentTable.columns.createdAt")} />
    ),
    size: 150,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.created_at} />,
  },
  {
    id: "actions",
    meta: { className: "text-right", headerClassName: "text-right" },
    header: () => <span className="sr-only">{t("policies.attachmentTable.columns.actions")}</span>,
    size: 88,
    enableSorting: false,
    enableHiding: false,
    cell: ({ row }) => (
      <div className="flex items-center justify-end gap-1">
        <ImpactPopover attachment={row.original} accessToken={accessToken} />
        <AttachmentRowActions attachment={row.original} isAdmin={isAdmin} onDeleteClick={onDeleteClick} />
      </div>
    ),
  },
];
