"use client";

import { ColumnDef } from "@tanstack/react-table";
import type { TFunction } from "i18next";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { DateCell, IdCell, IdentityCell, MoneyCell } from "@/components/shared/table_cells";
import { DeletedKeyResponse } from "@/app/(dashboard)/hooks/keys/useKeys";
import { userDetailHref } from "@/utils/entityLinks";

function TruncatedTextCell({ value }: { value: string | null | undefined }) {
  if (!value) {
    return <span className="text-muted-foreground">-</span>;
  }
  return (
    <span className="block max-w-60 truncate" title={value}>
      {value}
    </span>
  );
}

function UserLinkCell({ userId }: { userId: string | null | undefined }) {
  if (!userId) {
    return <span className="text-muted-foreground">-</span>;
  }
  return (
    <span className="block max-w-60" title={userId}>
      <IdentityCell title={userId} titleClassName="font-normal" href={userDetailHref(userId)} />
    </span>
  );
}

export const getDeletedKeysTableColumns = (t: TFunction): ColumnDef<DeletedKeyResponse>[] => [
  {
    id: "token",
    accessorKey: "token",
    meta: { title: t("logs.deletedKeys.columns.keyId") },
    header: t("logs.deletedKeys.columns.keyId"),
    size: 150,
    enableSorting: false,
    cell: ({ row }) => <IdCell value={row.original.token} variant="plain" />,
  },
  {
    id: "key_alias",
    accessorKey: "key_alias",
    meta: { title: t("logs.deletedKeys.columns.keyAlias") },
    header: t("logs.deletedKeys.columns.keyAlias"),
    size: 150,
    enableSorting: false,
    cell: ({ row }) => {
      const value = row.original.key_alias;
      if (!value) {
        return <span className="text-muted-foreground">-</span>;
      }
      return (
        <span className="block max-w-60 truncate font-mono text-xs" title={value}>
          {value}
        </span>
      );
    },
  },
  {
    id: "team_alias",
    accessorKey: "team_alias",
    meta: { title: t("logs.deletedKeys.columns.teamAlias") },
    header: t("logs.deletedKeys.columns.teamAlias"),
    size: 120,
    enableSorting: false,
    cell: ({ row }) => <TruncatedTextCell value={row.original.team_alias} />,
  },
  {
    id: "spend",
    accessorKey: "spend",
    meta: { title: t("logs.deletedKeys.columns.spendUsd"), numeric: true },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("logs.deletedKeys.columns.spendUsd")} />,
    size: 100,
    enableSorting: true,
    cell: ({ row }) => <MoneyCell value={row.original.spend} decimals={4} />,
  },
  {
    id: "max_budget",
    accessorKey: "max_budget",
    meta: { title: t("logs.deletedKeys.columns.budgetUsd"), numeric: true },
    header: t("logs.deletedKeys.columns.budgetUsd"),
    size: 110,
    enableSorting: false,
    cell: ({ row }) => (
      <MoneyCell
        value={row.original.max_budget}
        decimals={0}
        emptyText={t("logs.deletedKeys.columns.unlimited")}
        showZero
      />
    ),
  },
  {
    id: "user_email",
    accessorKey: "user_email",
    meta: { title: t("logs.deletedKeys.columns.userEmail") },
    header: t("logs.deletedKeys.columns.userEmail"),
    size: 160,
    enableSorting: false,
    cell: ({ row }) => <TruncatedTextCell value={row.original.user_email} />,
  },
  {
    id: "user_id",
    accessorKey: "user_id",
    meta: { title: t("logs.deletedKeys.columns.userId") },
    header: t("logs.deletedKeys.columns.userId"),
    size: 120,
    enableSorting: false,
    cell: ({ row }) => <UserLinkCell userId={row.original.user_id} />,
  },
  {
    id: "created_at",
    accessorKey: "created_at",
    meta: { title: t("logs.deletedKeys.columns.createdAt") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("logs.deletedKeys.columns.createdAt")} />,
    size: 120,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.created_at} precision="date" />,
  },
  {
    id: "created_by",
    accessorKey: "created_by",
    meta: { title: t("logs.deletedKeys.columns.createdBy") },
    header: t("logs.deletedKeys.columns.createdBy"),
    size: 120,
    enableSorting: false,
    cell: ({ row }) => <UserLinkCell userId={row.original.created_by} />,
  },
  {
    id: "deleted_at",
    accessorKey: "deleted_at",
    meta: { title: t("logs.deletedKeys.columns.deletedAt") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("logs.deletedKeys.columns.deletedAt")} />,
    size: 120,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.deleted_at} precision="date" />,
  },
  {
    id: "deleted_by",
    accessorKey: "deleted_by",
    meta: { title: t("logs.deletedKeys.columns.deletedBy") },
    header: t("logs.deletedKeys.columns.deletedBy"),
    size: 120,
    enableSorting: false,
    cell: ({ row }) => <UserLinkCell userId={row.original.deleted_by} />,
  },
];
