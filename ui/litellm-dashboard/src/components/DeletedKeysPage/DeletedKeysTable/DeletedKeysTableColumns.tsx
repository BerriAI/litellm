"use client";

import { ColumnDef } from "@tanstack/react-table";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { DateCell, IdCell, IdentityCell, MoneyCell } from "@/components/shared/table_cells";
import { DeletedKeyResponse } from "@/app/(dashboard)/hooks/keys/useKeys";
import { userDetailHref } from "@/utils/entityLinks";
import { englishTranslate, type Translate } from "@/i18n/language";

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

// 列定义接收当前语言，接口字段和业务单元格保持原值。
export const getDeletedKeysTableColumns = (t: Translate = englishTranslate): ColumnDef<DeletedKeyResponse>[] => [
  {
    id: "token",
    accessorKey: "token",
    meta: { title: t("Key ID") },
    header: t("Key ID"),
    size: 150,
    enableSorting: false,
    cell: ({ row }) => <IdCell value={row.original.token} variant="plain" />,
  },
  {
    id: "key_alias",
    accessorKey: "key_alias",
    meta: { title: t("Key Alias") },
    header: t("Key Alias"),
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
    meta: { title: t("Team Alias") },
    header: t("Team Alias"),
    size: 120,
    enableSorting: false,
    cell: ({ row }) => <TruncatedTextCell value={row.original.team_alias} />,
  },
  {
    id: "spend",
    accessorKey: "spend",
    meta: { title: t("Spend (USD)"), numeric: true },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Spend (USD)")} />,
    size: 100,
    enableSorting: true,
    cell: ({ row }) => <MoneyCell value={row.original.spend} decimals={4} />,
  },
  {
    id: "max_budget",
    accessorKey: "max_budget",
    meta: { title: t("Budget (USD)"), numeric: true },
    header: t("Budget (USD)"),
    size: 110,
    enableSorting: false,
    cell: ({ row }) => <MoneyCell value={row.original.max_budget} decimals={0} emptyText={t("Unlimited")} showZero />,
  },
  {
    id: "user_email",
    accessorKey: "user_email",
    meta: { title: t("User Email") },
    header: t("User Email"),
    size: 160,
    enableSorting: false,
    cell: ({ row }) => <TruncatedTextCell value={row.original.user_email} />,
  },
  {
    id: "user_id",
    accessorKey: "user_id",
    meta: { title: t("User ID") },
    header: t("User ID"),
    size: 120,
    enableSorting: false,
    cell: ({ row }) => <UserLinkCell userId={row.original.user_id} />,
  },
  {
    id: "created_at",
    accessorKey: "created_at",
    meta: { title: t("Created At") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Created At")} />,
    size: 120,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.created_at} precision="date" />,
  },
  {
    id: "created_by",
    accessorKey: "created_by",
    meta: { title: t("Created By") },
    header: t("Created By"),
    size: 120,
    enableSorting: false,
    cell: ({ row }) => <UserLinkCell userId={row.original.created_by} />,
  },
  {
    id: "deleted_at",
    accessorKey: "deleted_at",
    meta: { title: t("Deleted At") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Deleted At")} />,
    size: 120,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.deleted_at} precision="date" />,
  },
  {
    id: "deleted_by",
    accessorKey: "deleted_by",
    meta: { title: t("Deleted By") },
    header: t("Deleted By"),
    size: 120,
    enableSorting: false,
    cell: ({ row }) => <UserLinkCell userId={row.original.deleted_by} />,
  },
];
