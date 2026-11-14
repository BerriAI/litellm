"use client";

import { ColumnDef } from "@tanstack/react-table";
import type { TFunction } from "i18next";

import DefaultProxyAdminTag from "@/components/common_components/DefaultProxyAdminTag";
import { KeyResponse } from "@/components/key_team_helpers/key_list";
import { CellTooltip, DateCell } from "@/components/shared/table_cells";

function OwnerCell({ record }: { record: KeyResponse }) {
  const email = record.user?.user_email ?? record.user_id ?? null;
  if (!email) return <span className="text-sm">—</span>;
  return (
    <CellTooltip
      content={email}
      trigger={
        <span className="inline-flex max-w-60 truncate">
          <DefaultProxyAdminTag userId={email} />
        </span>
      }
    />
  );
}

export const getProjectKeysTableColumns = (t: TFunction): ColumnDef<KeyResponse>[] => [
  {
    id: "key_alias",
    accessorKey: "key_alias",
    meta: { title: t("projects.keys.columns.keyName") },
    header: t("projects.keys.columns.keyName"),
    enableSorting: false,
    cell: ({ row }) => <span className="text-sm">{row.original.key_alias || "—"}</span>,
  },
  {
    id: "owner",
    meta: { title: t("projects.keys.columns.owner") },
    header: t("projects.keys.columns.owner"),
    enableSorting: false,
    cell: ({ row }) => <OwnerCell record={row.original} />,
  },
  {
    id: "created_at",
    accessorKey: "created_at",
    meta: { title: t("projects.keys.columns.created") },
    header: t("projects.keys.columns.created"),
    size: 130,
    enableSorting: false,
    cell: ({ row }) => <DateCell value={row.original.created_at} precision="date" />,
  },
  {
    id: "last_active",
    accessorKey: "last_active",
    meta: { title: t("projects.keys.columns.lastActive") },
    header: t("projects.keys.columns.lastActive"),
    size: 130,
    enableSorting: false,
    cell: ({ row }) => (
      <DateCell value={row.original.last_active} precision="date" fallback={t("projects.keys.columns.never")} />
    ),
  },
];
