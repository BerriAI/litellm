"use client";

import { ColumnDef } from "@tanstack/react-table";
import type { TFunction } from "i18next";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { DateCell, IdCell, IdentityCell, ModelsCell, MoneyCell } from "@/components/shared/table_cells";
import { DeletedTeam } from "@/app/(dashboard)/hooks/teams/useTeams";
import { orgDetailHref, userDetailHref } from "@/utils/entityLinks";

function EntityCell({ value, href }: { value: string | null | undefined; href: string | undefined }) {
  if (!value) {
    return <span className="text-muted-foreground">-</span>;
  }
  return (
    <span className="block max-w-60" title={value}>
      <IdentityCell title={value} titleClassName="font-normal" href={href} />
    </span>
  );
}

export const getDeletedTeamsTableColumns = (t: TFunction): ColumnDef<DeletedTeam>[] => [
  {
    id: "team_alias",
    accessorKey: "team_alias",
    meta: { title: t("logs.deletedTeams.columns.teamName") },
    header: t("logs.deletedTeams.columns.teamName"),
    size: 150,
    enableSorting: false,
    cell: ({ row }) => {
      const value = row.original.team_alias;
      if (!value) {
        return <span className="text-muted-foreground">-</span>;
      }
      return (
        <span className="block max-w-60 truncate font-medium" title={value}>
          {value}
        </span>
      );
    },
  },
  {
    id: "team_id",
    accessorKey: "team_id",
    meta: { title: t("logs.deletedTeams.columns.teamId") },
    header: t("logs.deletedTeams.columns.teamId"),
    size: 150,
    enableSorting: false,
    cell: ({ row }) => <IdCell value={row.original.team_id} variant="plain" />,
  },
  {
    id: "created_at",
    accessorKey: "created_at",
    meta: { title: t("logs.deletedTeams.columns.created") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("logs.deletedTeams.columns.created")} />,
    size: 120,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.created_at} precision="date" />,
  },
  {
    id: "spend",
    accessorKey: "spend",
    meta: { title: t("logs.deletedTeams.columns.spendUsd"), numeric: true },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("logs.deletedTeams.columns.spendUsd")} />,
    size: 100,
    enableSorting: true,
    cell: ({ row }) => <MoneyCell value={row.original.spend} decimals={4} />,
  },
  {
    id: "max_budget",
    accessorKey: "max_budget",
    meta: { title: t("logs.deletedTeams.columns.budgetUsd"), numeric: true },
    header: t("logs.deletedTeams.columns.budgetUsd"),
    size: 110,
    enableSorting: false,
    cell: ({ row }) => (
      <MoneyCell
        value={row.original.max_budget}
        decimals={0}
        emptyText={t("logs.deletedTeams.columns.unlimited")}
        showZero
      />
    ),
  },
  {
    id: "models",
    accessorKey: "models",
    meta: { title: t("logs.deletedTeams.columns.models"), skeleton: "chips" },
    header: t("logs.deletedTeams.columns.models"),
    size: 200,
    enableSorting: false,
    cell: ({ row }) => <ModelsCell models={row.original.models} />,
  },
  {
    id: "organization_id",
    accessorKey: "organization_id",
    meta: { title: t("logs.deletedTeams.columns.organization") },
    header: t("logs.deletedTeams.columns.organization"),
    size: 150,
    enableSorting: false,
    cell: ({ row }) => {
      const orgId = row.original.organization_id;
      return <EntityCell value={orgId} href={orgId ? orgDetailHref(orgId) : undefined} />;
    },
  },
  {
    id: "deleted_at",
    accessorKey: "deleted_at",
    meta: { title: t("logs.deletedTeams.columns.deletedAt") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("logs.deletedTeams.columns.deletedAt")} />,
    size: 120,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.deleted_at} precision="date" />,
  },
  {
    id: "deleted_by",
    accessorKey: "deleted_by",
    meta: { title: t("logs.deletedTeams.columns.deletedBy") },
    header: t("logs.deletedTeams.columns.deletedBy"),
    size: 120,
    enableSorting: false,
    cell: ({ row }) => {
      const deletedBy = row.original.deleted_by;
      return <EntityCell value={deletedBy} href={deletedBy ? userDetailHref(deletedBy) : undefined} />;
    },
  },
];
