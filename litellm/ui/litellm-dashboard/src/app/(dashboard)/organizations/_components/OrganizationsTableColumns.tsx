"use client";

import { ColumnDef } from "@tanstack/react-table";
import type { TFunction } from "i18next";
import { MoreHorizontal, Pencil, Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";

import { DataTableMultiSortHeader, DataTableSortHeader } from "@/components/shared/DataTable";
import { DateCell, IdentityCell, ModelsCell, MoneyCell } from "@/components/shared/table_cells";
import { Organization } from "@/components/networking";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";

interface OrganizationBudget {
  max_budget?: number | null;
  tpm_limit?: number | null;
  rpm_limit?: number | null;
}

const getOrganizationBudget = (organization: Organization): OrganizationBudget =>
  (organization.litellm_budget_table ?? {}) as OrganizationBudget;

function OrganizationLimitsCell({ organization }: { organization: Organization }) {
  const { t } = useTranslation();
  const { tpm_limit, rpm_limit } = getOrganizationBudget(organization);
  return (
    <div className="flex flex-col text-xs text-muted-foreground">
      <span>
        {t("organizations.table.limits.tpmLabel")}
        {tpm_limit ?? t("organizations.table.limits.unlimited")}
      </span>
      <span>
        {t("organizations.table.limits.rpmLabel")}
        {rpm_limit ?? t("organizations.table.limits.unlimited")}
      </span>
    </div>
  );
}

interface OrganizationRowActionsProps {
  organization: Organization;
  onEditClick: (organizationId: string) => void;
  onDeleteClick: (organizationId: string) => void;
}

function OrganizationRowActions({ organization, onEditClick, onDeleteClick }: OrganizationRowActionsProps) {
  const { t } = useTranslation();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t("organizations.table.actions.open")}
        data-testid={`organization-actions-${organization.organization_id}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem
          data-testid="organization-action-edit"
          onClick={() => onEditClick(organization.organization_id)}
        >
          <Pencil />
          {t("organizations.table.actions.edit")}
        </DropdownMenuItem>
        <DropdownMenuItem
          variant="destructive"
          data-testid="organization-action-delete"
          onClick={() => onDeleteClick(organization.organization_id)}
        >
          <Trash2 />
          {t("organizations.table.actions.delete")}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export interface OrganizationsTableColumnsDeps {
  userRole: string;
  onOrganizationClick: (organizationId: string) => void;
  onEditClick: (organizationId: string) => void;
  onDeleteClick: (organizationId: string) => void;
  t: TFunction;
}

export const getOrganizationsTableColumns = ({
  userRole,
  onOrganizationClick,
  onEditClick,
  onDeleteClick,
  t,
}: OrganizationsTableColumnsDeps): ColumnDef<Organization>[] => [
  {
    id: "organization_id",
    accessorKey: "organization_id",
    meta: { title: t("organizations.table.columns.id") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("organizations.table.columns.id")} />,
    size: 220,
    enableSorting: true,
    cell: ({ row }) => (
      <IdentityCell
        title={row.original.organization_id}
        titleClassName="font-mono text-xs font-normal"
        className="max-w-56"
        onClick={() => onOrganizationClick(row.original.organization_id)}
      />
    ),
  },
  {
    id: "organization_alias",
    accessorKey: "organization_alias",
    meta: { title: t("organizations.table.columns.name") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("organizations.table.columns.name")} />,
    size: 200,
    enableSorting: true,
    cell: ({ row }) => {
      const alias = row.original.organization_alias;
      return (
        <span className="block max-w-56 truncate text-sm font-medium" title={alias ?? undefined}>
          {alias || "-"}
        </span>
      );
    },
  },
  {
    id: "created_at",
    accessorKey: "created_at",
    sortingFn: "datetime",
    meta: { title: t("organizations.table.columns.created") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("organizations.table.columns.created")} />,
    size: 130,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.created_at} precision="date" />,
  },
  {
    id: "spend",
    accessorKey: "spend",
    meta: { title: t("organizations.table.columns.spend"), numeric: true },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("organizations.table.columns.spend")} />,
    size: 120,
    enableSorting: true,
    cell: ({ row }) => <MoneyCell value={row.original.spend} decimals={4} />,
  },
  {
    id: "max_budget",
    accessorFn: (organization) => getOrganizationBudget(organization).max_budget ?? Number.POSITIVE_INFINITY,
    sortUndefined: "last",
    sortDescFirst: true,
    meta: { title: t("organizations.table.columns.budget"), numeric: true },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("organizations.table.columns.budget")} />,
    size: 120,
    enableSorting: true,
    cell: ({ row }) => (
      <MoneyCell
        value={getOrganizationBudget(row.original).max_budget}
        decimals={2}
        emptyText={t("organizations.table.budgetUnlimited")}
        showZero
      />
    ),
  },
  {
    id: "models",
    accessorFn: (organization) =>
      organization.models?.length && !organization.models.includes("all-proxy-models")
        ? [...organization.models].sort().join(", ")
        : undefined,
    sortUndefined: "last",
    sortDescFirst: false,
    meta: { title: t("organizations.table.columns.models"), skeleton: "chips" },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("organizations.table.columns.models")} />,
    size: 260,
    enableSorting: true,
    cell: ({ row }) => <ModelsCell models={row.original.models} />,
  },
  {
    id: "limits",
    meta: { title: t("organizations.table.columns.limits") },
    header: ({ table }) => (
      <DataTableMultiSortHeader
        table={table}
        fields={[
          { id: "tpm_limit", label: t("organizations.table.columns.tpmSort") },
          { id: "rpm_limit", label: t("organizations.table.columns.rpmSort") },
        ]}
      />
    ),
    size: 150,
    enableSorting: false,
    cell: ({ row }) => <OrganizationLimitsCell organization={row.original} />,
  },
  {
    id: "members",
    accessorFn: (organization) => organization.members?.length ?? 0,
    sortUndefined: "last",
    sortDescFirst: true,
    meta: { title: t("organizations.table.columns.members"), numeric: true },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("organizations.table.columns.members")} />,
    size: 100,
    enableSorting: true,
    cell: ({ row }) => (
      <span className="text-sm">
        {t("organizations.table.memberCount", { count: row.original.members?.length ?? 0 })}
      </span>
    ),
  },
  ...(["tpm_limit", "rpm_limit"] as const).map(
    (field): ColumnDef<Organization> => ({
      id: field,
      accessorFn: (organization) => getOrganizationBudget(organization)[field] ?? Number.POSITIVE_INFINITY,
      sortUndefined: "last",
      sortDescFirst: true,
      enableHiding: false,
    }),
  ),
  {
    id: "actions",
    meta: { className: "text-right", headerClassName: "text-right" },
    header: () => <span className="sr-only">{t("organizations.table.columns.actions")}</span>,
    size: 64,
    enableSorting: false,
    enableHiding: false,
    cell: ({ row }) =>
      userRole === "Admin" ? (
        <div className="flex justify-end">
          <OrganizationRowActions organization={row.original} onEditClick={onEditClick} onDeleteClick={onDeleteClick} />
        </div>
      ) : null,
  },
];
